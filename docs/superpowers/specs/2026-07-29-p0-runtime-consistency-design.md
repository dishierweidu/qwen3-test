# P0 Runtime Consistency Design

## Status

Approved in conversation on 2026-07-29.

## Context

The completed P0 training-correctness work already covers target masking,
strict training-time media decoding, multimodal attention masks, synchronized
non-finite checks, routed-only MoE behavior, and parameter accounting. This
follow-up closes six remaining correctness gaps before any
architecture-faithful Qwen3-Omni work begins:

1. Stage-2 inference passes `labels=None`, while the multimodal wrapper
   unconditionally reads `labels.dtype`.
2. The Stage-2 YAML stores data-loader values under `data` and the seed under
   `train`, but the trainer reads different levels and hard-codes training
   shuffle to `True`.
3. Model YAML files contain missing or incorrect multimodal token IDs and do
   not validate them against the tokenizer.
4. Tensor-parallel MoE accepts the top-k renormalization setting but never
   applies it.
5. Stage-2 training raises on referenced invalid media, while inference
   silently converts the same input into an absent modality.
6. The installed PyTorch family can drift across binary-incompatible releases;
   the observed environment has `torch==2.10.0` and `torchaudio==2.9.1`, whose
   package metadata requires `torch==2.9.1`.

## Goal

Make the current research prototype deterministic and internally consistent
across Stage-2 training, Stage-2 inference, standard MoE, tensor-parallel MoE,
tokenizer/model configuration, and supported runtime environments.

## Non-goals

This change does not:

- replace the one-token vision or audio adapters;
- implement ViT, AuT, video input, TM-RoPE, Talker, MTP, Code2Wav, or streaming;
- change model dimensions, expert counts, or training recipes;
- make the project architecture-faithful or checkpoint-compatible with the
  official Qwen3-Omni model;
- claim to reproduce Qwen3.5-Omni;
- add unrelated trainer, packaging, or distributed-system refactors.

## Chosen approach

Use small shared contracts rather than independent point fixes or a complete
Stage-2 rewrite. The contracts are pure or narrowly stateful, can be tested
without a large model, and preserve existing model parameter names.

The rejected alternatives are:

1. **Patch each call site independently.** This is faster initially but leaves
   duplicated configuration, media, and routing semantics that can drift again.
2. **Rewrite the full Stage-2 processor and trainer.** This creates a larger
   regression surface and would be superseded by the later architecture rewrite.

## Component design

### 1. Stage-2 runtime configuration

Introduce a pure `normalize_stage2_config` boundary that accepts the current
nested mapping or the legacy flat `Stage2TrainConfig` representation and
returns one validated runtime dataclass.

Canonical ownership is:

- `model`: model configuration path;
- `data`: corpus paths, media roots, `batch_size`, `max_seq_length`,
  `num_workers`, `shuffle`, and `skip_bad_media`;
- `train`: output directory, seed, epochs, optimizer settings, accumulation,
  logging/evaluation/checkpoint cadence, precision, and resume settings.

Value precedence is fixed and testable:

1. explicit CLI override;
2. legacy top-level value;
3. canonical nested value;
4. documented default.

A legacy top-level value emits `DeprecationWarning`. If legacy and canonical
values conflict, the legacy value wins for backward compatibility and the
warning names both values. Required paths and model keys have no silent
defaults.

Normalization validates positive batch and sequence sizes, non-negative worker
counts, boolean shuffle/media policy, and mutual exclusion of FP16, BF16, and
FP8 before tokenizer or model allocation. The Stage-2 trainer consumes only
the normalized dataclass and passes its `shuffle`, worker, batch, sequence, and
seed values through unchanged.

### 2. Tokenizer-owned multimodal token IDs

`multimodal/tokenization/special_tokens.py` becomes the semantic registry. It
stores names and token strings, not numeric IDs:

| Config field | Token string | Current Qwen3 tokenizer ID |
| --- | --- | ---: |
| `image_token_id` | `<|image_pad|>` | 151655 |
| `video_token_id` | `<|video_pad|>` | 151656 |
| `audio_token_id` | `<|audio_pad|>` | 151675 |
| `audio_start_token_id` | `<|audio_start|>` | 151669 |
| `audio_end_token_id` | `<|audio_end|>` | 151670 |

The resolver runs after tokenizer load and before model construction or
inference. It rejects missing/unknown tokens, duplicate resolved IDs, IDs
outside `[0, config.vocab_size)`, and a tokenizer whose length exceeds the
embedding vocabulary. A padded embedding vocabulary remains valid, so
`config.vocab_size` does not have to equal `len(tokenizer)`.

Numeric multimodal IDs are removed from model YAML files. If a legacy YAML or
checkpoint config contains a different value, the resolver emits a warning and
replaces the value in memory. It never rewrites source YAML or checkpoint files
implicitly. A subsequently saved checkpoint serializes the corrected in-memory
configuration.

### 3. Shared top-k routing selection

Extract one stateless top-k selection function used by both
`Qwen3OmniMoeMLP` and `TensorParallelMoeMLP`. It accepts router probabilities,
`k`, and the renormalization flag and returns selected values and indices.

- Softmax, top-k selection, and optional renormalization use FP32.
- With renormalization enabled, selected values sum to one per token.
- With renormalization disabled, selected values retain their original sum.
- No epsilon clamp or rank-local numerical exception masks NaN/Inf. Invalid
  values continue into the existing synchronized numerical-diagnostic path.
- Expert modules, router modules, and their parameter names do not change, so
  state-dict keys remain compatible.

TP-specific expert execution, collectives, ZeRO synchronization, and dummy
expert forwards stay in the TP implementation; only route selection is shared.

### 4. Shared Stage-2 media loading

Extract the image/audio decode behavior behind one public media loader used by
the collator and inference CLI. The caller supplies a sample ID, modality, and
resolved optional path. The result contains the tensor, a presence bit, and an
optional structured error.

The policy is:

- no path supplied: valid missing modality, zero tensor, `present=0`, no error;
- referenced path missing, unreadable, or corrupt in strict mode:
  `MediaLoadError`;
- the same failure with `skip_bad_media=True`: zero tensor, `present=0`, and a
  structured error containing sample ID, modality, path, exception type, and
  message;
- valid media: decoded tensor and `present=1`.

Strict mode remains the default in both training and inference. Inference gains
an explicit `--skip_bad_media` flag and reports every structured error; it does
not use bare exception handlers or silently merge corrupt media with omitted
media.

### 5. Optional Stage-2 labels

The multimodal wrapper accepts `labels: Optional[torch.Tensor]`.

- During training, it creates a full label tensor and prepends `-100` for the
  image and audio prefix positions.
- During inference, it passes `labels=None` directly to the Thinker and does
  not allocate a label tensor.
- Both paths use the same multimodal attention-mask construction and return the
  Thinker output unchanged.

## Runtime data flow

### Training

1. Load YAML.
2. Normalize and validate Stage-2 runtime configuration.
3. Set the normalized seed.
4. Load tokenizer and resolve multimodal token IDs into the model config.
5. Construct model, datasets, shared media loader, collator, and data loaders.
6. Decode media under the normalized strict/skip policy.
7. Call the multimodal model with supervised labels.

### Inference

1. Parse CLI arguments.
2. Load tokenizer and checkpoint configuration.
3. Resolve and validate multimodal token IDs in memory.
4. Resolve media paths and decode them with the shared loader.
5. Call the multimodal model with `labels=None`.
6. Generate tokens and report any structured skipped-media diagnostics.

### MoE

1. Compute router logits and FP32 probabilities.
2. Use the shared top-k selector.
3. Dispatch through the standard or TP-specific expert path.
4. Preserve existing auxiliary-loss and synchronized non-finite diagnostics.

## Error handling

Configuration, token, and dependency errors should fail before model parameters
are moved to a GPU. Media failures identify the sample and path. Distributed
forward code does not introduce rank-local numerical raises. Compatibility
warnings are visible and actionable; they do not silently choose a value.

No source configuration or checkpoint is automatically overwritten during
normalization or token reconciliation.

## Environment profiles

The repository keeps common Python dependencies separate from exact environment
constraints. Installation documentation must create distinct virtual
environments and must not install one profile over the other.

### Prototype profile

Framework anchors:

- Python 3.10;
- `torch==2.10.0`;
- `torchvision==0.25.0`;
- `torchaudio==2.10.0`;
- `transformers==4.57.6`.

This profile runs the current custom training/inference implementation.

### Official Qwen3-Omni reference profile

Framework anchors:

- Python 3.10;
- `torch==2.10.0`;
- `torchvision==0.25.0`;
- `torchaudio==2.10.0`;
- `transformers==5.2.0`;
- `qwen-omni-utils==0.0.9`;
- FFmpeg available on `PATH`.

This profile is reserved for comparing against official Qwen3-Omni interfaces
and does not redefine the custom prototype as an official implementation.

Both profiles use exact constraint files. Installation selects an official
PyTorch wheel index appropriate to the host, with CUDA 12.8 documented for the
current machine and CPU installation documented for test-only hosts. The
resolved environment is recorded with a package-version smoke command.

## Test design

Every behavior change follows red-green-refactor: add a focused failing test,
observe the expected failure, implement the smallest correction, and rerun the
focused and regression suites.

### Model tests

- `labels=None` reaches the Thinker as `None` and returns logits.
- Training labels receive exactly two `-100` prefix positions.
- Training and inference construct the same attention-mask prefix.

### Configuration tests

- Canonical nested YAML maps every Stage-2 data/train field correctly.
- Legacy flat dataclass and legacy top-level mappings still load and warn.
- Conflicts obey the documented precedence and name both values in the warning.
- A DataLoader spy observes normalized batch size, sequence length, workers,
  and `shuffle=False` from the checked-in Stage-2 YAML.
- Invalid precision combinations and numeric ranges fail before model creation.

### Token tests

- All six model YAML files resolve through the checked-in Qwen3 tokenizer to
  `151655`, `151656`, `151675`, `151669`, and `151670`.
- Missing, unknown, duplicate, and out-of-range IDs fail.
- A legacy mismatched config warns and is corrected only in memory.
- A padded model vocabulary larger than the tokenizer remains valid.

### MoE tests

- Selected values sum to one when renormalization is enabled.
- They retain the pre-normalization sum when it is disabled.
- Standard and TP world-size-one paths produce equivalent routed output from
  identical router/expert weights within the configured numerical tolerance.
- State-dict keys are unchanged.
- Existing NaN/Inf propagation and synchronized diagnostics continue to pass.

### Media tests

Training and inference both cover:

- omitted optional media;
- a referenced nonexistent path;
- a referenced corrupt file;
- explicit skip mode and its structured error.

No test accepts silent media failure.

### Environment tests

In each clean profile:

- import `torch`, `torchvision`, `torchaudio`, and `transformers`;
- assert the framework anchor versions;
- run the project unit suite;
- run compile checks.

The prototype profile also runs a tiny Stage-2 one-token decode. The official
reference profile imports the official Qwen3-Omni processor and conditional
generation classes plus `qwen_omni_utils.process_mm_info`.

## Acceptance criteria

The change is complete only when:

1. `pytest -q` has no collection errors and all tests pass in both profiles.
2. `python -m compileall -q src scripts tests` passes.
3. PyTorch, TorchVision, and TorchAudio import together at matching releases.
4. A tiny Stage-2 inference step generates at least one token without a labels
   exception.
5. The checked-in Stage-2 YAML actually applies its seed, batch size, sequence
   length, worker count, and `shuffle: false`.
6. The Qwen3 tokenizer is the sole runtime source of multimodal numeric IDs.
7. Standard and TP MoE routing agree within test tolerance when configured
   equivalently.
8. Referenced invalid media is never silently treated as omitted media.
9. Existing state-dict parameter names remain unchanged.

## Compatibility and migration

- Existing flat Stage-2 configuration remains readable during this P0 change
  and produces deprecation warnings.
- Existing checkpoints remain loadable. Incorrect token IDs are corrected in
  memory with a warning and are persisted only when a new checkpoint is saved.
- Model dimensions and parameter names remain stable.
- The current one-token media representation remains unchanged.
- README installation and troubleshooting instructions identify the two
  environments and the Torch/TorchAudio ABI failure mode explicitly.
