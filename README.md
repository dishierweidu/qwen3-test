# qwen3-omni-pretrain

Research implementation of a Qwen3-Omni-style pretraining pipeline.

## Project status

The repository now contains the legacy custom Thinker, the structure-aligned
Qwen3 reference path, a complete paper-inspired Qwen3.5-Omni runtime, and an
isolated MiMo-style tiny mechanism runtime. It is **not an exact reproduction
of any unavailable Qwen3.5-Omni checkpoint or an official MiMo checkpoint**:

- the unchanged legacy Stage-2 path still maps each image/audio item to one
  token;
- the reusable experimental path now has strict image, video, and audio
  sequence encoders, sequence assembly, Qwen3-disjoint positions, and an
  experimental TM-RoPE builder, but is not yet wired into the legacy Thinker;
- the Qwen3.5-inspired Talker, ARIA and Code2Wav path is a trainable/procedural
  prototype whose codec is explicitly a predecessor proxy;
- the experimental DeltaNet block is not an official Qwen3.5-Omni
  implementation.

The P0 correctness work in this branch remains the shared foundation beneath
the isolated Qwen3.5-inspired and MiMo-style mechanism profiles.

The architecture review and implementation roadmaps are checked in as:

- [Qwen3-Omni architecture gap](docs/research/2026-07-29-qwen3-omni-architecture-gap.md)
- [Qwen3.5-Omni reproduction plan](docs/research/2026-07-29-qwen3.5-omni-reproduction.md)
- [MiMo-V2.5 architecture lessons](docs/research/2026-07-29-mimo-v2.5-architecture-lessons.md)

## Architecture profiles

The profile registry exposes only implementations that can currently be built
or inspected. Compatibility labels describe the demonstrated boundary; a
structure-aligned, paper-inspired, or experimental profile is not
checkpoint-compatible with an official model.

| Profile | Status | Compatibility | Exact official checkpoint compatibility |
| --- | --- | --- | --- |
| `legacy_prototype` | Implemented local Thinker | `legacy-prototype` | No |
| `qwen3_omni_reference` | Pinned structure/runtime adapter | `structure-aligned` | No |
| `qwen35_omni_inspired` | Public-backbone paper prototype | `paper-inspired` | No |
| `mimo_v25_experimental` | Tiny SWA/MoE/MTP/EP experiment | `MiMo-style-experiment` | No |

Qwen3.5-inspired work remains non-exact unless a separate official profile is
introduced. The MiMo-style experiment does not claim an official MiMo model
type, repository identity, or checkpoint format. Both profiles have real lazy
factories and machine-readable manifests; their compatibility labels are a
hard boundary, not a quality claim.

New legacy saves use the non-colliding model type
`qwen3_omni_prototype`. Old saves that used `qwen3_omni_moe` are accepted only
through the explicit one-way legacy adapter, which emits a deprecation warning;
new files are never written with that old identity.

Profile validation is read-only and allocates no model:

```bash
.venv-prototype/bin/python -m qwen3_omni_pretrain.cli_profile validate \
  --profile legacy_prototype \
  --config-or-checkpoint configs/model/qwen3_omni_1_3b_moe.yaml
```

Inspection constructs the legacy Thinker on the meta device and reports its
manifest, layer topology, parameter counts, and unsupported capabilities:

```bash
.venv-prototype/bin/python -m qwen3_omni_pretrain.cli_profile inspect \
  --profile legacy_prototype \
  --config-or-checkpoint configs/model/qwen3_omni_1_3b_moe.yaml \
  --json
```

Both commands are offline by default. `--allow-network` is inspect-only and
must be supplied explicitly before a legacy tokenizer lookup may use the
network. Reference inspection returns a pinned oracle artifact with lazy
config and processor loaders, but the CLI invokes neither loader; it does not
load weights or claim an inference runtime. Unavailable tokenizer, layer, and
parameter metrics remain zero and are marked unsupported.

## Environment

Use Python 3.10 and keep the prototype and official-reference dependencies in
separate virtual environments. Do not use an ambient `python`: a mismatched
Torch/TorchVision/TorchAudio trio can import Python packages successfully but
fail later while loading compiled extensions.

### Prototype profile (CUDA 12.8)

```bash
python3.10 -m venv .venv-prototype
.venv-prototype/bin/python -m pip install --upgrade pip
.venv-prototype/bin/python -m pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-prototype/bin/python -m pip install \
  -c constraints/prototype-py310.txt -r requirements.txt -e .
```

For a CPU-only test host, replace the CUDA index URL with
`https://download.pytorch.org/whl/cpu`.

Verify both distribution metadata and compiled-extension imports:

```bash
.venv-prototype/bin/python - <<'PY'
import importlib.metadata as metadata
for package in ("torch", "torchvision", "torchaudio", "transformers"):
    print(f"{package}=={metadata.version(package)}")
import torch
import torchvision
import torchaudio
import transformers
PY

.venv-prototype/bin/python -m pytest -q
.venv-prototype/bin/python -m compileall -q src scripts tests
```

### Official Qwen3-Omni reference profile (CUDA 12.8)

The reference profile is intentionally separate because it pins a newer
Transformers release and the official multimodal utility package. FFmpeg must
also be available on `PATH`.

```bash
python3.10 -m venv .venv-qwen-reference
.venv-qwen-reference/bin/python -m pip install --upgrade pip
.venv-qwen-reference/bin/python -m pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-qwen-reference/bin/python -m pip install \
  -c constraints/qwen3-omni-reference-py310.txt \
  -r requirements-qwen3-omni-reference.txt -e .
```

Use `https://download.pytorch.org/whl/cpu` instead on CPU-only test hosts.
Verify the reference interpreter independently:

```bash
ffmpeg -version

.venv-qwen-reference/bin/python - <<'PY'
import importlib.metadata as metadata
for package in (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "qwen-omni-utils",
):
    print(f"{package}=={metadata.version(package)}")
import torch
import torchvision
import torchaudio
import transformers
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
assert callable(process_mm_info)
assert Qwen3OmniMoeForConditionalGeneration is not None
assert Qwen3OmniMoeProcessor is not None
PY

.venv-qwen-reference/bin/python -m pytest -q
.venv-qwen-reference/bin/python -m compileall -q src scripts tests
```

The reference profile is for architecture and API comparison. Installing its
dependencies does not make this custom implementation checkpoint-compatible
with official Qwen3-Omni models; the architectures and state dictionaries
remain different.

### Qwen3.5-inspired backbone profile

The Qwen3.5-inspired runtime requires the exact Torch family and Transformers
5.2.0 boundary recorded in `constraints/qwen35-backbone-py310.txt`. It can
reuse an already verified Qwen reference environment with those exact
distributions, or be installed in the separately ignored
`.venv-qwen35-backbone` environment:

```bash
python3.10 -m venv .venv-qwen35-backbone
.venv-qwen35-backbone/bin/python -m pip install \
  -r requirements-qwen35-backbone.txt

PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35 \
  tests/oracle/test_qwen35_public_config.py \
  tests/oracle/test_qwen35_gdn_oracle.py -q
```

The factory consumes a serialized `Qwen35InspiredConfig`. It binds the public
Qwen3.5-35B-A3B revision, keeps full KV plus GDN convolution/matrix state,
uses the 6.25 Hz offline AuT prototype and 160 ms timestamp policy, and builds
ARIA/Talker plus a mandatory predecessor-codec proxy. It deliberately reports
`streaming_audio_encoder=false`.

The checked-in tiny mechanism config can be validated in either environment
without allocating a model, and inspected from the prototype environment even
though that environment intentionally lacks Transformers 5.2.0:

```bash
.venv-prototype/bin/python -m qwen3_omni_pretrain.cli_profile validate \
  --profile qwen35_omni_inspired \
  --config-or-checkpoint configs/model/qwen35_omni_inspired_tiny.yaml

.venv-prototype/bin/python -m qwen3_omni_pretrain.cli_profile inspect \
  --profile qwen35_omni_inspired \
  --config-or-checkpoint configs/model/qwen35_omni_inspired_tiny.yaml \
  --json
```

Inspection returns an allocation-free config contract with zero parameter
counts and `allocation_free_inspection=true`; its capability flags describe
the buildable profile. Constructing the executable runtime still requires the
pinned Qwen3.5 environment.

### ABI troubleshooting

An error such as `undefined symbol` while loading `libtorchaudio.so` means the
installed TorchAudio wheel does not match the installed Torch release (and
often its CUDA build). Recreate the affected virtual environment and install
the exact Torch, TorchVision, and TorchAudio trio from one PyTorch wheel index.
Do not repair one profile by installing packages into the other profile or the
ambient interpreter.

## Tests

The tests cover Stage-2 target masking, strict media decoding, multimodal
sequence assembly and positions, multimodal prefill orchestration, MoE routing
scale, numerical fail-fast behavior, and parameter statistics. Run the
profile-specific commands above so test collection uses the intended
dependency set.

The MiMo-style mechanism suite and its two-rank EP smoke test run with the
prototype environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/hybrid_swa_moe tests/evaluation \
  tests/distributed/test_expert_parallel.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/torchrun \
  --standalone --nproc-per-node=2 \
  tests/distributed/run_expert_parallel_smoke.py
```

## MiMo-style mechanism experiments

`configs/model/hybrid_swa_moe_tiny.yaml` is a generic six-layer experiment:
five SWA layers and one full-attention layer, layer-0 dense SwiGLU, five
routed-only 8-expert/top-2 layers, a strict 128-token per-row SWA cache,
attention sink, one next-2 MTP head, corrected speculative verification, and
optional two-rank expert parallelism. It does not load `mimo_v2` weights or
claim 310B/1.02T-scale behavior.

All new benchmarks emit a validated `ExperimentReport` only after correctness
gates pass. Every supplied prompt/output/sequence/draft/batch value is measured
as a distinct report row; plural CLI arguments are not sampling hints:

```bash
.venv-prototype/bin/python scripts/benchmark_hybrid_attention.py \
  --config configs/model/hybrid_swa_moe_tiny.yaml \
  --attention-modes full swa --prompt-lengths 128 \
  --output-lengths 32 \
  --dtype float32 --device cpu

.venv-prototype/bin/python scripts/benchmark_moe.py \
  --config configs/model/hybrid_swa_moe_tiny.yaml \
  --experts 8 --top-k 2 --sequence-lengths 128

.venv-prototype/bin/python scripts/benchmark_mtp.py \
  --config configs/model/hybrid_swa_moe_tiny.yaml \
  --prompt-sources natural random --draft-lengths 1 3 --batch-sizes 1

.venv-prototype/bin/python scripts/benchmark_legacy_deltanet.py \
  --config configs/model/legacy_deltanet_tiny_benchmark.yaml \
  --sequence-lengths 128 --dtype float32 --device cpu
```

The ordinary full/SWA rows have different KV projection shapes, and the
legacy DeltaNet row uses another class with independently initialized weights;
both are absolute engineering baselines. Only a separate attention run with
`--paired-kv-heads 4` copies an exact name/shape parameter inventory and may be
interpreted as a tied-weight mechanism ablation. These reports are not MiMo
quality, checkpoint, or production-speed reproductions. The MTP field named
`end_to_end_speedup` is explicitly dimensioned as
`verification-only-synthetic` with `real_end_to_end_claim=false`; it is an
overhead sanity check, not a model-serving speed claim.

## Stage-2 media behavior

`OmniStage2Collator` treats an omitted image/audio path as a valid missing
modality. If a sample references a file that cannot be decoded, the default is
to raise `MediaLoadError` with the sample ID and path.

To quarantine bad media while auditing a corpus, explicitly enable:

```python
collator = OmniStage2Collator(tokenizer, skip_bad_media=True)
```

The collator then sets that modality's presence flag to zero and records a
structured entry in `_media_errors`. Underscore-prefixed keys are training
metadata and are removed before calling `model.forward`.

## Sequence-preserving multimodal prefill

New experimental profile runtimes should use
`MultimodalPrefillPipeline`. It owns only the image, video, and audio encoders;
the caller continues to own the text embedding and eventual decoder. A matched
expansion-policy/position-builder pair is an explicit caller choice:

- Qwen3-disjoint experiments pair `IdentityMediaExpansion` with
  `Qwen3DisjointPositionBuilder`;
- the 160 ms experiment pairs `TimestampInterleaveExpansion` with
  `TMRoPEPositionBuilder` configured with the same literal `0.16` second
  quantum.

Pass only the four model fields from `ProfileStage2Collator`. Diagnostics stay
with the data pipeline, so do not forward the complete mapping with
`**batch`:

```python
batch = profile_collator(samples)

pipeline = pipeline.to(device=device, dtype=dtype)
text_embedding = text_embedding.to(device=device, dtype=dtype)

output = pipeline.encode_and_assemble(
    input_ids=batch["input_ids"].to(device),
    attention_mask=batch["attention_mask"].to(device),
    labels=batch["labels"].to(device),
    decoded_media=batch["decoded_media"],
    text_embedding=text_embedding,
)
media_errors = batch["_media_errors"]
```

The orchestration boundary performs no implicit cast or device transfer for
text tensors, text embeddings, or encoded media sequences. All registered
encoder parameters and the external text embedding must already share one
device and floating dtype. The encoders alone may convert decoded floating
payloads to their parameter placement.

The common Qwen3-disjoint position implementation is guarded by pinned source
hashes and fixed-vector numerical tests. Full official joint audio/video
runtime parity still belongs to the separate Qwen3 reference-runtime adapter;
the cached official processor is an additional integration oracle. Passing the
common prefill tests demonstrates sequence semantics only—it does not imply
Qwen3, Qwen3.5, or MiMo checkpoint compatibility.

## Legacy incremental decode state

The legacy full-attention Thinker now implements typed `prefill()`/`decode()`
with request-owned, clone-detached KV snapshots. The cache stores RoPE-applied,
unrepeated K/V, uses one model-level rectangular `[B,1,Q,P+Q]` causal bias,
and keeps valid-token counts separate from storage-position cursors. The
legacy Stage-2 wrapper encodes present image/audio inputs exactly once during
prefill and records its fixed two-slot processed prefix; decode cannot receive
raw media.

The supported scope is intentionally narrow:

- standard all-MHA legacy Thinker and validated two-rank TP local KV shards;
- batch-one greedy generation with a pending-token checkpoint;
- no public streaming iterator, beam search, speculative rejection, or generic
  state truncation;
- legacy DeltaNet cache remains unsupported because its recurrent state has
  not been validated.

The generation engine creates a shared owner nonce for every TP group.
Programmatic TP callers that invoke `prefill()` directly must use
`model.create_state_owner(display_request_id)` on every rank; independently
calling `StateOwner.fresh()` per rank is rejected before embedding compute.

Inference selects the cache from the constructed layer topology. A hybrid
DeltaNet model fails before tokenizer/media/model compute unless the caller
explicitly opts into the old full-history loop:

```bash
.venv-prototype/bin/python -m qwen3_omni_pretrain.cli_infer_thinker \
  --stage stage1 --checkpoint /path/to/checkpoint \
  --prompt "你好" --num-beams 1

# Explicit compatibility path for an unsupported hybrid checkpoint:
.venv-prototype/bin/python -m qwen3_omni_pretrain.cli_infer_thinker \
  --stage stage1 --checkpoint /path/to/hybrid-checkpoint \
  --prompt "你好" --allow-uncached-fallback
```

The fallback emits one structured JSON warning per request and never changes
the declared capability to supported.

Run the correctness-gated tiny benchmark with:

```bash
.venv-prototype/bin/python scripts/benchmark_decode_cache.py \
  --config configs/model/legacy_full_attention_tiny.yaml \
  --prompt-length 8 --output-length 3 --warmup 1 --repetitions 2
```

Before recording any timing, the benchmark requires direct FP32 cached versus
uncached logit error `<= 1e-5` and exact greedy-token equality. Its JSON output
includes the architecture manifest, implementation commit, raw latency
samples, synchronization method, cache bytes by partition, peak memory and
fallback status.

## Numerical correctness

The training loop raises `NonFiniteTrainingError` if checked losses, logits, or
gradients contain NaN/Inf. Distributed ranks synchronize the failure flag so
all ranks stop at the same batch. Non-finite batches are not silently skipped.

## Checkpoint identity and recovery

New checkpoints publish `architecture.json` schema version 2 only after every
payload writer has finished. The sidecar records the training artifact
(`stage1-training` or `stage2-training`), the Thinker architecture summary, the
complete wrapper topology and state schema, the actual tokenizer length and
tokenizer serialization hash, and the implementation commit. Resume validates
this allocation-free identity before constructing a full model or reading
weights. Same-stage restores are strict; Stage-1-to-Stage-2 initialization is
an explicit Thinker-only transfer.

Accelerator, model-only, and tensor-parallel saves publish through a temporary
generation and atomically swap the destination, retaining the previous
generation as `.backup`. Resume tries a valid backup when the primary
generation is incomplete; explicit Stage-1-to-Stage-2 initialization applies
the same identity and strict-payload checks to its backup candidates.
Tensor-parallel checkpoints contain one immutable
`tp-shard-rank-XXXXX.pt` file per global rank plus `tp_shards.json`; restore
requires the complete shard set and matching world size, TP degree/rank, file
hash, and parameter schema.

In multi-rank training, SIGINT/SIGTERM is converted into an all-rank polling
decision before a coordinated emergency checkpoint. A unilateral raw Python
`KeyboardInterrupt` fails closed without starting a new collective because
peers may already be blocked inside another collective; single-process
`KeyboardInterrupt` still writes an emergency checkpoint.

## Parameter inspection

```bash
scripts/inspect_model_parameters.py \
  configs/model/qwen3_omni_1_3b_moe.yaml
```

Use `--json` for machine-readable output. The active-parameter value is an MoE
routing estimate, not measured FLOPs or activation memory. Tokenizer inspection
is offline and rejects repository code by default:

```bash
scripts/inspect_architecture.py \
  configs/model/qwen3_omni_1_3b_moe.yaml \
  --tokenizer /path/to/local/tokenizer --json
```

`--allow-network` permits tokenizer downloads and `--allow-remote-code` permits
repository code; these are independent opt-ins.

## Next architecture milestones

1. Extend the sequence-preserving prefill layer from the Qwen3.5-inspired
   runtime into the remaining text-only MiMo training and evaluation flows.
2. Replace paper-inspired media and predecessor-codec proxies only when exact
   official Qwen3.5-Omni artifacts become available.
3. Scale MiMo-style context, experts, and MTP depth only after correctness-
   gated quality, memory, and end-to-end speed reports justify each change.
4. Add production paging, encoder cache, DeepEP, and serving schedulers without
   weakening the profile identity or checkpoint-compatibility boundaries.
