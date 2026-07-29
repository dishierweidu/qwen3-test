# qwen3-omni-pretrain

Research implementation of a Qwen3-Omni-style pretraining pipeline.

## Project status

The current repository contains a custom Thinker, MoE experiments, distributed
training utilities, and a Stage-2 image/audio adapter. It is **not yet an
architecture-faithful reproduction of the official Qwen3-Omni model**:

- the current image and audio adapters each produce one token;
- video/TM-RoPE are not implemented;
- Talker and Code2Wav are configuration placeholders;
- the experimental DeltaNet block is not an official Qwen3.5-Omni
  implementation.

The P0 correctness work in this branch stabilizes the existing baseline before
those modules are replaced.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
# Install the PyTorch build matching the host CUDA version first.
pip install -r requirements.txt
pip install -e .
```

Python 3.10 or newer is required.

## Tests

```bash
pytest -q
python -m compileall src scripts tests
```

The tests cover Stage-2 target masking, strict media decoding, multimodal
attention masks, MoE routing scale, numerical fail-fast behavior, and parameter
statistics.

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

## Numerical correctness

The training loop raises `NonFiniteTrainingError` if checked losses, logits, or
gradients contain NaN/Inf. Distributed ranks synchronize the failure flag so
all ranks stop at the same batch. Non-finite batches are not silently skipped.

## Parameter inspection

```bash
python scripts/inspect_model_parameters.py \
  configs/model/qwen3_omni_1_3b_moe.yaml
```

Use `--json` for machine-readable output. The active-parameter value is an MoE
routing estimate, not measured FLOPs or activation memory.

## Next architecture milestones

1. Replace one-token adapters with sequence-preserving vision/audio encoders.
2. Add video and timestamp-aware TM-RoPE sequence construction.
3. Establish an official-structure-compatible Thinker baseline.
4. Implement Talker, codec MTP, Code2Wav, and streaming caches.
5. Evaluate Qwen3.5-style Hybrid Attention/ARIA and MiMo-style SWA/GA as
   separate experimental branches.
