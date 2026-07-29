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
assert callable(process_mm_info)
PY
```

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
