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

## Architecture profiles

The profile registry exposes only implementations that can currently be built
or inspected. Compatibility labels describe the demonstrated boundary; a
structure-aligned, paper-inspired, or experimental profile is not
checkpoint-compatible with an official model.

| Profile | Status | Compatibility | Exact official checkpoint compatibility |
| --- | --- | --- | --- |
| `legacy_prototype` | Implemented local Thinker | `legacy-prototype` | No |
| `qwen3_omni_reference` | Pinned oracle/config implemented; runtime pending | `structure-aligned` | No |
| `qwen35_omni_inspired` | Planned | `paper-inspired` | No |
| `mimo_v25_experimental` | Planned | `MiMo-style-experiment` | No |

Qwen3.5-inspired work always remains non-exact unless a separate official
profile is introduced. The MiMo-style experiment does not claim an official
MiMo model type, repository identity, or checkpoint format. Planned profiles
are deliberately absent from the registry until they have real factories.

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

### ABI troubleshooting

An error such as `undefined symbol` while loading `libtorchaudio.so` means the
installed TorchAudio wheel does not match the installed Torch release (and
often its CUDA build). Recreate the affected virtual environment and install
the exact Torch, TorchVision, and TorchAudio trio from one PyTorch wheel index.
Do not repair one profile by installing packages into the other profile or the
ambient interpreter.

## Tests

The tests cover Stage-2 target masking, strict media decoding, multimodal
attention masks, MoE routing scale, numerical fail-fast behavior, and parameter
statistics. Run the profile-specific commands above so test collection uses the
intended dependency set.

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

1. Replace one-token adapters with sequence-preserving vision/audio encoders.
2. Add video and timestamp-aware TM-RoPE sequence construction.
3. Establish an official-structure-compatible Thinker baseline.
4. Implement Talker, codec MTP, Code2Wav, and streaming caches.
5. Evaluate Qwen3.5-style Hybrid Attention/ARIA and MiMo-style SWA/GA as
   separate experimental branches.
