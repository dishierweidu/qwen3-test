# P0 Runtime Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage-2 training, Stage-2 inference, tokenizer configuration, standard/TP MoE routing, media failure handling, and supported Python environments internally consistent.

**Architecture:** Add small shared contracts instead of rewriting Stage-2: one runtime-config normalizer, one tokenizer-owned multimodal token resolver, one media loader, and one top-k route selector. Keep model dimensions and state-dict parameter names stable, preserve legacy inputs through explicit warnings, and verify the prototype and official-reference environments independently.

**Tech Stack:** Python 3.10, PyTorch 2.10.0, TorchVision 0.25.0, TorchAudio 2.10.0, Transformers 4.57.6/5.2.0, PyYAML, pytest.

## Global Constraints

- Prototype framework anchors are exactly `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0`, and `transformers==4.57.6`.
- Official-reference anchors are exactly the same PyTorch trio plus `transformers==5.2.0` and `qwen-omni-utils==0.0.9`; FFmpeg must be on `PATH`.
- Use Python 3.10 for both profiles and keep their virtual environments separate.
- Do not replace the one-token vision/audio adapters or implement ViT, AuT, video, TM-RoPE, Talker, MTP, Code2Wav, or streaming.
- Do not change model dimensions, expert counts, training recipes, or existing state-dict parameter names.
- The Qwen3 tokenizer is the sole runtime authority for multimodal numeric token IDs.
- Strict media decoding is the default in both training and inference; skipping invalid referenced media must be explicit and observable.
- NaN/Inf in distributed forward paths must propagate to the existing synchronized diagnostic boundary instead of causing a rank-local raise.
- Every behavior change follows red-green-refactor and receives a focused commit.

---

## File responsibility map

- `constraints/prototype-py310.txt`: exact prototype framework anchors.
- `constraints/qwen3-omni-reference-py310.txt`: exact official-reference anchors.
- `requirements-qwen3-omni-reference.txt`: reference-only Qwen utility dependency layered over common requirements.
- `src/qwen3_omni_pretrain/training/stage2_config.py`: legacy/canonical Stage-2 normalization and validation only.
- `src/qwen3_omni_pretrain/multimodal/tokenization/special_tokens.py`: symbolic multimodal-token registry and tokenizer/config reconciliation only.
- `src/qwen3_omni_pretrain/data/collators.py`: shared media decoding contract plus Stage-2 batch collation.
- `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modules/moe.py`: shared route selection and standard routed experts.
- `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py`: TP-specific expert execution consuming shared route selection.
- `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py`: optional-label multimodal wrapper.
- `src/qwen3_omni_pretrain/training/trainer_thinker.py`: orchestration only; it consumes normalized config, reconciled tokens, and shared media loading.
- `src/qwen3_omni_pretrain/cli_infer_thinker.py`: CLI orchestration only; it consumes reconciled tokens and shared media loading.

---

### Task 1: Pin and validate the two runtime profiles

**Files:**
- Create: `constraints/prototype-py310.txt`
- Create: `constraints/qwen3-omni-reference-py310.txt`
- Create: `requirements-qwen3-omni-reference.txt`
- Create: `tests/test_dependency_profiles.py`
- Modify: `.gitignore`
- Modify: `README.md:20-41`

**Interfaces:**
- Consumes: the exact framework anchors in the approved design.
- Produces: two constraint-file paths used by all later verification commands.

- [ ] **Step 1: Write the failing profile-contract test**

```python
from pathlib import Path


def _pins(path: str) -> dict[str, str]:
    result = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        result[name.lower()] = version
    return result


def test_prototype_profile_has_matching_torch_family():
    pins = _pins("constraints/prototype-py310.txt")
    assert pins["torch"] == "2.10.0"
    assert pins["torchvision"] == "0.25.0"
    assert pins["torchaudio"] == "2.10.0"
    assert pins["transformers"] == "4.57.6"


def test_reference_profile_is_separate_and_qwen_capable():
    pins = _pins("constraints/qwen3-omni-reference-py310.txt")
    assert pins["torch"] == "2.10.0"
    assert pins["torchvision"] == "0.25.0"
    assert pins["torchaudio"] == "2.10.0"
    assert pins["transformers"] == "5.2.0"
    assert pins["qwen-omni-utils"] == "0.0.9"
```

- [ ] **Step 2: Run the contract test and observe the missing-file failure**

Run:

```bash
python -m pytest -q tests/test_dependency_profiles.py
```

Expected: FAIL with `FileNotFoundError` for `constraints/prototype-py310.txt`.

- [ ] **Step 3: Add the exact constraint files**

`constraints/prototype-py310.txt`:

```text
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
transformers==4.57.6
```

`constraints/qwen3-omni-reference-py310.txt`:

```text
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
transformers==5.2.0
qwen-omni-utils==0.0.9
```

`requirements-qwen3-omni-reference.txt`:

```text
-r requirements.txt
qwen-omni-utils==0.0.9
```

- [ ] **Step 4: Ignore the local profile environments**

Add this entry to `.gitignore`:

```gitignore
.venv-prototype/
.venv-qwen-reference/
```

- [ ] **Step 5: Document isolated installation and ABI diagnosis**

Replace the single-environment README instructions with explicit prototype and
reference commands. The prototype CUDA 12.8 commands must be:

```bash
python3.10 -m venv .venv-prototype
.venv-prototype/bin/python -m pip install --upgrade pip
.venv-prototype/bin/python -m pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-prototype/bin/python -m pip install \
  -c constraints/prototype-py310.txt -r requirements.txt -e .
```

Document the analogous `.venv-qwen-reference` command with
`requirements-qwen3-omni-reference.txt` and
`constraints/qwen3-omni-reference-py310.txt`. Include the CPU wheel index
`https://download.pytorch.org/whl/cpu` for test-only hosts. Record each
environment with its own interpreter; do not use ambient `python`:

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

- [ ] **Step 6: Run the profile test**

Run:

```bash
python -m pytest -q tests/test_dependency_profiles.py
```

Expected: `2 passed`.

- [ ] **Step 7: Create the prototype environment and prove imports**

Run the documented `.venv-prototype` installation commands, then:

```bash
.venv-prototype/bin/python - <<'PY'
import torch
import torchvision
import torchaudio
import transformers
assert torch.__version__.split("+", 1)[0] == "2.10.0"
assert torchvision.__version__.split("+", 1)[0] == "0.25.0"
assert torchaudio.__version__.split("+", 1)[0] == "2.10.0"
assert transformers.__version__ == "4.57.6"
PY
```

Expected: exit code 0 with no undefined-symbol error.

- [ ] **Step 8: Commit the runtime-profile contract**

```bash
git add .gitignore README.md constraints requirements-qwen3-omni-reference.txt tests/test_dependency_profiles.py
git commit -m "build: pin isolated runtime profiles"
```

---

### Task 2: Allow Stage-2 inference without labels

**Files:**
- Modify: `tests/test_multimodal_attention_mask.py`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py:1-121`

**Interfaces:**
- Consumes: the existing Thinker contract where `labels=None` means no causal-LM loss.
- Produces: `Qwen3OmniMoeThinkerVisionAudioModel.forward(input_ids, attention_mask, labels: Optional[Tensor], pixel_values, audio_values, has_image, has_audio, output_hidden_states=False, **kwargs)`.

- [ ] **Step 1: Add a failing inference-label test**

Append:

```python
def test_wrapper_passes_none_labels_during_inference(monkeypatch):
    class CapturingThinker(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, 4)
            self.received_labels = "not-called"

        def forward(self, **kwargs):
            self.received_labels = kwargs["labels"]
            return {"logits": torch.zeros(1, 3, 32)}

    monkeypatch.setattr(module, "Qwen3OmniMoeThinkerTextModel", CapturingThinker)
    config = module.Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": False,
            "moe_shared_expert": False,
        },
    )
    model = module.Qwen3OmniMoeThinkerVisionAudioModel(config)

    output = model(
        input_ids=torch.tensor([[3]], dtype=torch.long),
        attention_mask=torch.tensor([[1]], dtype=torch.long),
        labels=None,
        pixel_values=torch.zeros(1, 3, 8, 8),
        audio_values=torch.zeros(1, 32000),
        has_image=torch.tensor([0], dtype=torch.long),
        has_audio=torch.tensor([0], dtype=torch.long),
    )

    assert model.thinker.received_labels is None
    assert output["logits"].shape == (1, 3, 32)


def test_wrapper_uses_same_attention_prefix_with_and_without_labels(
    monkeypatch,
):
    class CapturingThinker(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, 4)
            self.masks = []

        def forward(self, **kwargs):
            self.masks.append(kwargs["attention_mask"].clone())
            return {"logits": torch.zeros(1, 3, 32)}

    monkeypatch.setattr(module, "Qwen3OmniMoeThinkerTextModel", CapturingThinker)
    config = module.Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": False,
            "moe_shared_expert": False,
        },
    )
    model = module.Qwen3OmniMoeThinkerVisionAudioModel(config)
    common = {
        "input_ids": torch.tensor([[3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1]], dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 8, 8),
        "audio_values": torch.zeros(1, 32000),
        "has_image": torch.tensor([1], dtype=torch.long),
        "has_audio": torch.tensor([0], dtype=torch.long),
    }

    model(labels=torch.tensor([[3]], dtype=torch.long), **common)
    model(labels=None, **common)

    assert model.thinker.masks[0].tolist() == [[1, 0, 1]]
    assert torch.equal(model.thinker.masks[0], model.thinker.masks[1])


def test_real_tiny_wrapper_runs_one_autoregressive_step_without_labels():
    config = module.Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": False,
            "moe_shared_expert": False,
        },
    )
    model = module.Qwen3OmniMoeThinkerVisionAudioModel(config).eval()
    media = {
        "pixel_values": torch.zeros(1, 3, 8, 8),
        "audio_values": torch.zeros(1, 32000),
        "has_image": torch.tensor([0], dtype=torch.long),
        "has_audio": torch.tensor([0], dtype=torch.long),
    }
    input_ids = torch.tensor([[3]], dtype=torch.long)
    first = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=None,
        **media,
    )
    next_token = first["logits"][:, -1].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=-1)
    second = model(
        input_ids=generated,
        attention_mask=torch.ones_like(generated),
        labels=None,
        **media,
    )

    assert generated.shape == (1, 2)
    assert second["logits"].shape == (1, 4, 32)
```

- [ ] **Step 2: Run the test and observe the existing crash**

Run:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_multimodal_attention_mask.py
```

Expected: the three new inference-facing tests FAIL with
`AttributeError: 'NoneType' object has no attribute 'dtype'`; existing
supervised tests remain green.

- [ ] **Step 3: Implement conditional label-prefix construction**

Import `Optional`, annotate `labels`, and replace unconditional construction
with:

```python
labels_full: Optional[torch.Tensor] = None
if labels is not None:
    labels_full = torch.full(
        (batch_size, inputs_embeds.size(1)),
        fill_value=-100,
        dtype=labels.dtype,
        device=device,
    )
    labels_full[:, 2:] = labels.to(device)
```

Pass `labels_full` to the Thinker exactly as before.

- [ ] **Step 4: Run focused and existing multimodal tests**

Run:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_multimodal_attention_mask.py
```

Expected: all tests pass, including the existing supervised-label assertion.

- [ ] **Step 5: Commit the optional-label fix**

```bash
git add tests/test_multimodal_attention_mask.py \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py
git commit -m "fix: allow label-free Stage2 inference"
```

---

### Task 3: Normalize Stage-2 configuration before initialization

**Files:**
- Create: `src/qwen3_omni_pretrain/training/stage2_config.py`
- Create: `tests/test_trainer_import_safety.py`
- Create: `tests/test_stage2_config.py`
- Create: `tests/test_stage2_runtime.py`
- Create: `tests/test_cli_train_thinker.py`
- Modify: `src/qwen3_omni_pretrain/training/trainer_thinker.py:76-142,1462-1760`
- Modify: `src/qwen3_omni_pretrain/cli_train_thinker.py`

**Interfaces:**
- Consumes: nested YAML mappings, legacy flat mappings/dataclasses, and optional CLI overrides.
- Produces:
  - `Stage2RuntimeConfig`
  - `normalize_stage2_config(raw, *, overrides=None) -> Stage2RuntimeConfig`
  - `_prepare_stage2_runtime(cfg, resume_from_checkpoint=None) -> Stage2RuntimeConfig`
  - `_build_stage2_collator(runtime, tokenizer) -> OmniStage2Collator`
  - `_build_stage2_dataloaders(runtime, train_dataset, val_dataset, collator) -> tuple[DataLoader, DataLoader]`
  - `_run_stage2_epoch(...) -> Stage2EpochResult`
  - `_stage2_cli_runtime(args) -> Stage2RuntimeConfig`

Active Stage-2 DDP is outside this P0. `train.ddp: true` must fail before
seeding, tokenizer load, model allocation, or distributed initialization.

- [ ] **Step 1: Add an import-safety regression for DeepSpeed**

Create `tests/test_trainer_import_safety.py`:

```python
import subprocess
import sys
import textwrap


def test_importing_trainer_does_not_initialize_deepspeed():
    script = textwrap.dedent(
        """
        import importlib.machinery
        import sys
        import types

        fake = types.ModuleType("deepspeed")
        fake.__spec__ = importlib.machinery.ModuleSpec(
            "deepspeed", loader=None
        )
        fake.__version__ = "0.0.test"

        def init_distributed(*args, **kwargs):
            raise RuntimeError("deepspeed init called during import")

        fake.init_distributed = init_distributed
        sys.modules["deepspeed"] = fake

        import qwen3_omni_pretrain.training.trainer_thinker
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2: Verify import collection succeeds and behavior fails**

```bash
.venv-prototype/bin/python -m pytest --collect-only -q \
  tests/test_trainer_import_safety.py
.venv-prototype/bin/python -m pytest -q \
  tests/test_trainer_import_safety.py
```

Expected: collection succeeds; the test FAILS and stderr contains
`deepspeed init called during import`.

- [ ] **Step 3: Remove the trainer import side effect**

Replace the top-level block with:

```python
try:
    import deepspeed
except ImportError:
    deepspeed = None
```

Remove the now-unused `datetime` import. Do not add a second eager initializer:
the Stage-1 DeepSpeed path already calls `deepspeed.initialize` at runtime.

- [ ] **Step 4: Run import safety green and commit it**

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_trainer_import_safety.py
git add src/qwen3_omni_pretrain/training/trainer_thinker.py \
  tests/test_trainer_import_safety.py
git commit -m "fix: defer DeepSpeed initialization until training"
```

- [ ] **Step 5: Characterize and move the legacy dataclass compatibly**

Start `tests/test_stage2_config.py` with a passing characterization of the
existing public constructor:

```python
from qwen3_omni_pretrain.training import trainer_thinker


def test_legacy_stage2_config_constructor_contract():
    cfg = trainer_thinker.Stage2TrainConfig(
        stage1_init_ckpt="outputs/stage1",
        model_config_path="model.yaml",
        train_corpus_path="train.jsonl",
        val_corpus_path="val.jsonl",
        image_root="images",
        audio_root="audio",
        output_dir="outputs/stage2",
    )
    assert cfg.batch_size == 2
    assert cfg.max_seq_length == 1024
    assert cfg.learning_rate == 1e-4
    assert cfg.eval_steps == 200
    assert cfg.seed == 42
```

Run it once against the existing class:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_config.py::test_legacy_stage2_config_constructor_contract
```

Then move the class to `stage2_config.py` and re-import it from the trainer.
Preserve the required path fields and use these defaults, including the new
runtime fields that later tests will exercise:

```python
@dataclass
class Stage2TrainConfig:
    stage1_init_ckpt: str
    model_config_path: str
    train_corpus_path: str
    val_corpus_path: str
    image_root: str
    audio_root: str
    output_dir: str
    resume_from_checkpoint: Optional[str] = None
    num_epochs: int = 1
    max_steps: int = -1
    batch_size: int = 2
    max_seq_length: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    num_workers: int = 4
    shuffle: bool = True
    skip_bad_media: bool = False
    seed: int = 42
    ddp: bool = False
    fp16: bool = False
    bf16: bool = False
    fp8: bool = False
    int8_optimizer: bool = False
```

Run the characterization again after the move. The existing constructor
contract must stay green; keep the trainer re-export so existing imports work.

- [ ] **Step 6: Add a complete canonical-mapping RED**

Extend `tests/test_stage2_config.py`. Retain the characterization above and
replace its import block with:

```python
from dataclasses import asdict

import pytest

from conftest import TinyTokenizer
from qwen3_omni_pretrain.training import stage2_config, trainer_thinker


def complete_nested_config():
    return {
        "experiment_name": "stage2-test",
        "stage1_init_ckpt": "outputs/stage1",
        "model": {"model_config_path": "model.yaml"},
        "data": {
            "train_corpus_path": "train.jsonl",
            "val_corpus_path": "val.jsonl",
            "image_root": "images",
            "audio_root": "audio",
            "batch_size": 7,
            "max_seq_length": 123,
            "num_workers": 0,
            "shuffle": False,
            "skip_bad_media": True,
        },
        "train": {
            "output_dir": "outputs/stage2",
            "resume_from_checkpoint": "resume",
            "num_epochs": 2,
            "max_steps": 12,
            "learning_rate": 1e-5,
            "weight_decay": 0.02,
            "warmup_ratio": 0.05,
            "gradient_accumulation_steps": 3,
            "logging_steps": 4,
            "eval_steps": 5,
            "save_steps": 6,
            "seed": 777,
            "ddp": False,
            "fp16": False,
            "bf16": True,
            "fp8": False,
            "int8_optimizer": False,
        },
    }


EXPECTED_CANONICAL = {
    "experiment_name": "stage2-test",
    "stage1_init_ckpt": "outputs/stage1",
    "model_config_path": "model.yaml",
    "train_corpus_path": "train.jsonl",
    "val_corpus_path": "val.jsonl",
    "image_root": "images",
    "audio_root": "audio",
    "output_dir": "outputs/stage2",
    "resume_from_checkpoint": "resume",
    "num_epochs": 2,
    "max_steps": 12,
    "batch_size": 7,
    "max_seq_length": 123,
    "learning_rate": 1e-5,
    "weight_decay": 0.02,
    "warmup_ratio": 0.05,
    "gradient_accumulation_steps": 3,
    "logging_steps": 4,
    "eval_steps": 5,
    "save_steps": 6,
    "num_workers": 0,
    "shuffle": False,
    "skip_bad_media": True,
    "seed": 777,
    "ddp": False,
    "fp16": False,
    "bf16": True,
    "fp8": False,
    "int8_optimizer": False,
}


def test_complete_nested_mapping():
    result = stage2_config.normalize_stage2_config(
        complete_nested_config()
    )
    assert asdict(result) == EXPECTED_CANONICAL
```

- [ ] **Step 7: Verify canonical RED, then implement the runtime dataclass**

Run collect-only first, then the test:

```bash
.venv-prototype/bin/python -m pytest --collect-only -q \
  tests/test_stage2_config.py
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_config.py::test_complete_nested_mapping
```

Expected: collection succeeds; the test FAILS inside its body with
`AttributeError` because `normalize_stage2_config` is absent.

Define the runtime contract:

```python
@dataclass(frozen=True)
class Stage2RuntimeConfig:
    experiment_name: str
    stage1_init_ckpt: str
    model_config_path: str
    train_corpus_path: str
    val_corpus_path: str
    image_root: str
    audio_root: str
    output_dir: str
    resume_from_checkpoint: Optional[str]
    num_epochs: int
    max_steps: int
    batch_size: int
    max_seq_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    num_workers: int
    shuffle: bool
    skip_bad_media: bool
    seed: int
    ddp: bool
    fp16: bool
    bf16: bool
    fp8: bool
    int8_optimizer: bool
```

Implement the first green as a direct nested mapping with the public
`overrides=None` parameter already present but intentionally unused. Map
`model_config_path` from `model`, the corpus/media/batch fields from `data`,
and every output/training/cadence/precision field from `train`. Use direct
lookups only; do not add compatibility, defaults, coercion, or validation in
this cycle.

- [ ] **Step 8: Run canonical mapping green**

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_config.py::test_complete_nested_mapping
```

Expected: PASS.

- [ ] **Step 9: Add precedence/default RED tests**

Append:

```python
def test_cli_wins_but_legacy_still_warns():
    raw = complete_nested_config()
    raw["batch_size"] = 11
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"batch_size.*legacy=11.*canonical=7.*"
            r"cli=13.*selected=cli"
        ),
    ):
        result = stage2_config.normalize_stage2_config(
            raw, overrides={"batch_size": 13}
        )
    assert result.batch_size == 13


def test_legacy_wins_over_canonical_and_warns():
    raw = complete_nested_config()
    raw["batch_size"] = 11
    with pytest.warns(
        DeprecationWarning,
        match=r"batch_size.*legacy=11.*canonical=7.*selected=legacy",
    ):
        result = stage2_config.normalize_stage2_config(raw)
    assert result.batch_size == 11


def test_false_and_zero_cli_overrides_are_not_dropped():
    result = stage2_config.normalize_stage2_config(
        complete_nested_config(),
        overrides={"shuffle": False, "num_workers": 0},
    )
    assert result.shuffle is False
    assert result.num_workers == 0


def test_documented_defaults_are_used():
    raw = complete_nested_config()
    for key in (
        "num_epochs",
        "max_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "seed",
        "ddp",
        "fp16",
        "bf16",
        "fp8",
        "int8_optimizer",
    ):
        raw["train"].pop(key)
    for key in (
        "batch_size",
        "max_seq_length",
        "num_workers",
        "shuffle",
        "skip_bad_media",
    ):
        raw["data"].pop(key)
    raw["train"].pop("resume_from_checkpoint")

    result = stage2_config.normalize_stage2_config(raw)

    assert result.resume_from_checkpoint is None
    assert result.num_epochs == 1
    assert result.max_steps == -1
    assert result.batch_size == 2
    assert result.max_seq_length == 1024
    assert result.learning_rate == 1e-4
    assert result.weight_decay == 0.01
    assert result.warmup_ratio == 0.03
    assert result.gradient_accumulation_steps == 1
    assert result.logging_steps == 10
    assert result.eval_steps == 200
    assert result.save_steps == 200
    assert result.num_workers == 4
    assert result.shuffle is True
    assert result.skip_bad_media is False
    assert result.seed == 42
    assert result.ddp is False
    assert result.fp16 is False
    assert result.bf16 is False
    assert result.fp8 is False
    assert result.int8_optimizer is False


def test_legacy_flat_dataclass_is_supported_with_warning():
    legacy = stage2_config.Stage2TrainConfig(
        stage1_init_ckpt="outputs/stage1",
        model_config_path="model.yaml",
        train_corpus_path="train.jsonl",
        val_corpus_path="val.jsonl",
        image_root="images",
        audio_root="audio",
        output_dir="outputs/stage2",
        batch_size=9,
    )
    with pytest.warns(DeprecationWarning, match="legacy top-level"):
        result = stage2_config.normalize_stage2_config(legacy)
    assert result.batch_size == 9
    assert result.experiment_name == "thinker_stage2"
```

- [ ] **Step 10: Verify precedence/default tests fail behaviorally**

```bash
.venv-prototype/bin/python -m pytest -q tests/test_stage2_config.py
```

Expected: canonical mapping stays green; new tests fail with missing warning,
wrong selected value, missing default, or unsupported dataclass behavior.

- [ ] **Step 11: Implement the four-tier selector**

Convert dataclasses with `asdict`, validate section mappings, and use:

```python
_MISSING = object()


def _select(raw, section, key, default=_MISSING, overrides=None):
    nested = raw.get(section, {})
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise TypeError(f"{section} must be a mapping")

    has_cli = (
        overrides is not None
        and key in overrides
        and overrides[key] is not None
    )
    has_legacy = key in raw
    canonical = nested.get(key, _MISSING)

    if has_legacy:
        details = [f"legacy={raw[key]!r}"]
        if canonical is not _MISSING:
            details.append(f"canonical={canonical!r}")
        if has_cli:
            details.append(f"cli={overrides[key]!r}")
        selected = "cli" if has_cli else "legacy"
        warnings.warn(
            f"legacy top-level {key} is deprecated "
            f"({', '.join(details)}, selected={selected})",
            DeprecationWarning,
            stacklevel=3,
        )

    if has_cli:
        return overrides[key]
    if has_legacy:
        return raw[key]
    if canonical is not _MISSING:
        return canonical
    if default is _MISSING:
        raise KeyError(
            f"missing required configuration value: {section}.{key}"
        )
    return default
```

Refactor every `model`, `data`, and `train` field through `_select`. Use the
defaults asserted in `test_documented_defaults_are_used`; keep
`experiment_name` and `stage1_init_ckpt` at their existing top-level canonical
locations. Preserve the trainer's existing
`experiment_name="thinker_stage2"` fallback so a legacy dataclass remains
normalizable; `stage1_init_ckpt` remains a required key with no silent default.

- [ ] **Step 12: Run precedence/default tests green**

```bash
.venv-prototype/bin/python -m pytest -q tests/test_stage2_config.py
```

Expected: all tests currently in the file pass.

- [ ] **Step 13: Add validation and unsupported-DDP RED tests**

Append parameterized range, strict-bool, precision, and early-failure cases:

```python
@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("train", "num_epochs", 0, "num_epochs must be positive"),
        ("train", "max_steps", 0, "max_steps must be -1 or positive"),
        ("train", "max_steps", -2, "max_steps must be -1 or positive"),
        ("data", "batch_size", 0, "batch_size must be positive"),
        ("data", "max_seq_length", 0, "max_seq_length must be positive"),
        ("data", "num_workers", -1, "num_workers must be non-negative"),
        ("train", "learning_rate", 0, "learning_rate must be positive"),
        ("train", "weight_decay", -0.1, "weight_decay must be non-negative"),
        ("train", "warmup_ratio", 1.1, "warmup_ratio must be between 0 and 1"),
        (
            "train",
            "gradient_accumulation_steps",
            0,
            "gradient_accumulation_steps must be positive",
        ),
        ("train", "logging_steps", 0, "logging_steps must be positive"),
        ("train", "eval_steps", -1, "eval_steps must be non-negative"),
        ("train", "save_steps", -1, "save_steps must be non-negative"),
    ],
)
def test_invalid_ranges_fail(section, key, value, message):
    raw = complete_nested_config()
    raw[section][key] = value
    with pytest.raises(ValueError, match=message):
        stage2_config.normalize_stage2_config(raw)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("data", "shuffle"),
        ("data", "skip_bad_media"),
        ("train", "ddp"),
        ("train", "fp16"),
        ("train", "bf16"),
        ("train", "fp8"),
        ("train", "int8_optimizer"),
    ],
)
def test_boolean_fields_reject_string_values(section, key):
    raw = complete_nested_config()
    raw[section][key] = "false"
    with pytest.raises(TypeError, match=rf"{key} must be a boolean"):
        stage2_config.normalize_stage2_config(raw)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "stage1_init_ckpt"),
        ("model", "model_config_path"),
        ("data", "train_corpus_path"),
        ("data", "val_corpus_path"),
        ("data", "image_root"),
        ("data", "audio_root"),
        ("train", "output_dir"),
    ],
)
def test_required_paths_reject_empty_strings(section, key):
    raw = complete_nested_config()
    target = raw if section is None else raw[section]
    target[key] = ""
    with pytest.raises(ValueError, match=rf"{key} must be a non-empty string"):
        stage2_config.normalize_stage2_config(raw)


def test_optional_resume_path_rejects_non_string_values():
    raw = complete_nested_config()
    raw["train"]["resume_from_checkpoint"] = 7
    with pytest.raises(
        TypeError,
        match="resume_from_checkpoint must be a string or None",
    ):
        stage2_config.normalize_stage2_config(raw)


def test_invalid_precision_combination_fails():
    raw = complete_nested_config()
    raw["train"]["fp16"] = True
    with pytest.raises(ValueError, match="Only one of fp16/bf16/fp8"):
        stage2_config.normalize_stage2_config(raw)


def test_cli_resume_wins_but_legacy_resume_still_warns(monkeypatch):
    from qwen3_omni_pretrain.training import trainer_thinker

    raw = complete_nested_config()
    raw["resume_from_checkpoint"] = "legacy"
    monkeypatch.setattr(trainer_thinker, "set_seed", lambda seed: None)
    with pytest.warns(
        DeprecationWarning,
        match=r"resume_from_checkpoint.*selected=cli",
    ):
        runtime = trainer_thinker._prepare_stage2_runtime(
            raw, "cli-checkpoint"
        )
    assert runtime.resume_from_checkpoint == "cli-checkpoint"


def test_normalized_runtime_is_an_idempotent_boundary_value():
    runtime = stage2_config.normalize_stage2_config(
        complete_nested_config()
    )
    assert stage2_config.normalize_stage2_config(runtime) is runtime


def test_stage2_ddp_fails_before_seed_or_tokenizer(monkeypatch):
    from qwen3_omni_pretrain.training import trainer_thinker

    raw = complete_nested_config()
    raw["train"]["ddp"] = True
    monkeypatch.setattr(
        trainer_thinker,
        "set_seed",
        lambda seed: pytest.fail("seed ran before config validation"),
    )
    monkeypatch.setattr(
        trainer_thinker.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail(
            "tokenizer allocated before config validation"
        ),
    )
    with pytest.raises(NotImplementedError, match="Stage2 ddp=true"):
        trainer_thinker.train_thinker_stage2(
            raw, tokenizer_name_or_path="unused"
        )
```

Create `tests/test_cli_train_thinker.py` to cover the real process entry:

```python
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from qwen3_omni_pretrain import cli_train_thinker as cli
from qwen3_omni_pretrain.training import distributed as distributed_module
from qwen3_omni_pretrain.training.stage2_config import (
    Stage2RuntimeConfig,
)


def cli_args(**updates):
    values = {
        "config": "stage2.yaml",
        "tokenizer_name_or_path": "tokenizer",
        "stage": "stage2",
        "tensorboard": False,
        "log_dir": "./runs",
        "resume_from_checkpoint": "cli-checkpoint",
        "local_rank": None,
        "deepspeed": None,
        "accelerator_config": None,
        "use_accelerator": False,
        "use_tensor_parallel": False,
        "tp_size": 1,
        "pp_size": 1,
    }
    values.update(updates)
    return Namespace(**values)


def checked_stage2_config():
    return yaml.safe_load(
        Path(
            "configs/train/stage2_omni_vision_audio.yaml"
        ).read_text(encoding="utf-8")
    )


def test_stage2_cli_rejects_ddp_before_any_distributed_context(
    monkeypatch,
):
    raw = checked_stage2_config()
    raw["train"]["ddp"] = True
    monkeypatch.setattr(cli, "parse_args", cli_args)
    monkeypatch.setattr(cli, "load_yaml", lambda path: raw)
    monkeypatch.setattr(
        cli,
        "distributed_context",
        lambda: pytest.fail("distributed context entered"),
    )
    monkeypatch.setattr(
        distributed_module,
        "distributed_tensor_parallel_context",
        lambda **kwargs: pytest.fail("TP context entered"),
    )
    monkeypatch.setattr(
        cli,
        "train_thinker_stage2",
        lambda **kwargs: pytest.fail("trainer entered"),
    )

    with pytest.raises(NotImplementedError, match="Stage2 ddp=true"):
        cli.main()


def test_stage2_cli_bypasses_context_and_passes_one_runtime(
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(cli, "parse_args", cli_args)
    monkeypatch.setattr(
        cli, "load_yaml", lambda path: checked_stage2_config()
    )
    monkeypatch.setattr(
        cli,
        "distributed_context",
        lambda: pytest.fail("distributed context entered"),
    )
    monkeypatch.setattr(
        cli,
        "train_thinker_stage2",
        lambda **kwargs: captured.update(kwargs),
    )

    cli.main()

    assert isinstance(captured["cfg"], Stage2RuntimeConfig)
    assert captured["cfg"].resume_from_checkpoint == "cli-checkpoint"
    assert captured["resume_from_checkpoint"] is None


@pytest.mark.parametrize(
    "updates",
    [
        {"use_accelerator": True},
        {"accelerator_config": "accelerate.yaml"},
        {"deepspeed": "deepspeed.json"},
        {"use_tensor_parallel": True, "tp_size": 2},
        {"pp_size": 2},
        {"local_rank": 0},
    ],
)
def test_stage2_cli_rejects_unsupported_execution_modes(
    monkeypatch, updates
):
    monkeypatch.setattr(
        cli, "parse_args", lambda: cli_args(**updates)
    )
    monkeypatch.setattr(
        cli, "load_yaml", lambda path: checked_stage2_config()
    )
    monkeypatch.setattr(
        cli,
        "distributed_context",
        lambda: pytest.fail("distributed context entered"),
    )
    monkeypatch.setattr(
        distributed_module,
        "distributed_tensor_parallel_context",
        lambda **kwargs: pytest.fail("TP context entered"),
    )
    monkeypatch.setattr(
        cli,
        "train_thinker_stage2",
        lambda **kwargs: pytest.fail("trainer entered"),
    )

    with pytest.raises(
        NotImplementedError,
        match="Stage2 distributed execution",
    ):
        cli.main()


def test_stage2_cli_rejects_launcher_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(cli, "parse_args", cli_args)
    monkeypatch.setattr(
        cli, "load_yaml", lambda path: checked_stage2_config()
    )
    monkeypatch.setattr(
        cli,
        "distributed_context",
        lambda: pytest.fail("distributed context entered"),
    )
    monkeypatch.setattr(
        distributed_module,
        "distributed_tensor_parallel_context",
        lambda **kwargs: pytest.fail("TP context entered"),
    )
    monkeypatch.setattr(
        cli,
        "train_thinker_stage2",
        lambda **kwargs: pytest.fail("trainer entered"),
    )

    with pytest.raises(
        NotImplementedError,
        match="Stage2 distributed execution",
    ):
        cli.main()
```

- [ ] **Step 14: Verify validation tests fail for the right reasons**

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_config.py tests/test_cli_train_thinker.py
```

Expected: prior tests stay green; new tests fail with `DID NOT RAISE`, the
wrong exception, or missing `_prepare_stage2_runtime` because the boundary
validation path is not implemented yet. CLI tests fail before their assertions
because the current entry enters a context or passes the raw mapping.

- [ ] **Step 15: Implement strict validation, then run the config suite green**

Use real booleans only:

```python
def _require_bool(name, value):
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value
```

Import `replace` from `dataclasses`. Make the normalized dataclass an
idempotent boundary value, extract validation into
`_validate_stage2_runtime`, and begin normalization with:

```python
if isinstance(raw, Stage2RuntimeConfig):
    effective = {
        key: value
        for key, value in (overrides or {}).items()
        if value is not None
    }
    if not effective:
        return _validate_stage2_runtime(raw)
    return _validate_stage2_runtime(replace(raw, **effective))
```

Validate all parameterized ranges, required non-empty path strings, optional
checkpoint strings, and precision mutual exclusion after constructing the
frozen runtime dataclass. End `_validate_stage2_runtime` with:

```python
if result.ddp:
    raise NotImplementedError(
        "Stage2 ddp=true is not implemented in this P0; "
        "set train.ddp=false"
    )
return result
```

Add the boundary helper:

```python
def _prepare_stage2_runtime(
    cfg,
    resume_from_checkpoint: Optional[str] = None,
) -> Stage2RuntimeConfig:
    overrides = {}
    if resume_from_checkpoint is not None:
        overrides["resume_from_checkpoint"] = resume_from_checkpoint
    runtime = normalize_stage2_config(cfg, overrides=overrides)
    set_seed(runtime.seed)
    return runtime
```

At the first executable line of `train_thinker_stage2`, call this helper.
Normalization therefore finishes before seeding or allocating the tokenizer.
Extend the function annotation to accept `Stage2RuntimeConfig` alongside the
legacy dataclass and mapping.

In `cli_train_thinker.py`, import `os`, `Stage2RuntimeConfig`, and
`normalize_stage2_config`, then add:

```python
def _stage2_cli_runtime(args) -> Stage2RuntimeConfig:
    launched_distributed = (
        "LOCAL_RANK" in os.environ
        or "RANK" in os.environ
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    unsupported = (
        launched_distributed
        or args.use_accelerator
        or args.accelerator_config is not None
        or args.deepspeed is not None
        or args.use_tensor_parallel
        or args.tp_size > 1
        or args.pp_size > 1
        or args.local_rank is not None
    )
    if unsupported:
        raise NotImplementedError(
            "Stage2 distributed execution is not implemented in this P0"
        )
    overrides = {}
    if args.resume_from_checkpoint is not None:
        overrides["resume_from_checkpoint"] = (
            args.resume_from_checkpoint
        )
    return normalize_stage2_config(
        load_yaml(args.config),
        overrides=overrides,
    )
```

Immediately after `args = parse_args()`, handle Stage 2 and return:

```python
if args.stage == "stage2":
    runtime = _stage2_cli_runtime(args)
    train_thinker_stage2(
        cfg=runtime,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        enable_tensorboard=args.tensorboard,
        log_dir=args.log_dir,
        resume_from_checkpoint=None,
    )
    return
```

Delete both older Stage-2 branches lower in `main`; leave all Stage-1 context,
Accelerate, DeepSpeed, and TP behavior unchanged. This ensures supported
single-process Stage 2 never enters `distributed_context`, while `ddp=true`
and launcher-only flags fail before any context initialization.

Run and commit the complete config contract:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_config.py tests/test_cli_train_thinker.py
git add src/qwen3_omni_pretrain/training/stage2_config.py \
  src/qwen3_omni_pretrain/training/trainer_thinker.py \
  src/qwen3_omni_pretrain/cli_train_thinker.py \
  tests/test_stage2_config.py tests/test_cli_train_thinker.py
git commit -m "fix: normalize complete Stage2 configuration"
```

- [ ] **Step 16: Add failing checked-in YAML propagation tests**

Append:

```python
def test_checked_in_yaml_reaches_seed_collator_and_dataloaders(monkeypatch):
    from qwen3_omni_pretrain.training import trainer_thinker
    from qwen3_omni_pretrain.utils.config_utils import load_yaml

    raw = load_yaml("configs/train/stage2_omni_vision_audio.yaml")
    calls = []
    seeds = []

    class SpyDataLoader:
        def __init__(self, dataset, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(trainer_thinker, "DataLoader", SpyDataLoader)
    monkeypatch.setattr(trainer_thinker, "set_seed", seeds.append)
    runtime = trainer_thinker._prepare_stage2_runtime(raw)
    collator = trainer_thinker._build_stage2_collator(
        runtime, TinyTokenizer()
    )
    trainer_thinker._build_stage2_dataloaders(
        runtime, object(), object(), collator
    )

    assert seeds == [42]
    assert runtime.max_steps == -1
    assert runtime.gradient_accumulation_steps == 8
    assert runtime.eval_steps == 200
    assert runtime.save_steps == 200
    assert runtime.ddp is False
    assert collator.max_seq_length == 1024
    assert calls[0]["batch_size"] == 2
    assert calls[0]["num_workers"] == 4
    assert calls[0]["shuffle"] is False
    assert calls[1]["shuffle"] is False
```

- [ ] **Step 17: Run propagation RED, implement the builders, and rerun**

First run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_config.py::test_checked_in_yaml_reaches_seed_collator_and_dataloaders
```

Expected: the test fails inside its body because `_build_stage2_collator` and
`_build_stage2_dataloaders` are absent; `_prepare_stage2_runtime` remains
green from Step 15.

Implement:

```python
def _build_stage2_collator(runtime, tokenizer):
    return OmniStage2Collator(
        tokenizer=tokenizer,
        max_seq_length=runtime.max_seq_length,
        skip_bad_media=runtime.skip_bad_media,
    )


def _build_stage2_dataloaders(
    runtime, train_dataset, val_dataset, collator
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=runtime.batch_size,
        shuffle=runtime.shuffle,
        num_workers=runtime.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=runtime.batch_size,
        shuffle=False,
        num_workers=runtime.num_workers,
        collate_fn=collator,
    )
    return train_loader, val_loader
```

Use these helpers as the only Stage-2 construction path. Preserve the output
directory timestamp suffix and replace every later ad-hoc config read with the
corresponding `runtime` field. Rerun the propagation test and the complete
config contract green, then commit:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_stage2_config.py
git add src/qwen3_omni_pretrain/training/trainer_thinker.py \
  tests/test_stage2_config.py
git commit -m "fix: propagate Stage2 runtime configuration"
```

- [ ] **Step 18: Add failing step-cadence and schedule tests**

Create `tests/test_stage2_runtime.py` with a complete runtime fixture copied
from `EXPECTED_CANONICAL`, then test the epoch driver:

```python
from dataclasses import replace

import pytest
import torch

from qwen3_omni_pretrain.training import trainer_thinker
from qwen3_omni_pretrain.training.stage2_config import Stage2RuntimeConfig


def runtime_config():
    return Stage2RuntimeConfig(
        experiment_name="stage2-test",
        stage1_init_ckpt="outputs/stage1",
        model_config_path="model.yaml",
        train_corpus_path="train.jsonl",
        val_corpus_path="val.jsonl",
        image_root="images",
        audio_root="audio",
        output_dir="outputs/stage2",
        resume_from_checkpoint="resume",
        num_epochs=2,
        max_steps=12,
        batch_size=7,
        max_seq_length=123,
        learning_rate=1e-5,
        weight_decay=0.02,
        warmup_ratio=0.05,
        gradient_accumulation_steps=3,
        logging_steps=4,
        eval_steps=5,
        save_steps=6,
        num_workers=0,
        shuffle=False,
        skip_bad_media=True,
        seed=777,
        ddp=False,
        fp16=False,
        bf16=True,
        fp8=False,
        int8_optimizer=False,
    )


def test_epoch_driver_applies_eval_save_and_max_steps(monkeypatch):
    runtime = replace(
        runtime_config(),
        max_steps=5,
        eval_steps=2,
        save_steps=3,
    )
    events = []
    model = torch.nn.Linear(1, 1)

    def fake_train_one_epoch(**kwargs):
        for step in range(1, 10):
            if kwargs["should_stop_fn"]():
                break
            kwargs["after_step_fn"](
                step=step,
                loss=1.0,
                batch_idx=step - 1,
                ce_loss=None,
                aux_loss=None,
                lr=1e-4,
                model=model,
            )
            if kwargs["should_stop_fn"]():
                break
        return 1.0

    losses = iter((0.5, 0.6))

    def evaluate_fn(step):
        events.append(("eval", step))
        model.eval()
        return next(losses)

    def checkpoint_fn(tag, epoch, step, best):
        events.append(("save", tag, epoch, step, best))

    monkeypatch.setattr(
        trainer_thinker, "train_one_epoch", fake_train_one_epoch
    )
    result = trainer_thinker._run_stage2_epoch(
        runtime=runtime,
        epoch=0,
        starting_global_step=0,
        best_val_loss=1.0,
        model=model,
        train_loader=object(),
        optimizer=object(),
        scheduler=object(),
        device=torch.device("cpu"),
        autocast_dtype=None,
        grad_scaler=None,
        evaluate_fn=evaluate_fn,
        checkpoint_fn=checkpoint_fn,
    )

    assert result.global_step == 5
    assert result.should_stop is True
    assert result.best_val_loss == 0.5
    assert ("eval", 2) in events
    assert ("eval", 4) in events
    assert any(event[1] == "best_step_2" for event in events if event[0] == "save")
    assert any(event[1] == "step_3" for event in events if event[0] == "save")
    assert model.training is True


def test_epoch_driver_resumes_global_step_and_stops_after_one_update(
    monkeypatch,
):
    runtime = replace(
        runtime_config(),
        max_steps=5,
        eval_steps=0,
        save_steps=0,
    )
    local_steps = []
    logged_steps = []

    def fake_train_one_epoch(**kwargs):
        for step in range(1, 4):
            if kwargs["should_stop_fn"]():
                break
            local_steps.append(step)
            kwargs["log_step_fn"](
                step=step,
                loss=1.0,
                batch_idx=step - 1,
                ce_loss=None,
                aux_loss=None,
                lr=1e-4,
            )
            kwargs["after_step_fn"](
                step=step,
                loss=1.0,
                batch_idx=step - 1,
                ce_loss=None,
                aux_loss=None,
                lr=1e-4,
                model=kwargs["model"],
            )
            if kwargs["should_stop_fn"]():
                break
        return 1.0

    monkeypatch.setattr(
        trainer_thinker, "train_one_epoch", fake_train_one_epoch
    )
    result = trainer_thinker._run_stage2_epoch(
        runtime=runtime,
        epoch=1,
        starting_global_step=4,
        best_val_loss=1.0,
        model=torch.nn.Linear(1, 1),
        train_loader=object(),
        optimizer=object(),
        scheduler=object(),
        device=torch.device("cpu"),
        autocast_dtype=None,
        grad_scaler=None,
        evaluate_fn=lambda step: pytest.fail("evaluation must be disabled"),
        checkpoint_fn=lambda *args: pytest.fail(
            "periodic checkpoints must be disabled"
        ),
        log_step_fn=lambda **event: logged_steps.append(event["step"]),
    )

    assert local_steps == [1]
    assert logged_steps == [5]
    assert result.global_step == 5
    assert result.should_stop is True


def test_schedule_shape_uses_ceil_and_max_steps():
    runtime = replace(
        runtime_config(),
        num_epochs=2,
        max_steps=5,
        gradient_accumulation_steps=2,
    )
    assert trainer_thinker._stage2_schedule_shape(
        runtime, num_batches=5
    ) == (3, 5, 2)
```

Keep this fixture literal in the cadence test; do not import a production
builder to calculate expected values.

- [ ] **Step 19: Verify cadence tests fail inside test bodies**

```bash
.venv-prototype/bin/python -m pytest --collect-only -q \
  tests/test_stage2_runtime.py
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_runtime.py
```

Expected: collection succeeds; tests FAIL with missing
`_run_stage2_epoch`/`_stage2_schedule_shape`.

- [ ] **Step 20: Implement step cadence without changing the generic loop**

Add:

```python
@dataclass(frozen=True)
class Stage2EpochResult:
    train_loss: float
    global_step: int
    best_val_loss: float
    should_stop: bool


def _stage2_schedule_shape(runtime, num_batches):
    updates_per_epoch = max(
        1,
        math.ceil(
            int(num_batches) / runtime.gradient_accumulation_steps
        ),
    )
    if runtime.max_steps > 0:
        total_updates = runtime.max_steps
        effective_epochs = max(
            runtime.num_epochs,
            math.ceil(runtime.max_steps / updates_per_epoch),
        )
    else:
        total_updates = updates_per_epoch * runtime.num_epochs
        effective_epochs = runtime.num_epochs
    return updates_per_epoch, total_updates, effective_epochs


def _run_stage2_epoch(
    *,
    runtime,
    epoch,
    starting_global_step,
    best_val_loss,
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    autocast_dtype,
    grad_scaler,
    evaluate_fn,
    checkpoint_fn,
    log_step_fn=None,
):
    state = {
        "global_step": int(starting_global_step),
        "best_val_loss": float(best_val_loss),
    }

    def should_stop():
        return (
            runtime.max_steps > 0
            and state["global_step"] >= runtime.max_steps
        )

    def on_log(**event):
        if log_step_fn is None:
            return
        event["step"] = starting_global_step + event["step"]
        log_step_fn(**event)

    def after_step(**event):
        global_step = starting_global_step + event["step"]
        state["global_step"] = global_step

        if runtime.eval_steps > 0 and global_step % runtime.eval_steps == 0:
            try:
                val_loss = evaluate_fn(global_step)
            finally:
                model.train()
            if val_loss < state["best_val_loss"]:
                state["best_val_loss"] = val_loss
                checkpoint_fn(
                    f"best_step_{global_step}",
                    epoch,
                    global_step,
                    state["best_val_loss"],
                )

        if runtime.save_steps > 0 and global_step % runtime.save_steps == 0:
            checkpoint_fn(
                f"step_{global_step}",
                epoch,
                global_step,
                state["best_val_loss"],
            )

    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=(
            runtime.gradient_accumulation_steps
        ),
        log_step_fn=on_log if log_step_fn is not None else None,
        autocast_dtype=autocast_dtype,
        grad_scaler=grad_scaler,
        after_step_fn=after_step,
        should_stop_fn=should_stop,
    )
    return Stage2EpochResult(
        train_loss=train_loss,
        global_step=state["global_step"],
        best_val_loss=state["best_val_loss"],
        should_stop=should_stop(),
    )
```

The driver deliberately composes the existing `train_one_epoch`
`after_step_fn` and `should_stop_fn` hooks; it does not duplicate the generic
training loop.

Use `_stage2_schedule_shape` for scheduler length/effective epochs and call
`_run_stage2_epoch` from the outer Stage-2 loop. Preserve epoch-end
best/latest saves, update resume state from the returned result, and break the
outer loop after the current epoch-end bookkeeping when `should_stop` is true.
Refactor the trainer's logging callback so its `step` argument is already the
global step supplied by `_run_stage2_epoch`; remove the old
`step_offset + step` calculation and update the outer `global_step` only from
`Stage2EpochResult`.
Do not modify `training/loop.py` and do not add Stage-2 distributed code.

- [ ] **Step 21: Run cadence and all Stage-2 config regressions**

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_trainer_import_safety.py \
  tests/test_cli_train_thinker.py \
  tests/test_stage2_config.py \
  tests/test_stage2_runtime.py \
  tests/test_stage2_collator.py
```

Expected: all tests pass, including exact max-step stop, periodic events,
resume-safe global step calculation, and checked-in YAML propagation.

- [ ] **Step 22: Commit Stage-2 cadence**

```bash
git add src/qwen3_omni_pretrain/training/trainer_thinker.py \
  tests/test_stage2_runtime.py
git commit -m "fix: honor Stage2 step cadence"
```

---

### Task 4: Make the tokenizer authoritative for multimodal IDs

**Files:**
- Modify: `src/qwen3_omni_pretrain/multimodal/tokenization/special_tokens.py`
- Create: `tests/test_special_tokens.py`
- Create: `tests/test_stage2_token_wiring.py`
- Modify: `configs/model/qwen3_omni_1_3b_moe.yaml`
- Modify: `configs/model/qwen3_omni_7b_moe.yaml`
- Modify: `configs/model/qwen3_omni_30b_moe.yaml`
- Modify: `configs/model/qwen3_omni_70b_moe.yaml`
- Modify: `src/qwen3_omni_pretrain/training/trainer_thinker.py`
- Modify: `src/qwen3_omni_pretrain/cli_infer_thinker.py`

**Interfaces:**
- Consumes: a tokenizer with `get_vocab()`/`__len__()` and a model config with `vocab_size`.
- Produces:
  - `MULTIMODAL_SPECIAL_TOKENS: Mapping[str, str]`
  - `reconcile_multimodal_token_ids(config, tokenizer) -> dict[str, int]`
  - `_build_reconciled_stage2_model(config, tokenizer)`
  - `_load_reconciled_stage2_model(checkpoint, tokenizer, load_kwargs=None)`

- [ ] **Step 1: Add basic resolver and config-persistence tests**

Create `tests/test_special_tokens.py`. Import the existing module, not the
missing symbol, so pytest collection succeeds:

```python
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.multimodal.tokenization import (
    special_tokens as subject,
)


TOKEN_IDS = {
    "<|image_pad|>": 1,
    "<|video_pad|>": 2,
    "<|audio_pad|>": 3,
    "<|audio_start|>": 4,
    "<|audio_end|>": 5,
}
EXPECTED_IDS = {
    "image_token_id": 1,
    "video_token_id": 2,
    "audio_token_id": 3,
    "audio_start_token_id": 4,
    "audio_end_token_id": 5,
}


class FakeTokenizer:
    def __init__(self, vocab=None, length=6):
        self._vocab = dict(TOKEN_IDS if vocab is None else vocab)
        self._length = int(length)

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self):
        return self._length


def tiny_config(**kwargs):
    return Qwen3OmniMoeConfig(
        vocab_size=16,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        **kwargs,
    )


def test_happy_path_uses_tokenizer_ids_without_resizing_vocab():
    config = tiny_config()
    before_vocab_size = config.vocab_size

    resolved = subject.reconcile_multimodal_token_ids(
        config, FakeTokenizer()
    )

    assert resolved == EXPECTED_IDS
    assert {
        field: getattr(config, field) for field in EXPECTED_IDS
    } == EXPECTED_IDS
    assert config.vocab_size == before_vocab_size


def test_reconciled_ids_survive_config_round_trip(tmp_path):
    config = tiny_config()
    subject.reconcile_multimodal_token_ids(config, FakeTokenizer())

    config.save_pretrained(tmp_path)
    reloaded = Qwen3OmniMoeConfig.from_pretrained(
        tmp_path, local_files_only=True
    )

    assert {
        field: getattr(reloaded, field) for field in EXPECTED_IDS
    } == EXPECTED_IDS
    assert reloaded.vocab_size == 16
```

- [ ] **Step 2: Verify the basic resolver tests fail inside test bodies**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_special_tokens.py::test_happy_path_uses_tokenizer_ids_without_resizing_vocab \
  tests/test_special_tokens.py::test_reconciled_ids_survive_config_round_trip
```

Expected: both tests are collected and FAIL with `AttributeError` at
`subject.reconcile_multimodal_token_ids`.

- [ ] **Step 3: Implement only the symbolic registry and happy path**

```python
from __future__ import annotations

from types import MappingProxyType
from typing import Any


MULTIMODAL_SPECIAL_TOKENS = MappingProxyType(
    {
        "image_token_id": "<|image_pad|>",
        "video_token_id": "<|video_pad|>",
        "audio_token_id": "<|audio_pad|>",
        "audio_start_token_id": "<|audio_start|>",
        "audio_end_token_id": "<|audio_end|>",
    }
)


def reconcile_multimodal_token_ids(
    config: Any,
    tokenizer: Any,
) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    resolved = {
        field: int(vocab[token])
        for field, token in MULTIMODAL_SPECIAL_TOKENS.items()
    }
    for field, token_id in resolved.items():
        setattr(config, field, token_id)
    return resolved
```

Do not add tokens, resize embeddings, or change `config.vocab_size`.

- [ ] **Step 4: Run the two tests green and commit the basic resolver**

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_special_tokens.py::test_happy_path_uses_tokenizer_ids_without_resizing_vocab \
  tests/test_special_tokens.py::test_reconciled_ids_survive_config_round_trip
git add src/qwen3_omni_pretrain/multimodal/tokenization/special_tokens.py \
  tests/test_special_tokens.py
git commit -m "feat: resolve multimodal IDs from tokenizer"
```

- [ ] **Step 5: Add edge-validation tests before validation code**

Append:

```python
def test_missing_token_is_rejected_without_partial_mutation():
    config = tiny_config(image_token_id=9)
    tokenizer = FakeTokenizer({"<|image_pad|>": 1})

    with pytest.raises(ValueError, match="missing required multimodal token"):
        subject.reconcile_multimodal_token_ids(config, tokenizer)

    assert config.image_token_id == 9


def test_duplicate_ids_are_rejected():
    tokenizer = FakeTokenizer(
        {token: 1 for token in TOKEN_IDS},
        length=2,
    )
    with pytest.raises(ValueError, match="distinct IDs"):
        subject.reconcile_multimodal_token_ids(tiny_config(), tokenizer)


def test_out_of_range_id_is_rejected():
    vocab = dict(TOKEN_IDS)
    vocab["<|audio_end|>"] = 16
    with pytest.raises(ValueError, match="outside model vocab_size"):
        subject.reconcile_multimodal_token_ids(
            tiny_config(), FakeTokenizer(vocab, length=6)
        )


def test_tokenizer_larger_than_embedding_is_rejected():
    with pytest.raises(ValueError, match="exceeds model vocab_size"):
        subject.reconcile_multimodal_token_ids(
            tiny_config(), FakeTokenizer(length=17)
        )


def test_legacy_mismatch_warns_and_tokenizer_wins():
    config = tiny_config(image_token_id=9)
    with pytest.warns(RuntimeWarning, match="image_token_id"):
        resolved = subject.reconcile_multimodal_token_ids(
            config, FakeTokenizer()
        )
    assert config.image_token_id == resolved["image_token_id"] == 1


def test_padded_embedding_vocab_remains_valid():
    config = tiny_config()
    subject.reconcile_multimodal_token_ids(
        config, FakeTokenizer(length=6)
    )
    assert config.vocab_size == 16
```

- [ ] **Step 6: Verify edge tests expose the basic implementation**

Run:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_special_tokens.py
```

Expected: the two basic tests and padded-vocabulary characterization stay
green; missing-token fails with `KeyError`, and duplicate/range/length/warning
tests fail their assertions.

- [ ] **Step 7: Implement atomic validation and legacy warnings**

Add `warnings` and replace the resolver body:

```python
def reconcile_multimodal_token_ids(
    config: Any,
    tokenizer: Any,
) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    missing = [
        token
        for token in MULTIMODAL_SPECIAL_TOKENS.values()
        if token not in vocab
    ]
    if missing:
        raise ValueError(
            f"tokenizer is missing required multimodal token {missing[0]!r}"
        )

    resolved = {
        field: int(vocab[token])
        for field, token in MULTIMODAL_SPECIAL_TOKENS.items()
    }
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(
            "multimodal special tokens must resolve to distinct IDs"
        )

    vocab_size = int(config.vocab_size)
    if len(tokenizer) > vocab_size:
        raise ValueError(
            f"tokenizer length {len(tokenizer)} exceeds "
            f"model vocab_size {vocab_size}"
        )
    invalid = {
        field: token_id
        for field, token_id in resolved.items()
        if not 0 <= token_id < vocab_size
    }
    if invalid:
        field, token_id = next(iter(invalid.items()))
        raise ValueError(
            f"{field}={token_id} is outside model vocab_size={vocab_size}"
        )

    for field, token_id in resolved.items():
        current = getattr(config, field, None)
        if current is not None and int(current) != token_id:
            warnings.warn(
                f"{field}={current} disagrees with tokenizer ID "
                f"{token_id}; using tokenizer value in memory",
                RuntimeWarning,
                stacklevel=2,
            )
    for field, token_id in resolved.items():
        setattr(config, field, token_id)
    return resolved
```

Validation completes before the first `setattr`, so a failure cannot partially
mutate the config.

- [ ] **Step 8: Run edge tests green and commit validation**

```bash
.venv-prototype/bin/python -m pytest -q tests/test_special_tokens.py
git add src/qwen3_omni_pretrain/multimodal/tokenization/special_tokens.py \
  tests/test_special_tokens.py
git commit -m "fix: validate multimodal tokenizer IDs atomically"
```

- [ ] **Step 9: Add failing model-YAML ownership tests**

Append:

```python
MODEL_CONFIGS = sorted(
    Path("configs/model").glob("qwen3_omni_*_moe.yaml")
)
EXPECTED_TOKEN_STRINGS = {
    "image_token_id": "<|image_pad|>",
    "video_token_id": "<|video_pad|>",
    "audio_token_id": "<|audio_pad|>",
    "audio_start_token_id": "<|audio_start|>",
    "audio_end_token_id": "<|audio_end|>",
}


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return AutoTokenizer.from_pretrained(
        "src/tokenizer/Qwen3",
        local_files_only=True,
        use_fast=True,
    )


@pytest.mark.parametrize("config_path", MODEL_CONFIGS, ids=lambda p: p.stem)
def test_model_yaml_defers_numeric_ids_to_tokenizer(
    config_path, qwen3_tokenizer
):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert set(EXPECTED_TOKEN_STRINGS).isdisjoint(raw)

    config = Qwen3OmniMoeConfig(**raw)
    resolved = subject.reconcile_multimodal_token_ids(
        config, qwen3_tokenizer
    )
    vocab = qwen3_tokenizer.get_vocab()
    assert resolved == {
        field: int(vocab[token])
        for field, token in EXPECTED_TOKEN_STRINGS.items()
    }
```

- [ ] **Step 10: Run the YAML tests red, then remove numeric IDs**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_special_tokens.py::test_model_yaml_defers_numeric_ids_to_tokenizer
```

Expected: four parameter cases fail the disjoint assertion; the 14B and 120B
cases pass. Then delete the five numeric ID fields from only the 1.3B, 7B,
30B, and 70B YAML files.

- [ ] **Step 11: Run YAML tests green and commit ownership cleanup**

```bash
.venv-prototype/bin/python -m pytest -q tests/test_special_tokens.py
git add configs/model tests/test_special_tokens.py
git commit -m "refactor: defer multimodal IDs to tokenizer"
```

- [ ] **Step 12: Add failing training and inference ordering tests**

Create `tests/test_stage2_token_wiring.py`:

```python
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import yaml

from qwen3_omni_pretrain import cli_infer_thinker as cli
from qwen3_omni_pretrain.training import trainer_thinker


class WiringReached(RuntimeError):
    pass


def test_training_builder_reconciles_before_model_construction(monkeypatch):
    events = []
    config = SimpleNamespace(vocab_size=16, image_token_id=None)
    tokenizer = object()

    def reconcile(config_arg, tokenizer_arg):
        events.append("reconcile")
        assert config_arg is config
        assert tokenizer_arg is tokenizer
        config_arg.image_token_id = 7
        return {"image_token_id": 7}

    class SpyModel:
        def __init__(self, config_arg):
            events.append("construct")
            assert config_arg is config
            assert config_arg.image_token_id == 7

    monkeypatch.setattr(
        trainer_thinker,
        "reconcile_multimodal_token_ids",
        reconcile,
        raising=False,
    )
    monkeypatch.setattr(
        trainer_thinker,
        "Qwen3OmniMoeThinkerVisionAudioModel",
        SpyModel,
    )

    model = trainer_thinker._build_reconciled_stage2_model(
        config, tokenizer
    )
    assert isinstance(model, SpyModel)
    assert events == ["reconcile", "construct"]


def test_training_entry_calls_reconciled_builder(
    monkeypatch, tmp_path
):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 16,
                "thinker_config": {
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 16,
                    "use_moe": False,
                    "moe_shared_expert": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trainer_thinker.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        trainer_thinker,
        "_build_reconciled_stage2_model",
        lambda config, tokenizer: (_ for _ in ()).throw(WiringReached()),
        raising=False,
    )
    monkeypatch.setattr(
        trainer_thinker,
        "Qwen3OmniMoeThinkerVisionAudioModel",
        lambda config: pytest.fail("entry bypassed reconciled builder"),
    )
    raw = {
        "experiment_name": "token-wiring-test",
        "stage1_init_ckpt": "unused-stage1",
        "model": {"model_config_path": str(model_yaml)},
        "data": {
            "train_corpus_path": "train.jsonl",
            "val_corpus_path": "val.jsonl",
            "image_root": "images",
            "audio_root": "audio",
        },
        "train": {"output_dir": str(tmp_path / "output")},
    }

    with pytest.raises(WiringReached):
        trainer_thinker.train_thinker_stage2(
            raw, tokenizer_name_or_path="tokenizer"
        )


def test_inference_loads_config_then_reconciles_before_weights(monkeypatch):
    events = []
    config = SimpleNamespace(vocab_size=16, image_token_id=99)
    tokenizer = object()
    captured = {}

    class SpyConfig:
        @classmethod
        def from_pretrained(cls, checkpoint):
            events.append("config-load")
            return config

    def reconcile(config_arg, tokenizer_arg):
        events.append("reconcile")
        config_arg.image_token_id = 7
        return {"image_token_id": 7}

    class SpyModel:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            events.append("weight-load")
            captured.update(kwargs)
            assert kwargs["config"] is config
            assert config.image_token_id == 7
            return cls()

    monkeypatch.setattr(cli, "Qwen3OmniMoeConfig", SpyConfig, raising=False)
    monkeypatch.setattr(
        cli, "reconcile_multimodal_token_ids", reconcile, raising=False
    )
    monkeypatch.setattr(
        cli, "Qwen3OmniMoeThinkerVisionAudioModel", SpyModel
    )

    model = cli._load_reconciled_stage2_model(
        "checkpoint-x",
        tokenizer,
        load_kwargs={"torch_dtype": torch.float32},
    )
    assert isinstance(model, SpyModel)
    assert events == ["config-load", "reconcile", "weight-load"]
    assert captured["torch_dtype"] is torch.float32


def test_inference_entry_calls_reconciled_loader(monkeypatch):
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    monkeypatch.setattr(
        cli,
        "_ensure_tokenizer",
        lambda tokenizer_name_or_path, checkpoint: tokenizer,
    )
    monkeypatch.setattr(
        cli,
        "_load_reconciled_stage2_model",
        lambda checkpoint, tokenizer, load_kwargs=None: (
            (_ for _ in ()).throw(WiringReached())
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli.Qwen3OmniMoeThinkerVisionAudioModel,
        "from_pretrained",
        classmethod(
            lambda cls, checkpoint, **kwargs: pytest.fail(
                "entry bypassed reconciled loader"
            )
        ),
    )
    args = Namespace(
        checkpoint="checkpoint",
        tokenizer_name_or_path=None,
        dtype="auto",
        max_seq_length=8,
        skip_bad_media=False,
    )

    with pytest.raises(WiringReached):
        cli.run_stage2(args)
```

- [ ] **Step 13: Run wiring tests and observe behavior failures**

Run:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_stage2_token_wiring.py
```

Expected: collection succeeds. Builder/loader tests fail with `AttributeError`
inside their bodies; entry tests fail because the current direct constructor
and direct `from_pretrained` paths bypass the sentinels.

- [ ] **Step 14: Add reconciled construction helpers and wire both entries**

In `trainer_thinker.py`:

```python
def _build_reconciled_stage2_model(
    model_config: Qwen3OmniMoeConfig,
    tokenizer: Any,
) -> Qwen3OmniMoeThinkerVisionAudioModel:
    reconcile_multimodal_token_ids(model_config, tokenizer)
    return Qwen3OmniMoeThinkerVisionAudioModel(model_config)
```

`train_thinker_stage2` must call this helper with the exact config object
loaded from YAML.

In `cli_infer_thinker.py`:

```python
def _load_reconciled_stage2_model(
    checkpoint: str,
    tokenizer: Any,
    *,
    load_kwargs: Optional[Mapping[str, Any]] = None,
) -> Qwen3OmniMoeThinkerVisionAudioModel:
    model_config = Qwen3OmniMoeConfig.from_pretrained(checkpoint)
    reconcile_multimodal_token_ids(model_config, tokenizer)
    return Qwen3OmniMoeThinkerVisionAudioModel.from_pretrained(
        checkpoint,
        config=model_config,
        **dict(load_kwargs or {}),
    )
```

Call this helper inside the existing `try` block in `run_stage2`, preserving
the torch-load/CVE error translation exactly. Do not add tokenizer tokens,
resize embeddings, modify dimensions, or change state-dict names.

- [ ] **Step 15: Run all token wiring tests green and commit**

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_special_tokens.py \
  tests/test_stage2_token_wiring.py \
  tests/test_multimodal_attention_mask.py \
  tests/test_stage2_config.py
git add src/qwen3_omni_pretrain/training/trainer_thinker.py \
  src/qwen3_omni_pretrain/cli_infer_thinker.py \
  tests/test_stage2_token_wiring.py
git commit -m "fix: reconcile Stage2 IDs before model loading"
```

---

### Task 5: Share strict media loading across training and inference

**Files:**
- Modify: `src/qwen3_omni_pretrain/data/collators.py:1-256`
- Modify: `src/qwen3_omni_pretrain/training/trainer_thinker.py`
- Modify: `src/qwen3_omni_pretrain/cli_infer_thinker.py:18-289`
- Modify: `tests/test_stage2_collator.py`
- Create: `tests/test_stage2_inference.py`

**Interfaces:**
- Consumes: optional resolved media paths and the normalized `skip_bad_media`.
- Produces:
  - `MediaLoadResult(tensor, present, error)`
  - `Stage2MediaLoader.load_optional(modality, path, sample_id) -> MediaLoadResult`
  - Stage-2 inference results containing `media_errors`.

- [ ] **Step 1: Add failing shared-loader tests**

First replace the existing one-case corrupt-media tests with a missing/corrupt
matrix that remains green against the current collator:

```python
@pytest.mark.parametrize("create_corrupt_file", [False, True])
def test_referenced_invalid_media_raises_by_default(
    tmp_path: Path, create_corrupt_file: bool
):
    bad_image = tmp_path / "bad.jpg"
    if create_corrupt_file:
        bad_image.write_text("not an image", encoding="utf-8")
    stage2_collator = collators.OmniStage2Collator(
        TinyTokenizer(), max_seq_length=4, image_size=2, max_audio_len=4
    )

    with pytest.raises(collators.MediaLoadError) as exc_info:
        stage2_collator(
            [{
                "id": "bad-image",
                "input_text": "A",
                "target_text": "B",
                "image_path": str(bad_image),
            }]
        )

    assert "bad-image" in str(exc_info.value)
    assert str(bad_image) in str(exc_info.value)


@pytest.mark.parametrize("create_corrupt_file", [False, True])
def test_skip_bad_media_records_each_invalid_reference(
    tmp_path: Path, create_corrupt_file: bool
):
    bad_audio = tmp_path / "bad.wav"
    if create_corrupt_file:
        bad_audio.write_text("not audio", encoding="utf-8")
    stage2_collator = collators.OmniStage2Collator(
        TinyTokenizer(),
        max_seq_length=4,
        image_size=2,
        max_audio_len=4,
        skip_bad_media=True,
    )

    batch = stage2_collator(
        [{
            "id": "bad-audio",
            "input_text": "A",
            "target_text": "B",
            "audio_path": str(bad_audio),
        }]
    )

    assert batch["has_audio"].tolist() == [0]
    error = batch["_media_errors"][0]
    assert error["sample_id"] == "bad-audio"
    assert error["modality"] == "audio"
    assert error["path"] == str(bad_audio)
    assert error["error_type"]
    assert error["error"]
```

Then append the new shared-loader and trainer-wiring tests:

```python
def test_media_loader_distinguishes_omitted_from_missing(tmp_path: Path):
    loader = collators.Stage2MediaLoader(image_size=2, max_audio_len=4)
    omitted = loader.load_optional(
        modality="image", path=None, sample_id="omitted"
    )
    assert omitted.present == 0
    assert omitted.error is None

    missing = tmp_path / "missing.jpg"
    with pytest.raises(collators.MediaLoadError, match="missing"):
        loader.load_optional(
            modality="image", path=str(missing), sample_id="referenced"
        )


def test_media_loader_skip_returns_structured_error(tmp_path: Path):
    missing_audio = tmp_path / "missing.wav"
    loader = collators.Stage2MediaLoader(
        image_size=2, max_audio_len=4, skip_bad_media=True
    )
    result = loader.load_optional(
        modality="audio",
        path=str(missing_audio),
        sample_id="audio-7",
    )
    assert result.present == 0
    assert result.error["sample_id"] == "audio-7"
    assert result.error["modality"] == "audio"
    assert result.error["path"] == str(missing_audio)
    assert result.error["error_type"]
    assert result.error["error"]


def test_trainer_collator_helper_uses_the_shared_media_policy():
    from types import SimpleNamespace

    from qwen3_omni_pretrain.training import trainer_thinker

    runtime = SimpleNamespace(max_seq_length=4, skip_bad_media=True)
    stage2_collator = trainer_thinker._build_stage2_collator(
        runtime, TinyTokenizer()
    )
    assert isinstance(
        stage2_collator.media_loader, collators.Stage2MediaLoader
    )
    assert stage2_collator.media_loader.skip_bad_media is True
```

- [ ] **Step 2: Run the tests and observe the missing loader**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_collator.py::test_referenced_invalid_media_raises_by_default \
  tests/test_stage2_collator.py::test_skip_bad_media_records_each_invalid_reference \
  tests/test_stage2_collator.py::test_media_loader_distinguishes_omitted_from_missing \
  tests/test_stage2_collator.py::test_media_loader_skip_returns_structured_error \
  tests/test_stage2_collator.py::test_trainer_collator_helper_uses_the_shared_media_policy
```

Expected: both parameterized collator matrices pass. The three new tests are
collected and FAIL inside their test bodies because
`collators.Stage2MediaLoader` does not exist yet.

- [ ] **Step 3: Implement the shared media result and loader**

Extend the imports to include `dataclass` and `Optional`, then add:

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MediaLoadResult:
    tensor: torch.Tensor
    present: int
    error: Optional[Dict[str, str]] = None


class Stage2MediaLoader:
    def __init__(
        self,
        image_size: int = 224,
        max_audio_len: int = 32000,
        *,
        skip_bad_media: bool = False,
    ) -> None:
        self.image_size = int(image_size)
        self.max_audio_len = int(max_audio_len)
        self.skip_bad_media = bool(skip_bad_media)

    def _zero(self, modality: str) -> torch.Tensor:
        if modality == "image":
            return torch.zeros(3, self.image_size, self.image_size)
        if modality == "audio":
            return torch.zeros(self.max_audio_len)
        raise ValueError(f"unsupported modality: {modality}")

    def load_optional(
        self,
        *,
        modality: str,
        path: Optional[str],
        sample_id: str,
    ) -> MediaLoadResult:
        if modality not in {"image", "audio"}:
            raise ValueError(f"unsupported modality: {modality}")
        if path is None:
            return MediaLoadResult(self._zero(modality), 0)
        loader = self._load_image if modality == "image" else self._load_audio
        try:
            return MediaLoadResult(loader(path), 1)
        except Exception as cause:
            error = MediaLoadError(
                modality=modality,
                path=path,
                sample_id=sample_id,
                cause=cause,
            )
            if not self.skip_bad_media:
                raise error from cause
            return MediaLoadResult(self._zero(modality), 0, error.to_dict())
```

Move the current `_load_image` and `_load_audio` implementations unchanged
from `OmniStage2Collator` into `Stage2MediaLoader`.

- [ ] **Step 4: Make the collator consume `Stage2MediaLoader`**

In the collator constructor:

```python
self.media_loader = Stage2MediaLoader(
    image_size=image_size,
    max_audio_len=max_audio_len,
    skip_bad_media=skip_bad_media,
)
self.image_size = self.media_loader.image_size
self.max_audio_len = self.media_loader.max_audio_len
self.skip_bad_media = self.media_loader.skip_bad_media
```

For each modality:

```python
result = self.media_loader.load_optional(
    modality="image",
    path=str(image_path) if image_path else None,
    sample_id=sample_id,
)
images.append(result.tensor)
has_image_flags.append(result.present)
if result.error is not None:
    media_errors.append(result.error)
```

Use the analogous block for audio and remove the collator's private
`_load_or_handle`, `_load_image`, and `_load_audio`.

- [ ] **Step 5: Run all collator tests**

Run:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_stage2_collator.py
```

Expected: all existing and new tests pass.

- [ ] **Step 6: Add a failing CLI flag test**

Create `tests/test_stage2_inference.py`:

```python
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from qwen3_omni_pretrain import cli_infer_thinker as cli
from qwen3_omni_pretrain.data.collators import (
    MediaLoadError,
    Stage2MediaLoader,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)


class TinyInferenceTokenizer:
    eos_token_id = None
    _vocab = {
        "<|image_pad|>": 10,
        "<|video_pad|>": 11,
        "<|audio_pad|>": 12,
        "<|audio_start|>": 13,
        "<|audio_end|>": 14,
    }

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "x"

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self):
        return 15


class TinyInferenceModel(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), 4)
        logits[:, -1, 2] = 1
        return {"logits": logits}


def test_parse_args_accepts_explicit_skip_bad_media(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3-omni-infer",
            "--stage",
            "stage2",
            "--checkpoint",
            "checkpoint",
            "--skip_bad_media",
        ],
    )
    assert cli.parse_args().skip_bad_media is True
```

- [ ] **Step 7: Run the CLI test and observe the rejected flag**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_inference.py::test_parse_args_accepts_explicit_skip_bad_media
```

Expected: the test FAILS with argparse `SystemExit(2)` and
`unrecognized arguments: --skip_bad_media`.

- [ ] **Step 8: Add only the explicit CLI switch**

Add to `parse_args`:

```python
parser.add_argument(
    "--skip_bad_media",
    action="store_true",
    help="Treat referenced invalid media as absent and report structured errors.",
)
```

- [ ] **Step 9: Re-run the CLI flag test**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_inference.py::test_parse_args_accepts_explicit_skip_bad_media
```

Expected: PASS.

- [ ] **Step 10: Add failing inference and `run_stage2` observability tests**

Append:

```python
def test_stage2_inference_omitted_media_generates_without_errors():
    result = cli.greedy_decode_stage2(
        model=TinyInferenceModel(),
        tokenizer=TinyInferenceTokenizer(),
        media_loader=Stage2MediaLoader(image_size=2, max_audio_len=4),
        device=torch.device("cpu"),
        sample={"id": "sample-0", "input_text": "A"},
        max_new_tokens=1,
        image_root="",
        audio_root="",
        max_seq_length=8,
    )
    assert result["text"] == "x"
    assert result["media_errors"] == []


@pytest.mark.parametrize("create_corrupt_file", [False, True])
def test_stage2_inference_is_strict_for_referenced_invalid_media(
    tmp_path: Path, create_corrupt_file: bool
):
    bad_image = tmp_path / "bad.jpg"
    if create_corrupt_file:
        bad_image.write_text("not an image", encoding="utf-8")
    with pytest.raises(MediaLoadError, match="sample-1"):
        cli.greedy_decode_stage2(
            model=TinyInferenceModel(),
            tokenizer=TinyInferenceTokenizer(),
            media_loader=Stage2MediaLoader(image_size=2, max_audio_len=4),
            device=torch.device("cpu"),
            sample={"id": "sample-1", "input_text": "A", "image": str(bad_image)},
            max_new_tokens=1,
            image_root="",
            audio_root="",
            max_seq_length=8,
        )


@pytest.mark.parametrize("create_corrupt_file", [False, True])
def test_stage2_inference_skip_reports_each_invalid_media_and_generates(
    tmp_path: Path, create_corrupt_file: bool
):
    bad_audio = tmp_path / "bad.wav"
    if create_corrupt_file:
        bad_audio.write_text("not audio", encoding="utf-8")
    result = cli.greedy_decode_stage2(
        model=TinyInferenceModel(),
        tokenizer=TinyInferenceTokenizer(),
        media_loader=Stage2MediaLoader(
            image_size=2, max_audio_len=4, skip_bad_media=True
        ),
        device=torch.device("cpu"),
        sample={
            "id": "sample-2",
            "input_text": "A",
            "audio": str(bad_audio),
        },
        max_new_tokens=1,
        image_root="",
        audio_root="",
        max_seq_length=8,
    )
    assert result["text"] == "x"
    assert result["stats"]["output_tokens"] == 1
    error = result["media_errors"][0]
    assert error["sample_id"] == "sample-2"
    assert error["modality"] == "audio"
    assert error["path"] == str(bad_audio)
    assert error["error_type"]
    assert error["error"]


def test_run_stage2_uses_skip_flag_and_emits_json_error(
    monkeypatch, capsys, tmp_path: Path
):
    missing_audio = tmp_path / "missing.wav"
    config = Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": False,
            "moe_shared_expert": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_ensure_tokenizer",
        lambda tokenizer_name_or_path, checkpoint: TinyInferenceTokenizer(),
    )
    monkeypatch.setattr(
        cli.Qwen3OmniMoeConfig,
        "from_pretrained",
        classmethod(lambda cls, checkpoint: config),
    )
    monkeypatch.setattr(
        cli.Qwen3OmniMoeThinkerVisionAudioModel,
        "from_pretrained",
        classmethod(
            lambda cls, checkpoint, **kwargs: TinyInferenceModel()
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_jsonl",
        lambda path, limit: [
            {
                "id": "observable",
                "input_text": "A",
                "audio": str(missing_audio),
            }
        ],
    )
    args = Namespace(
        checkpoint="checkpoint",
        tokenizer_name_or_path=None,
        dtype="auto",
        max_seq_length=8,
        max_new_tokens=1,
        image_root="",
        audio_root="",
        skip_bad_media=True,
        chat=False,
        prompt=None,
        jsonl="samples.jsonl",
        num_samples=1,
    )

    cli.run_stage2(args)

    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"media_error"')
    ]
    assert len(payloads) == 1
    error = payloads[0]["media_error"]
    assert error["sample_id"] == "observable"
    assert error["modality"] == "audio"
    assert error["path"] == str(missing_audio)
    assert error["error_type"]
    assert error["error"]
```

- [ ] **Step 11: Run inference tests and observe behavior-level failures**

Run:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_stage2_inference.py
```

Expected: the CLI flag test stays green. The direct inference tests FAIL with
`TypeError` because `greedy_decode_stage2` has no `media_loader` parameter,
and the `run_stage2` test FAILS because no structured JSON error is emitted.

- [ ] **Step 12: Wire the shared loader into inference and observable output**

Replace the `OmniStage2Collator` import with `Stage2MediaLoader` and add:

```python
def _build_stage2_media_loader(args: argparse.Namespace) -> Stage2MediaLoader:
    return Stage2MediaLoader(skip_bad_media=args.skip_bad_media)


def _emit_stage2_media_errors(errors: List[Dict[str, str]]) -> None:
    for error in errors:
        print(json.dumps({"media_error": error}, ensure_ascii=False))
```

Replace the `collator` parameter of `greedy_decode_stage2` with
`media_loader: Stage2MediaLoader`. Remove `os.path.exists` and both bare
exception handlers. Load both modalities with:

```python
sample_id = str(sample.get("id", "inference-sample"))
image_result = media_loader.load_optional(
    modality="image",
    path=_resolve_path(sample.get("image"), image_root),
    sample_id=sample_id,
)
audio_result = media_loader.load_optional(
    modality="audio",
    path=_resolve_path(sample.get("audio"), audio_root),
    sample_id=sample_id,
)
media_errors = [
    error
    for error in (image_result.error, audio_result.error)
    if error is not None
]
```

Build tensors/presence flags from the two results and include
`"media_errors": media_errors` in the returned dictionary.

At the start of `run_stage2`, construct exactly one loader:

```python
media_loader = _build_stage2_media_loader(args)
```

Pass it to every `greedy_decode_stage2` call. Immediately after each result,
call:

```python
_emit_stage2_media_errors(result["media_errors"])
```

Keep the existing torch-load safety error handling around model loading.

- [ ] **Step 13: Preserve the single training collator construction path**

Do not construct `OmniStage2Collator` directly in Task 5. Retain Task 3's
single `_build_stage2_collator(runtime, tokenizer)` path; because it already
passes `runtime.skip_bad_media`, the collator refactor now creates its shared
loader with the same policy. The failing helper regression from Step 1 proves
this path.

- [ ] **Step 14: Run all Stage-2 behavior tests**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_stage2_collator.py \
  tests/test_stage2_inference.py \
  tests/test_multimodal_attention_mask.py \
  tests/test_stage2_config.py
```

Expected: all tests pass; the missing/corrupt matrices cover both strict and
skip behavior in training and inference, each inference skip case completes a
one-token decode, and the `run_stage2` integration test proves the flag is
observable as one JSON diagnostic.

- [ ] **Step 15: Commit shared media semantics**

```bash
git add src/qwen3_omni_pretrain/data/collators.py \
  src/qwen3_omni_pretrain/training/trainer_thinker.py \
  src/qwen3_omni_pretrain/cli_infer_thinker.py \
  tests/test_stage2_collator.py tests/test_stage2_inference.py
git commit -m "fix: share strict Stage2 media loading"
```

---

### Task 6: Share top-k selection with tensor-parallel MoE

**Files:**
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modules/moe.py:1-167`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py:24-396`
- Modify: `tests/test_moe.py`
- Create: `tests/test_tp_moe.py`

**Interfaces:**
- Consumes: 2-D router probabilities, `k`, and a renormalization flag.
- Produces: `select_topk_routes(router_probs, k, renormalize) -> tuple[Tensor, Tensor]`.

- [ ] **Step 1: Add failing helper semantics**

Append to `tests/test_moe.py`:

```python
from qwen3_omni_pretrain.models.qwen3_omni_moe.modules import (
    moe as moe_module,
)


def test_shared_topk_selector_respects_renormalization_flag():
    probabilities = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32)
    normalized, normalized_indices = moe_module.select_topk_routes(
        probabilities, k=2, renormalize=True
    )
    raw, raw_indices = moe_module.select_topk_routes(
        probabilities, k=2, renormalize=False
    )
    assert torch.equal(normalized_indices, raw_indices)
    assert normalized.sum(dim=-1).item() == pytest.approx(1.0)
    assert raw.sum(dim=-1).item() == pytest.approx(0.9)


def test_standard_moe_bfloat16_forward_preserves_output_dtype():
    model = Qwen3OmniMoeMLP(
        hidden_size=2,
        intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        use_shared_expert=False,
    ).to(dtype=torch.bfloat16)
    output, _ = model(
        torch.ones(1, 1, 2, dtype=torch.bfloat16)
    )
    assert output.dtype is torch.bfloat16
```

- [ ] **Step 2: Run and observe the missing helper**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_moe.py::test_shared_topk_selector_respects_renormalization_flag \
  tests/test_moe.py::test_standard_moe_bfloat16_forward_preserves_output_dtype
```

Expected: the selector test FAILS at the call with
`AttributeError: module ...moe has no attribute 'select_topk_routes'`; test
collection itself succeeds. The BF16 characterization remains green and guards
the dtype cast required by the FP32 selector.

- [ ] **Step 3: Implement the shared FP32 selector**

Add to `modules/moe.py`:

```python
def select_topk_routes(
    router_probs: torch.Tensor,
    *,
    k: int,
    renormalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values, indices = router_probs.float().topk(k=int(k), dim=-1)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values, indices
```

Change standard `_dispatch_tokens` to call this helper and keep its existing
flattened `(token_indices, expert_indices, scores)` return contract. In the
standard forward path, compute probabilities with:

```python
gate_probs = torch.softmax(gate_logits.float(), dim=-1)
```

Before multiplying expert outputs, restore the selected-score dtype so FP16 or
BF16 output accumulation remains valid:

```python
weighted_output = expert_output * selected_scores.to(
    expert_output.dtype
).unsqueeze(-1)
```

- [ ] **Step 4: Run standard MoE tests**

Run:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_moe.py
```

Expected: all standard MoE tests pass.

- [ ] **Step 5: Add a failing standard/TP parity test**

Create `tests/test_tp_moe.py`:

```python
import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text_tp import (
    TensorParallelMoeMLP,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modules.moe import (
    Qwen3OmniMoeMLP,
)


def test_tp_world_size_one_matches_standard_routing_and_state_keys():
    standard = Qwen3OmniMoeMLP(
        hidden_size=1,
        intermediate_size=1,
        num_experts=3,
        num_experts_per_tok=2,
        use_shared_expert=False,
        router_normalize_init=False,
        renormalize_topk=True,
    )
    parallel = TensorParallelMoeMLP(
        hidden_size=1,
        intermediate_size=1,
        num_experts=3,
        num_experts_per_tok=2,
        use_shared_expert=False,
        router_normalize_init=False,
        renormalize_topk=True,
    )
    with torch.no_grad():
        gate = torch.log(torch.tensor([[0.6], [0.3], [0.1]]))
        standard.gate.weight.copy_(gate)
        parallel.gate.weight.copy_(gate)
        for standard_expert, parallel_expert in zip(
            standard.experts, parallel.experts
        ):
            standard_expert.fc1.weight.fill_(1)
            standard_expert.fc2.weight.fill_(1)
            parallel_expert.fc1.weight.fill_(1)
            parallel_expert.fc2.weight.fill_(1)

    standard_output, _ = standard(torch.ones(1, 1, 1))
    parallel_output, _ = parallel(torch.ones(1, 1, 1))

    assert torch.allclose(standard_output, parallel_output, atol=1e-6, rtol=1e-6)
    expected_keys = {
        "gate.weight",
        "experts.0.fc1.weight", "experts.0.fc2.weight",
        "experts.1.fc1.weight", "experts.1.fc2.weight",
        "experts.2.fc1.weight", "experts.2.fc2.weight",
    }
    assert set(standard.state_dict()) == expected_keys
    assert set(parallel.state_dict()) == expected_keys
```

- [ ] **Step 6: Run parity test and observe the 0.9-scale TP output**

Run:

```bash
.venv-prototype/bin/python -m pytest -q tests/test_tp_moe.py
```

Expected: FAIL at `torch.allclose`; TP uses unnormalized selected probabilities.

- [ ] **Step 7: Use the helper in TP without changing expert execution**

Import `select_topk_routes` into `modeling_thinker_text_tp.py` and replace the
local `topk` call with:

```python
topk_vals, topk_idx = select_topk_routes(
    gate_probs,
    k=self.num_experts_per_tok,
    renormalize=self.renormalize_topk,
)
```

Do not change TP expert classes, dummy forwards, collectives, auxiliary-loss
calculation, or module attribute names.

- [ ] **Step 8: Run all MoE and numerical tests**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_moe.py tests/test_tp_moe.py tests/test_training_loop_numerics.py
```

Expected: all tests pass, including non-finite propagation.

- [ ] **Step 9: Commit TP routing parity**

```bash
git add src/qwen3_omni_pretrain/models/qwen3_omni_moe/modules/moe.py \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py \
  tests/test_moe.py tests/test_tp_moe.py
git commit -m "fix: align TP MoE top-k routing"
```

---

### Task 7: Run clean-profile acceptance and update final documentation

**Files:**
- Modify: `README.md`
- Test: full `tests/` suite

**Interfaces:**
- Consumes: all contracts produced by Tasks 1-6.
- Produces: a verified prototype profile, verified official-reference profile, and final user-facing commands.

- [ ] **Step 1: Run the complete prototype suite**

Run:

```bash
.venv-prototype/bin/python -m pytest -q
.venv-prototype/bin/python -m compileall -q src scripts tests
```

Expected: all tests pass with no collection errors; compile exits 0.

- [ ] **Step 2: Run the tiny Stage-2 inference regression explicitly**

Run:

```bash
.venv-prototype/bin/python -m pytest -q \
  tests/test_multimodal_attention_mask.py::test_real_tiny_wrapper_runs_one_autoregressive_step_without_labels
```

Expected: PASS; the real
`Qwen3OmniMoeThinkerVisionAudioModel` wrapper performs two label-free forwards,
and the hand-written greedy step extends the text sequence by one token.

- [ ] **Step 3: Create the official-reference environment**

Run:

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

Expected: installation succeeds without replacing `.venv-prototype`.

- [ ] **Step 4: Run reference imports**

Run:

```bash
ffmpeg -version
.venv-qwen-reference/bin/python - <<'PY'
import torch
import torchvision
import torchaudio
import transformers
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
assert torch.__version__.split("+", 1)[0] == "2.10.0"
assert torchvision.__version__.split("+", 1)[0] == "0.25.0"
assert torchaudio.__version__.split("+", 1)[0] == "2.10.0"
assert transformers.__version__ == "5.2.0"
assert callable(process_mm_info)
assert Qwen3OmniMoeForConditionalGeneration is not None
assert Qwen3OmniMoeProcessor is not None
PY
```

Expected: FFmpeg is found on `PATH`; the Python smoke command exits 0.

- [ ] **Step 5: Run the project suite in the reference profile**

Run:

```bash
.venv-qwen-reference/bin/python -m pytest -q
.venv-qwen-reference/bin/python -m compileall -q src scripts tests
```

Expected: all tests pass and compile exits 0 under Transformers 5.2.0.

- [ ] **Step 6: Record exact verification commands and failure guidance**

Ensure README contains:

- both creation/install command sets;
- profile-specific test commands;
- the version/import smoke command;
- `ffmpeg -version` for the reference profile;
- an explanation that an undefined symbol in `libtorchaudio.so` means the
  TorchAudio wheel does not match the installed Torch release;
- a statement that the reference profile is for comparison and does not make
  this repository checkpoint-compatible.

- [ ] **Step 7: Run final repository checks**

Run:

```bash
git diff --check
git status --short
.venv-prototype/bin/python -m pytest -q
.venv-prototype/bin/python -m compileall -q src scripts tests
ffmpeg -version
.venv-qwen-reference/bin/python -m pytest -q
.venv-qwen-reference/bin/python -m compileall -q src scripts tests
```

Expected: no whitespace errors, only intended README changes before the final
commit, and both profiles pass.

- [ ] **Step 8: Commit final acceptance documentation**

```bash
git add README.md
git commit -m "docs: record P0 runtime verification"
```

- [ ] **Step 9: Confirm clean final state**

Run:

```bash
git status --short --branch
git log --oneline --decorate -20
```

Expected: clean working tree on `fix/p0-training-correctness-20260729`, with
the design/plan commits and focused implementation commits for each completed
TDD cycle.
