# Profile Contracts and Qwen3 Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立四个架构 profile 的机器可验证身份、兼容级别、配置约束和架构摘要，并把固定版本的官方 Qwen3-Omni 实现接入为 golden oracle。

**Architecture:** 保留现有模型作为 `legacy_prototype`，但移除其与官方 `qwen3_omni_moe` 的 model-type 冲突。公共 `architecture` 包只保存 profile、manifest、validation 和 summary；各 profile 的具体模型在后续计划中注册。官方 Qwen3-Omni oracle 只在隔离的 Transformers 5.2.0 环境运行，普通 CPU 测试不下载大权重。

**Tech Stack:** Python 3.10, dataclasses, enum, PyYAML, PyTorch 2.10.0, Transformers 4.57.6/5.2.0, qwen-omni-utils 0.0.9, pytest.

## Global Constraints

- 架构 profile 名称固定为 `legacy_prototype`、`qwen3_omni_reference`、`qwen35_omni_inspired`、`mimo_v25_experimental`。
- 兼容级别固定为 `legacy-prototype`、`structure-aligned`、`checkpoint-compatible`、`paper-inspired`、`MiMo-style-experiment`。
- `qwen35_omni_inspired` 必须始终保存 `exact_official_checkpoint_compatible=false`。
- `mimo_v25_experimental` 不使用 MiMo 官方 model type 或 checkpoint 名称。
- legacy profile 必须继续读取旧 `model_type="qwen3_omni_moe"` checkpoint，但新保存的配置必须使用非官方 model type。
- legacy 配置迁移以当前普通、非 TP 构造器的实际层选择为权威；当前 TP 与普通路径的 MoE 层差异属于待修复错误，不作为兼容行为保留。
- `thinker_config` 是 legacy Thinker 结构字段的唯一来源；顶层同名字段只作为序列化镜像，不得形成第二套尺寸。
- tokenizer 和 profile 不匹配、配置自相矛盾、非法稀疏 MoE 配置必须在模型构建前失败。
- 普通测试不得联网或下载大权重；权重级 oracle 测试必须显式 `--run-large-model-tests`。
- Prototype 环境固定为 PyTorch 2.10.0、TorchVision 0.25.0、TorchAudio 2.10.0、Transformers 4.57.6。
- Qwen3 oracle 环境固定为相同 PyTorch trio、Transformers 5.2.0、qwen-omni-utils 0.0.9，并要求 `ffmpeg` 在 `PATH`。
- 当前 111 个测试必须持续通过。
- Snippet 中的 `...` 只表示 Protocol/签名节选；任务提交不得保留未实现
  stub，必须由该任务列出的行为测试证明可用。
- 每个行为变更使用 red-green-refactor，并形成独立提交。

---

## File responsibility map

- `src/qwen3_omni_pretrain/architecture/profiles.py`: profile 和 compatibility 枚举。
- `src/qwen3_omni_pretrain/architecture/manifest.py`: source revision、manifest 序列化和交叉字段校验。
- `src/qwen3_omni_pretrain/architecture/config_validation.py`: layer、MoE、context 和 profile-specific 配置校验。
- `src/qwen3_omni_pretrain/architecture/summary.py`: 每层结构和参数摘要，不负责模型构建。
- `src/qwen3_omni_pretrain/profiles/registry.py`: profile factory 注册和惰性构建。
- `src/qwen3_omni_pretrain/profiles/legacy_prototype/config_adapter.py`: 旧 checkpoint config 的单向兼容迁移。
- `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/oracle.py`: 固定 revision、官方 config/processor/model 加载入口。
- `tests/fixtures/qwen3_omni/config_contract.json`: 从固定官方 config 提取的小型、可审计契约。
- `scripts/inspect_architecture.py`: 加载 YAML、校验并打印 manifest、layer summary 和参数摘要。

---

### Task 1: Define profile and manifest contracts

**Files:**
- Create: `src/qwen3_omni_pretrain/architecture/__init__.py`
- Create: `src/qwen3_omni_pretrain/architecture/profiles.py`
- Create: `src/qwen3_omni_pretrain/architecture/manifest.py`
- Create: `tests/architecture/test_profile_manifest.py`

**Interfaces:**
- Consumes: no earlier implementation task.
- Produces: `ArchitectureProfile`, `CompatibilityLevel`, `SourceRevision`, `ProfileManifest`, `ProfileManifest.from_dict()`, `ProfileManifest.to_dict()`, and `ProfileManifest.validate()` for every later plan.

- [ ] **Step 1: Write failing enum and round-trip tests**

```python
from qwen3_omni_pretrain.architecture.manifest import (
    ProfileManifest,
    SourceRevision,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)


def test_manifest_round_trip_preserves_provenance():
    manifest = ProfileManifest(
        architecture_profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
        compatibility_level=CompatibilityLevel.PAPER_INSPIRED,
        sources={
            "backbone": SourceRevision(
                name="Qwen3.5-35B-A3B",
                revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
            )
        },
        assumptions=("predecessor codec proxy",),
        exact_official_checkpoint_compatible=False,
    )
    assert ProfileManifest.from_dict(manifest.to_dict()) == manifest


def test_qwen35_manifest_rejects_checkpoint_compatibility():
    with pytest.raises(ValueError, match="cannot use"):
        ProfileManifest(
            architecture_profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
            compatibility_level=CompatibilityLevel.CHECKPOINT_COMPATIBLE,
            sources={},
            assumptions=(),
            exact_official_checkpoint_compatible=True,
        )


def test_manifest_does_not_coerce_boolean_strings():
    raw = valid_manifest().to_dict()
    raw["exact_official_checkpoint_compatible"] = "false"
    with pytest.raises(TypeError, match="boolean"):
        ProfileManifest.from_dict(raw)


def test_manifest_rejects_unknown_fields():
    raw = valid_manifest().to_dict()
    raw["typo"] = 1
    with pytest.raises(ValueError, match="unknown"):
        ProfileManifest.from_dict(raw)


def test_mimo_experiment_cannot_claim_checkpoint_compatibility():
    with pytest.raises(ValueError, match="cannot use"):
        ProfileManifest(
            architecture_profile=ArchitectureProfile.MIMO_V25_EXPERIMENTAL,
            compatibility_level=CompatibilityLevel.CHECKPOINT_COMPATIBLE,
            sources={},
            assumptions=("tiny generic mechanism experiment",),
            exact_official_checkpoint_compatible=True,
        )


def test_structure_aligned_qwen3_cannot_set_exact_flag():
    with pytest.raises(ValueError, match="checkpoint-compatible level"):
        ProfileManifest(
            architecture_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
            compatibility_level=CompatibilityLevel.STRUCTURE_ALIGNED,
            sources={},
            assumptions=(),
            exact_official_checkpoint_compatible=True,
        )


def test_manifest_context_and_digest_are_strict_and_stable():
    manifest = valid_manifest()
    assert manifest.validated_context_length == 0
    assert len(manifest.canonical_sha256()) == 64
    assert (
        ProfileManifest.from_dict(manifest.to_dict()).canonical_sha256()
        == manifest.canonical_sha256()
    )
    raw = manifest.to_dict()
    raw["validated_context_length"] = True
    with pytest.raises(TypeError, match="validated_context_length"):
        ProfileManifest.from_dict(raw)
```

- [ ] **Step 2: Run the tests and confirm the package is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/architecture/test_profile_manifest.py -q
```

Expected: collection fails with `ModuleNotFoundError: qwen3_omni_pretrain.architecture`.

- [ ] **Step 3: Implement the exact enums**

`profiles.py`:

```python
from enum import Enum


class ArchitectureProfile(str, Enum):
    LEGACY_PROTOTYPE = "legacy_prototype"
    QWEN3_OMNI_REFERENCE = "qwen3_omni_reference"
    QWEN35_OMNI_INSPIRED = "qwen35_omni_inspired"
    MIMO_V25_EXPERIMENTAL = "mimo_v25_experimental"


class CompatibilityLevel(str, Enum):
    LEGACY_PROTOTYPE = "legacy-prototype"
    STRUCTURE_ALIGNED = "structure-aligned"
    CHECKPOINT_COMPATIBLE = "checkpoint-compatible"
    PAPER_INSPIRED = "paper-inspired"
    MIMO_STYLE_EXPERIMENT = "MiMo-style-experiment"
```

- [ ] **Step 4: Implement immutable manifest serialization and validation**

`manifest.py` must contain this public shape:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from .profiles import ArchitectureProfile, CompatibilityLevel


@dataclass(frozen=True)
class SourceRevision:
    name: str
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(
            self.revision, str
        ):
            raise TypeError("source name and revision must be strings")
        if not self.name.strip() or not self.revision.strip():
            raise ValueError("source name and revision must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "revision": self.revision}


@dataclass(frozen=True)
class ProfileManifest:
    architecture_profile: ArchitectureProfile
    compatibility_level: CompatibilityLevel
    sources: Mapping[str, SourceRevision]
    assumptions: tuple[str, ...]
    exact_official_checkpoint_compatible: bool
    # Zero means that no implementation-backed long-context gate has passed.
    # It is deliberately not inferred from max_position_embeddings/RoPE config.
    validated_context_length: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_profile, ArchitectureProfile):
            raise TypeError("architecture_profile must be ArchitectureProfile")
        if not isinstance(self.compatibility_level, CompatibilityLevel):
            raise TypeError("compatibility_level must be CompatibilityLevel")
        if type(self.exact_official_checkpoint_compatible) is not bool:
            raise TypeError(
                "exact_official_checkpoint_compatible must be boolean"
            )
        if (
            type(self.validated_context_length) is not int
            or self.validated_context_length < 0
        ):
            raise TypeError(
                "validated_context_length must be a non-negative integer"
            )
        if not isinstance(self.sources, Mapping):
            raise TypeError("sources must be a mapping")
        if any(
            not isinstance(source, SourceRevision)
            for source in self.sources.values()
        ):
            raise TypeError("source values must be SourceRevision")
        if not isinstance(self.assumptions, tuple):
            raise TypeError("assumptions must be a tuple")
        object.__setattr__(
            self,
            "sources",
            MappingProxyType(dict(self.sources)),
        )
        self.validate()

    def validate(self) -> None:
        allowed_levels = {
            ArchitectureProfile.LEGACY_PROTOTYPE: {
                CompatibilityLevel.LEGACY_PROTOTYPE,
            },
            ArchitectureProfile.QWEN3_OMNI_REFERENCE: {
                CompatibilityLevel.STRUCTURE_ALIGNED,
                CompatibilityLevel.CHECKPOINT_COMPATIBLE,
            },
            ArchitectureProfile.QWEN35_OMNI_INSPIRED: {
                CompatibilityLevel.PAPER_INSPIRED,
            },
            ArchitectureProfile.MIMO_V25_EXPERIMENTAL: {
                CompatibilityLevel.MIMO_STYLE_EXPERIMENT,
            },
        }
        if self.compatibility_level not in allowed_levels[
            self.architecture_profile
        ]:
            raise ValueError(
                f"{self.architecture_profile.value} cannot use "
                f"{self.compatibility_level.value}"
            )
        if (
            self.architecture_profile
            is not ArchitectureProfile.QWEN3_OMNI_REFERENCE
            and self.exact_official_checkpoint_compatible
        ):
            raise ValueError(
                "only qwen3_omni_reference may claim exact official "
                "checkpoint compatibility"
            )
        if (
            self.compatibility_level
            is CompatibilityLevel.CHECKPOINT_COMPATIBLE
            and not self.exact_official_checkpoint_compatible
        ):
            raise ValueError(
                "checkpoint-compatible requires "
                "exact_official_checkpoint_compatible=true"
            )
        if (
            self.exact_official_checkpoint_compatible
            and self.compatibility_level
            is not CompatibilityLevel.CHECKPOINT_COMPATIBLE
        ):
            raise ValueError(
                "exact checkpoint compatibility requires the "
                "checkpoint-compatible level"
            )
        if any(not isinstance(key, str) for key in self.sources):
            raise TypeError("source keys must be strings")
        if any(not key.strip() for key in self.sources):
            raise ValueError("source keys must be non-empty")
        if any(not isinstance(item, str) for item in self.assumptions):
            raise TypeError("assumptions must be strings")
        if any(not item.strip() for item in self.assumptions):
            raise ValueError("assumptions must be non-empty strings")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "architecture_profile": self.architecture_profile.value,
            "compatibility_level": self.compatibility_level.value,
            "sources": {
                key: source.to_dict()
                for key, source in sorted(self.sources.items())
            },
            "assumptions": list(self.assumptions),
            "exact_official_checkpoint_compatible": (
                self.exact_official_checkpoint_compatible
            ),
            "validated_context_length": self.validated_context_length,
        }

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ProfileManifest":
        if not isinstance(raw, Mapping):
            raise TypeError("manifest must be a mapping")
        allowed = {
            "architecture_profile",
            "compatibility_level",
            "sources",
            "assumptions",
            "exact_official_checkpoint_compatible",
            "validated_context_length",
        }
        missing = allowed - set(raw)
        unknown = set(raw) - allowed
        if missing or unknown:
            raise ValueError(
                f"invalid manifest keys: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        raw_sources = raw["sources"]
        if not isinstance(raw_sources, Mapping):
            raise ValueError("sources must be a mapping")
        for key, value in raw_sources.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise TypeError("source entries must be string-to-mapping")
            if set(value) != {"name", "revision"}:
                raise ValueError(
                    f"source {key!r} requires name and revision only"
                )
            if not isinstance(value["name"], str) or not isinstance(
                value["revision"], str
            ):
                raise TypeError("source name and revision must be strings")
        raw_assumptions = raw["assumptions"]
        if not isinstance(raw_assumptions, (list, tuple)) or any(
            not isinstance(item, str) for item in raw_assumptions
        ):
            raise TypeError("assumptions must be a sequence of strings")
        raw_exact = raw["exact_official_checkpoint_compatible"]
        if type(raw_exact) is not bool:
            raise TypeError(
                "exact_official_checkpoint_compatible must be a boolean"
            )
        raw_context = raw["validated_context_length"]
        if type(raw_context) is not int or raw_context < 0:
            raise TypeError(
                "validated_context_length must be a non-negative integer"
            )
        raw_profile = raw["architecture_profile"]
        raw_compatibility = raw["compatibility_level"]
        if not isinstance(raw_profile, str) or not isinstance(
            raw_compatibility, str
        ):
            raise TypeError("profile and compatibility level must be strings")
        manifest = cls(
            architecture_profile=ArchitectureProfile(raw_profile),
            compatibility_level=CompatibilityLevel(raw_compatibility),
            sources={
                key: SourceRevision(
                    name=value["name"],
                    revision=value["revision"],
                )
                for key, value in raw_sources.items()
            },
            assumptions=tuple(raw_assumptions),
            exact_official_checkpoint_compatible=raw_exact,
            validated_context_length=raw_context,
        )
        manifest.validate()
        return manifest
```

- [ ] **Step 5: Export the contracts and run the focused tests**

`architecture/__init__.py` must re-export the four public types. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/architecture/test_profile_manifest.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_omni_pretrain/architecture tests/architecture
git commit -m "feat: define architecture profile manifests"
```

---

### Task 2: Migrate the legacy model identity without breaking old checkpoints

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/__init__.py`
- Create: `src/qwen3_omni_pretrain/profiles/legacy_prototype/__init__.py`
- Create: `src/qwen3_omni_pretrain/profiles/legacy_prototype/config_adapter.py`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py:9-82`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py:104-245`
- Modify: `src/qwen3_omni_pretrain/training/trainer_thinker.py:1596-1610`
- Modify: `src/qwen3_omni_pretrain/cli_infer_thinker.py:164-177`
- Modify: `configs/model/qwen3_omni_1_3b_moe.yaml`
- Modify: `configs/model/qwen3_omni_7b_moe.yaml`
- Modify: `configs/model/qwen3_omni_14b_moe.yaml`
- Modify: `configs/model/qwen3_omni_30b_moe.yaml`
- Modify: `configs/model/qwen3_omni_70b_moe.yaml`
- Modify: `configs/model/qwen3_omni_120b_moe.yaml`
- Create: `tests/architecture/test_legacy_identity.py`
- Modify: `tests/test_checkpoint_transformers_compat.py`
- Modify: `tests/test_stage2_token_wiring.py`

**Interfaces:**
- Consumes: `ArchitectureProfile`, `CompatibilityLevel`, and `ProfileManifest`.
- Produces: `LEGACY_MODEL_TYPE`, `OLD_COLLIDING_MODEL_TYPE`,
  `adapt_legacy_config_dict(raw)`, and `load_legacy_config_dict(path)` used by
  legacy training and inference.

- [ ] **Step 1: Write the failing migration tests**

```python
from qwen3_omni_pretrain.profiles.legacy_prototype.config_adapter import (
    LEGACY_MODEL_TYPE,
    adapt_legacy_config_dict,
)


def test_old_colliding_model_type_is_migrated_with_warning():
    raw = {"model_type": "qwen3_omni_moe", "vocab_size": 32}
    with pytest.warns(DeprecationWarning, match="qwen3_omni_moe"):
        migrated = adapt_legacy_config_dict(raw)
    assert migrated["model_type"] == LEGACY_MODEL_TYPE
    assert migrated["architecture_profile"] == "legacy_prototype"
    assert raw["model_type"] == "qwen3_omni_moe"


def test_new_legacy_config_round_trip_never_saves_official_model_type():
    config = Qwen3OmniMoeConfig(vocab_size=32)
    saved = config.to_dict()
    assert config.model_type == "qwen3_omni_prototype"
    assert saved["architecture_profile"] == "legacy_prototype"
    assert saved["profile_manifest"]["compatibility_level"] == (
        "legacy-prototype"
    )
```

- [ ] **Step 2: Run the tests and observe the missing adapter failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/architecture/test_legacy_identity.py -q
```

Expected: FAIL because `profiles.legacy_prototype.config_adapter` does not exist.

- [ ] **Step 3: Implement a non-mutating one-way adapter**

`config_adapter.py`:

```python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping
import json
import warnings


LEGACY_MODEL_TYPE = "qwen3_omni_prototype"
OLD_COLLIDING_MODEL_TYPE = "qwen3_omni_moe"


def adapt_legacy_config_dict(
    raw: Mapping[str, object],
) -> dict[str, object]:
    migrated = deepcopy(dict(raw))
    model_type = migrated.get("model_type", LEGACY_MODEL_TYPE)
    if model_type == OLD_COLLIDING_MODEL_TYPE:
        warnings.warn(
            "legacy model_type='qwen3_omni_moe' collides with the official "
            "Transformers architecture; loading as qwen3_omni_prototype",
            DeprecationWarning,
            stacklevel=2,
        )
        migrated["model_type"] = LEGACY_MODEL_TYPE
    elif model_type != LEGACY_MODEL_TYPE:
        raise ValueError(f"not a legacy prototype config: {model_type!r}")
    thinker = deepcopy(dict(migrated.get("thinker_config", {})))
    indices = str(thinker.get("moe_layer_indices") or "").strip()
    raw_use_moe = thinker.get("use_moe", False)
    if type(raw_use_moe) is not bool:
        raise TypeError("legacy thinker use_moe must be boolean")
    effective_use_moe = raw_use_moe or bool(indices)
    if indices and not raw_use_moe:
        warnings.warn(
            "legacy standard model enabled MoE from moe_layer_indices while "
            "the TP path ignored it; preserving the standard layer graph",
            DeprecationWarning,
            stacklevel=2,
        )
    thinker["use_moe"] = effective_use_moe
    if not effective_use_moe:
        thinker["routing_kind"] = "dense"
    elif int(thinker["num_experts_per_tok"]) < int(thinker["num_experts"]):
        thinker["routing_kind"] = "sparse"
    elif int(thinker["num_experts_per_tok"]) == int(
        thinker["num_experts"]
    ):
        thinker["routing_kind"] = "dense_ensemble"
    else:
        raise ValueError("legacy top-k cannot exceed num_experts")
    migrated["thinker_config"] = thinker
    migrated["architecture_profile"] = "legacy_prototype"
    return migrated


def load_legacy_config_dict(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return adapt_legacy_config_dict(raw)
```

- [ ] **Step 4: Change only the new serialization identity**

In `Qwen3OmniMoeConfig`:

```python
model_type = "qwen3_omni_prototype"
architecture_profile = "legacy_prototype"
```

Add `routing_kind: str = "dense"` to `Qwen3OmniMoeThinkerConfig` in this
task so the adapter output is immediately constructible. Task 3 replaces
string-only acceptance with the enum-backed strict validator.

Construct and serialize this exact default manifest:

```python
ProfileManifest(
    architecture_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
    compatibility_level=CompatibilityLevel.LEGACY_PROTOTYPE,
    sources={},
    assumptions=("custom research architecture",),
    exact_official_checkpoint_compatible=False,
)
```

Store it as `self.profile_manifest`; `to_dict()` must emit
`profile_manifest.to_dict()` and must not accept a serialized manifest that
claims another profile.

- [ ] **Step 5: Migrate checked-in YAML identities**

Change every `configs/model/qwen3_omni_*_moe.yaml` to:

```yaml
model_type: qwen3_omni_prototype
architecture_profile: legacy_prototype
```

Do not change dimensions or layer selection in this task. Add a parametrized
test proving no checked-in config still serializes `qwen3_omni_moe`.

- [ ] **Step 6: Route legacy checkpoint loading through the adapter**

Add a classmethod:

```python
@classmethod
def from_legacy_pretrained_config(
    cls,
    path: str,
) -> "Qwen3OmniMoeConfig":
    return cls(**load_legacy_config_dict(path))
```

Use this method before `from_pretrained` in the Stage-2 training and inference
loaders. Route every current legacy checkpoint entrypoint through it:
`cli_infer_thinker.py` Stage-1 and Stage-2 loaders and
`trainer_thinker.py` Stage-1 resume/Stage-2 initialization. Pass the resulting
config explicitly to model construction so Transformers never attempts to
register the old official-colliding name. Tests monkeypatch each entrypoint
and prove adaptation occurs before any `from_pretrained` call.

Add fixture configs representing the old 14B and 30B contradiction
(`use_moe=false`, non-empty indices) and assert the adapter emits
`use_moe=true`, `routing_kind=sparse`, preserves the indices, and leaves the
input mapping unchanged. Add the old 7B top-2-of-2 case and assert
`dense_ensemble`.

- [ ] **Step 7: Run migration and existing checkpoint tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/architecture/test_legacy_identity.py \
  tests/test_checkpoint_transformers_compat.py \
  tests/test_stage2_token_wiring.py -q
```

Expected: all selected tests pass and old configs emit one deprecation warning.

- [ ] **Step 8: Commit**

```bash
git add src/qwen3_omni_pretrain/profiles \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py \
  src/qwen3_omni_pretrain/training/trainer_thinker.py \
  src/qwen3_omni_pretrain/cli_infer_thinker.py \
  configs/model/qwen3_omni_*_moe.yaml \
  tests/architecture/test_legacy_identity.py \
  tests/test_checkpoint_transformers_compat.py \
  tests/test_stage2_token_wiring.py
git commit -m "fix: isolate the legacy model identity"
```

---

### Task 3: Validate architecture configuration before model allocation

**Files:**
- Create: `src/qwen3_omni_pretrain/architecture/config_validation.py`
- Modify: `configs/model/qwen3_omni_1_3b_moe.yaml`
- Modify: `configs/model/qwen3_omni_7b_moe.yaml`
- Modify: `configs/model/qwen3_omni_14b_moe.yaml`
- Modify: `configs/model/qwen3_omni_30b_moe.yaml`
- Modify: `configs/model/qwen3_omni_70b_moe.yaml`
- Modify: `configs/model/qwen3_omni_120b_moe.yaml`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py:9-82`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py`
- Create: `tests/architecture/test_config_validation.py`

**Interfaces:**
- Consumes: `ArchitectureProfile`.
- Produces: `RoutingKind`, `parse_layer_indices(value, layer_count, field)`, and `validate_legacy_thinker_config(config)`.

- [ ] **Step 1: Write failing contradiction and sparsity tests**

```python
def test_boolean_and_layer_indices_cannot_disagree():
    config = Qwen3OmniMoeThinkerConfig(
        num_hidden_layers=4,
        use_moe=False,
        moe_layer_indices="1,3",
    )
    with pytest.raises(ValueError, match="use_moe"):
        validate_legacy_thinker_config(config)


def test_sparse_route_requires_topk_smaller_than_expert_count():
    config = Qwen3OmniMoeThinkerConfig(
        use_moe=True,
        num_experts=2,
        num_experts_per_tok=2,
        routing_kind="sparse",
    )
    with pytest.raises(ValueError, match="top_k < num_experts"):
        validate_legacy_thinker_config(config)


def test_explicit_dense_ensemble_allows_topk_equal_to_experts():
    config = Qwen3OmniMoeThinkerConfig(
        use_moe=True,
        num_experts=2,
        num_experts_per_tok=2,
        routing_kind="dense_ensemble",
    )
    validate_legacy_thinker_config(config)
```

- [ ] **Step 2: Run the tests and observe missing validation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/architecture/test_config_validation.py -q
```

Expected: FAIL because enum-backed routing validation functions do not exist.

- [ ] **Step 3: Implement strict parsing and validation**

The public enum and validator must use:

```python
class RoutingKind(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    DENSE_ENSEMBLE = "dense_ensemble"


def parse_layer_indices(
    value: str | None,
    *,
    layer_count: int,
    field: str,
) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    parts = value.split(",")
    if any(not part.strip().isdigit() for part in parts):
        raise ValueError(f"{field} must be a comma-separated integer list")
    result = tuple(int(part.strip()) for part in parts)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate layer indices")
    if any(index < 0 or index >= layer_count for index in result):
        raise ValueError(f"{field} contains an out-of-range layer index")
    return result
```

`validate_legacy_thinker_config()` must also enforce:

- `num_hidden_layers > 0`;
- attention and KV heads are positive and divisible as required;
- `0 < rope_partial_factor <= 1` at the top-level config boundary;
- explicit `moe_layer_indices` requires `use_moe=true`;
- explicit `deltanet_layer_indices` requires `use_deltanet=true`;
- `RoutingKind.SPARSE` requires `num_experts_per_tok < num_experts`;
- `RoutingKind.DENSE_ENSEMBLE` requires equality;
- `RoutingKind.DENSE` requires `use_moe=false`;
- global booleans and layer indices are never allowed to override one another.

The string field was introduced with the legacy adapter in Task 2. Convert it
to enum-backed validation here; the adapter continues to derive a canonical
string from old `use_moe`/top-k fields.

- [ ] **Step 4: Make all six checked-in YAML semantics explicit**

For `qwen3_omni_7b_moe.yaml`, retain two experts/top-2 but set:

```yaml
routing_kind: dense_ensemble
```

For `qwen3_omni_14b_moe.yaml` and `qwen3_omni_30b_moe.yaml`, preserve the
current standard non-TP layer-index behavior by setting:

```yaml
use_moe: true
routing_kind: sparse
```

Keep each file's existing `moe_layer_indices`. This intentionally fixes the TP
path that previously ignored those layers while leaving the ordinary path's
actual layer graph unchanged. Add a config test proving standard and TP
constructors choose the same MoE layers.

For `qwen3_omni_1_3b_moe.yaml`, `qwen3_omni_70b_moe.yaml`, and
`qwen3_omni_120b_moe.yaml`, retain `use_moe: true` and set
`routing_kind: sparse` because top-k is smaller than expert count. A
parametrized test loads every `qwen3_omni_*_moe.yaml` and proves no file relies
on the dataclass default or compatibility derivation.

- [ ] **Step 5: Validate in `Qwen3OmniMoeConfig.__init__`**

Call the validator after all compatibility field mappings and before any model
module is constructed. Make `thinker_config` authoritative and mirror its
`hidden_size`, `intermediate_size`, layer/head counts, max positions, MoE
counts, and MoE enablement back to the top-level serialized fields. Delete the
two ad-hoc layer-index parsers from standard and TP constructors and call
`parse_layer_indices()` in both paths.

- [ ] **Step 6: Run configuration, MoE, and TP tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/architecture/test_config_validation.py \
  tests/test_moe.py tests/test_tp_moe.py tests/test_model_stats.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/architecture/config_validation.py \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe \
  configs/model/qwen3_omni_1_3b_moe.yaml \
  configs/model/qwen3_omni_7b_moe.yaml \
  configs/model/qwen3_omni_14b_moe.yaml \
  configs/model/qwen3_omni_30b_moe.yaml \
  configs/model/qwen3_omni_70b_moe.yaml \
  configs/model/qwen3_omni_120b_moe.yaml \
  tests/architecture/test_config_validation.py tests/test_tp_moe.py
git commit -m "fix: validate architecture configuration explicitly"
```

---

### Task 4: Produce machine-readable architecture summaries

**Files:**
- Create: `src/qwen3_omni_pretrain/architecture/summary.py`
- Modify: `src/qwen3_omni_pretrain/utils/model_stats.py:1-126`
- Create: `scripts/inspect_architecture.py`
- Modify: `scripts/inspect_model_parameters.py:1-65`
- Create: `tests/architecture/test_architecture_summary.py`

**Interfaces:**
- Consumes: validated config, `ProfileManifest`, and existing `ParameterStats`.
- Produces: `LayerArchitecture`, `ArchitectureSummary`, `summarize_model(model, manifest)`, and a JSON CLI.

- [ ] **Step 1: Write a failing summary snapshot test**

```python
def test_legacy_summary_reports_real_layer_and_routing_types():
    config = Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": True,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "routing_kind": "sparse",
            "moe_layer_indices": "1",
        },
    )
    model = Qwen3OmniMoeThinkerTextModel(config)
    summary = summarize_model(model, config.profile_manifest)
    assert [layer.ffn_type for layer in summary.layers] == [
        "dense",
        "shared-dense-plus-routed-moe",
    ]
    assert summary.total_parameters > summary.active_parameters_per_token
    assert summary.profile == "legacy_prototype"
```

- [ ] **Step 2: Run the test and verify `summarize_model` is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/architecture/test_architecture_summary.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement focused summary dataclasses**

```python
@dataclass(frozen=True)
class LayerArchitecture:
    index: int
    attention_type: str
    cache_type: str
    ffn_type: str
    routed_experts: int
    experts_per_token: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "LayerArchitecture":
        require_exact_typed_keys(
            raw,
            ints=(
                "index",
                "routed_experts",
                "experts_per_token",
            ),
            strings=("attention_type", "cache_type", "ffn_type"),
        )
        return cls(**raw)


@dataclass(frozen=True)
class ArchitectureSummary:
    profile: str
    compatibility_level: str
    model_type: str
    tokenizer_vocab_size: int
    embedding_vocab_size: int
    total_parameters: int
    active_parameters_per_token: int
    routed_parameters: int
    shared_parameters: int
    dense_parameters: int
    capabilities: Mapping[str, bool]
    layers: tuple[LayerArchitecture, ...]
    unsupported_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(key, str) or type(value) is not bool
            for key, value in self.capabilities.items()
        ):
            raise TypeError("capabilities must map strings to booleans")
        object.__setattr__(
            self,
            "capabilities",
            MappingProxyType(dict(self.capabilities)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "compatibility_level": self.compatibility_level,
            "model_type": self.model_type,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "embedding_vocab_size": self.embedding_vocab_size,
            "total_parameters": self.total_parameters,
            "active_parameters_per_token": self.active_parameters_per_token,
            "routed_parameters": self.routed_parameters,
            "shared_parameters": self.shared_parameters,
            "dense_parameters": self.dense_parameters,
            "capabilities": dict(sorted(self.capabilities.items())),
            "layers": [layer.to_dict() for layer in self.layers],
            "unsupported_capabilities": list(
                self.unsupported_capabilities
            ),
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ArchitectureSummary":
        validated = validate_architecture_summary_mapping(raw)
        layers = validated.pop("layers")
        unsupported = validated.pop("unsupported_capabilities")
        capabilities = validated.pop("capabilities")
        return cls(
            **validated,
            layers=tuple(
                LayerArchitecture.from_dict(layer)
                for layer in layers
            ),
            capabilities=capabilities,
            unsupported_capabilities=tuple(unsupported),
        )
```

`summarize_model()` must inspect the built modules, not infer layer types from
the YAML filename. Extend `ParameterStats` with routed/shared/dense counts while
preserving its existing fields and tests.
`require_exact_typed_keys()`/`validate_architecture_summary_mapping()` copy
their input before returning it and reject
missing/unknown keys, booleans in integer fields, negative parameter counts,
non-contiguous layer indices, non-string capability entries and mutable
container types. Add a strict
`ArchitectureSummary.from_dict(summary.to_dict()) == summary` test plus
unknown-key/type-negative tests; checkpoint metadata must call this method
rather than constructing a partial summary.

`capabilities` is part of sidecar schema version 1 from the start. Later plans
may add string→boolean capability entries without adding top-level summary
fields, so older sidecars with `{}` remain readable and no silent schema
rewrite is required.

- [ ] **Step 4: Add one inspection CLI and retain the old command**

`scripts/inspect_architecture.py` accepts:

```text
MODEL_CONFIG
--json
--tokenizer TOKENIZER_PATH
```

It must:

1. load and adapt the legacy YAML;
2. reconcile tokenizer IDs when `--tokenizer` is supplied;
3. build on the meta device;
4. validate the manifest;
5. print the exact `ArchitectureSummary`.

Change `inspect_model_parameters.py` into a thin compatibility wrapper that
calls the new summary code and prints only its parameter fields.

- [ ] **Step 5: Snapshot all checked-in legacy model YAMLs**

Add a parametrized test over `configs/model/qwen3_omni_*_moe.yaml`. The
comment-only `configs/model/thinker_text.yaml` is not a model config and must
be rejected with `model configuration is empty`, not included in snapshots.
Explicitly assert:

- the 7B file is labeled `dense_ensemble`, not sparse;
- the 30B standard and TP layer summaries agree;
- no filename-derived parameter count appears in the summary;
- every summary's `profile` and `compatibility_level` equal the separately
  supplied legacy manifest; the manifest itself remains a single copy in the
  checkpoint sidecar.

- [ ] **Step 6: Run summary and CLI tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/architecture/test_architecture_summary.py \
  tests/test_model_stats.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/architecture/summary.py \
  src/qwen3_omni_pretrain/utils/model_stats.py \
  scripts/inspect_architecture.py scripts/inspect_model_parameters.py \
  tests/architecture/test_architecture_summary.py tests/test_model_stats.py
git commit -m "feat: report machine-readable architecture summaries"
```

---

### Task 5: Round-trip profile metadata through every checkpoint path

**Files:**
- Create: `src/qwen3_omni_pretrain/architecture/checkpoint_metadata.py`
- Modify: `src/qwen3_omni_pretrain/training/checkpoint.py:25-474`
- Modify: `src/qwen3_omni_pretrain/training/accelerator_utils.py:336-487`
- Modify: `src/qwen3_omni_pretrain/training/trainer_thinker.py`
- Create: `tests/architecture/test_checkpoint_metadata.py`

**Interfaces:**
- Consumes: `ProfileManifest` and `ArchitectureSummary`.
- Produces: `CheckpointMetadata`, `write_checkpoint_metadata()`, and
  `load_checkpoint_metadata()`.

- [ ] **Step 1: Write failing metadata round-trip tests**

```python
def test_checkpoint_metadata_round_trip_is_atomic(tmp_path):
    metadata = CheckpointMetadata(
        manifest=legacy_manifest(),
        architecture=toy_architecture_summary(),
        tokenizer_sha256="a" * 64,
        implementation_commit="dc71e7ca1d03666798ecdbee5143e132e49210f7",
    )
    write_checkpoint_metadata(tmp_path, metadata)
    assert load_checkpoint_metadata(tmp_path) == metadata
    assert not (tmp_path / "architecture.json.tmp").exists()


def test_nonlegacy_checkpoint_requires_metadata(tmp_path):
    with pytest.raises(ValueError, match="architecture.json"):
        load_checkpoint_metadata(
            tmp_path,
            expected_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        )
```

- [ ] **Step 2: Run and observe the missing metadata module**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/architecture/test_checkpoint_metadata.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement one sidecar schema**

```python
@dataclass(frozen=True)
class CheckpointMetadata:
    manifest: ProfileManifest
    architecture: ArchitectureSummary
    tokenizer_sha256: str
    implementation_commit: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "manifest": self.manifest.to_dict(),
            "architecture": self.architecture.to_dict(),
            "tokenizer_sha256": self.tokenizer_sha256,
            "implementation_commit": self.implementation_commit,
        }
```

Write `architecture.json` through a same-directory temporary file, `fsync`,
and `os.replace`. Validate schema version, manifest, SHA-256 length, expected
profile and compatibility level on load.

- [ ] **Step 4: Integrate all save mechanisms**

Every successful path must write the same sidecar:

- `atomic_save_checkpoint`;
- `save_checkpoint`;
- `save_checkpoint_accelerator`;
- `save_model_only_accelerator`;
- `save_tp_sharded_checkpoint`;
- `save_accelerator_checkpoint` in `training/accelerator_utils.py`;
- the DeepSpeed/full-checkpoint closure in `trainer_thinker.py`.

Only rank zero writes metadata after model files are durable. Other ranks wait
at the existing checkpoint barrier. Add monkeypatch tests for every entrypoint.

- [ ] **Step 5: Define old-checkpoint behavior**

Metadata absence is accepted only when the caller explicitly selects
`legacy_prototype`; emit one deprecation warning and synthesize the legacy
manifest from the adapted config. Reference and experimental profiles fail
before loading any tensors. Never write the synthesized metadata back into the
old checkpoint.

- [ ] **Step 6: Run checkpoint and import-safety tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/architecture/test_checkpoint_metadata.py \
  tests/test_checkpoint_transformers_compat.py \
  tests/test_trainer_import_safety.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/architecture/checkpoint_metadata.py \
  src/qwen3_omni_pretrain/training/checkpoint.py \
  src/qwen3_omni_pretrain/training/accelerator_utils.py \
  src/qwen3_omni_pretrain/training/trainer_thinker.py \
  tests/architecture/test_checkpoint_metadata.py
git commit -m "feat: persist architecture checkpoint metadata"
```

---

### Task 6: Pin and capture the official Qwen3-Omni oracle

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/__init__.py`
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/oracle.py`
- Create: `tests/fixtures/qwen3_omni/config_contract.json`
- Create: `tests/oracle/test_qwen3_omni_config_oracle.py`
- Create: `tests/oracle/test_qwen3_omni_processor_oracle.py`
- Modify: `tests/conftest.py:1-71`
- Modify: `pytest.ini`

**Interfaces:**
- Consumes: `ProfileManifest` and the pinned reference environment.
- Produces: `QWEN3_OMNI_MODEL_ID`, `QWEN3_OMNI_REVISION`, `QWEN3_TRANSFORMERS_VERSION`, `qwen3_reference_manifest()`, `load_reference_config()`, `load_reference_processor()`, and `load_reference_model()`.

- [ ] **Step 1: Write failing pin and fixture tests**

```python
def test_qwen3_oracle_pins_are_immutable():
    assert QWEN3_OMNI_MODEL_ID == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert QWEN3_OMNI_REVISION == (
        "26291f793822fb6be9555850f06dfe95f2d7e695"
    )
    assert QWEN3_TRANSFORMERS_VERSION == "5.2.0"
    manifest = qwen3_reference_manifest()
    assert manifest.architecture_profile.value == "qwen3_omni_reference"
    assert manifest.compatibility_level.value == "structure-aligned"
    assert not manifest.exact_official_checkpoint_compatible


def test_checked_in_contract_has_official_dimensions():
    contract = json.loads(
        Path("tests/fixtures/qwen3_omni/config_contract.json").read_text()
    )
    assert contract["thinker"] == {
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
    }
    assert contract["talker"]["num_hidden_layers"] == 20
    assert contract["code_predictor"]["num_hidden_layers"] == 5
```

- [ ] **Step 2: Run the contract test and observe missing oracle files**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/test_qwen3_omni_config_oracle.py -q
```

Expected: FAIL because the oracle module and fixture do not exist.

- [ ] **Step 3: Implement pinned, local-first loaders**

`oracle.py` must define:

```python
QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QWEN3_OMNI_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
QWEN3_TRANSFORMERS_VERSION = "5.2.0"


def _require_reference_transformers() -> None:
    if transformers.__version__ != QWEN3_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "qwen3_omni_reference requires transformers==5.2.0"
        )


def load_reference_config(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    local_files_only: bool = True,
):
    _require_reference_transformers()
    return Qwen3OmniMoeConfig.from_pretrained(
        source,
        revision=QWEN3_OMNI_REVISION,
        local_files_only=local_files_only,
    )
```

Implement processor and model loaders with the same `revision` and
`local_files_only` defaults. `load_reference_model()` must require the caller
to pass explicit `torch_dtype` and `device_map`; it must never select hardware
implicitly.

- [ ] **Step 4: Generate and review the small config contract**

Add a script-local extraction function that reads the official config and
writes only:

- regular and embedding vocab sizes;
- AuT convolution, layer, head, hidden and projector dimensions;
- ViT patch, merge, DeepStack, layer, head, hidden and projector dimensions;
- TM-RoPE section and theta;
- Thinker/Talker/MTP structural fields;
- Code2Wav codebook and output sample-rate fields.

Run it once in the reference environment against a locally cached config or an
explicitly approved network fetch. Commit the resulting JSON; do not commit
weights, tokenizer files, audio, or images.

- [ ] **Step 5: Add opt-in processor oracle tests**

Register markers:

```ini
markers =
    reference: requires the isolated official-reference environment
    large_model: requires locally available large model weights
```

Add `--run-large-model-tests` in `tests/conftest.py`. Skip `large_model` tests
unless the flag is present. Processor tests use one 1-second synthetic waveform
and one 32×32 deterministic image and assert:

- token IDs and media sentinel counts;
- audio feature mask and length;
- image grid shape;
- no network request when `local_files_only=True`.

The official processor does not produce position IDs. Add a separate
configuration-only RoPE oracle in the same test module: bind the pinned
official model class's `get_rope_index()` implementation to a lightweight
facade containing only the validated config fields it reads, feed the
processor's token IDs plus exact image/video/audio grid tensors, and assert
position axes, dtype/non-negativity, token-axis shape and returned rope delta.
The test hashes/checks the pinned implementation revision, never calls
`from_pretrained`, and allocates no model layers or weights. Multi-axis
spatial/time resets are compared exactly; monotonicity is required only for a
one-axis text fixture.

- [ ] **Step 6: Run fixture and processor tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/test_qwen3_omni_config_oracle.py \
  tests/oracle/test_qwen3_omni_processor_oracle.py -q
```

Expected: config, processor and configuration-only RoPE tests pass when artifacts are cached or
skip with an explicit cache-missing reason. No large weights are loaded.

`qwen3_reference_manifest()` is the source/provenance manifest and remains
`structure-aligned` in this task. Only the evidence verifier in the
reference-runtime plan may atomically promote a concrete build's manifest,
runtime outputs and summary to `checkpoint-compatible` after state, numerical,
cache, offline-text and offline-speech gates pass.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/profiles/qwen3_omni_reference \
  tests/fixtures/qwen3_omni tests/oracle tests/conftest.py pytest.ini
git commit -m "test: pin the official Qwen3 Omni oracle"
```

---

### Task 7: Add a lazy profile registry and profile inspection CLI

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/registry.py`
- Create: `src/qwen3_omni_pretrain/profiles/legacy_prototype/factory.py`
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/factory.py`
- Create: `src/qwen3_omni_pretrain/cli_profile.py`
- Modify: `src/qwen3_omni_pretrain/profiles/__init__.py`
- Modify: `README.md`
- Create: `tests/architecture/test_profile_registry.py`
- Create: `tests/test_cli_profile.py`

**Interfaces:**
- Consumes: profile enum, manifests, legacy config adapter, architecture summary, and Qwen3 oracle loader.
- Produces: `ProfileBuildRequest`, `ProfileBuildResult`, `ProfileFactory` protocol, `register_profile_factory()`, `get_profile_factory()`, and CLI subcommands `inspect` and `validate`.

- [ ] **Step 1: Write failing lazy-import tests**

```python
def test_registry_does_not_import_reference_backend_for_legacy(monkeypatch):
    imported = []
    real_import = importlib.import_module

    def capture(name, package=None):
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", capture)
    factory = get_profile_factory(ArchitectureProfile.LEGACY_PROTOTYPE)
    assert factory.profile is ArchitectureProfile.LEGACY_PROTOTYPE
    assert not any("qwen3_omni_reference" in name for name in imported)


def test_unknown_profile_fails_before_model_construction():
    with pytest.raises(ValueError, match="unknown architecture profile"):
        parse_profile("qwen3.5-omni-plus")
```

- [ ] **Step 2: Run the tests and observe missing registry failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/architecture/test_profile_registry.py tests/test_cli_profile.py -q
```

Expected: FAIL at registry import.

- [ ] **Step 3: Implement the build request and factory protocol**

```python
@dataclass(frozen=True)
class ProfileBuildRequest:
    profile: ArchitectureProfile
    config_or_checkpoint: str
    tokenizer: str | None = None
    local_files_only: bool = True
    dtype: str | None = None
    device: str | None = None
    requested_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileBuildResult:
    artifact: object
    manifest: ProfileManifest
    architecture_summary: ArchitectureSummary


class ProfileFactory(Protocol):
    profile: ArchitectureProfile

    def manifest(self, request: ProfileBuildRequest) -> ProfileManifest:
        ...

    def validate(self, request: ProfileBuildRequest) -> None:
        ...

    def build(self, request: ProfileBuildRequest) -> ProfileBuildResult:
        ...
```

The registry stores module paths and class names, not imported factory objects.
`get_profile_factory()` imports only the selected profile. Initially register
legacy and Qwen3 reference; later plans add Qwen3.5 and MiMo. `artifact` is
intentionally typed as `object`: legacy/Qwen3.5/MiMo builders return an
`nn.Module`, while the Qwen3 reference builder returns a non-module runtime
facade so official state-dict keys never gain a wrapper prefix. Callers must
inspect the manifest/capabilities or narrow the artifact protocol before use.

The legacy factory adapts YAML/checkpoint config, builds the existing
`Qwen3OmniMoeThinkerTextModel`, and returns its manifest/summary. The initial
Qwen3 reference factory is an oracle factory: `validate`/`manifest`/`inspect`
work without weights, and `build` returns a `Qwen3OracleArtifact` containing
the pinned official config contract plus lazy config/processor loaders; it
does not pretend to be an inference model. The later reference-runtime plan
replaces that artifact with the non-module runtime facade after strict weight
loading. No registered factory has an error-only `build` implementation.

```python
@dataclass(frozen=True)
class Qwen3OracleArtifact:
    config_contract: Mapping[str, object]
    load_config: Callable[[], object]
    load_processor: Callable[[], object]
```

- [ ] **Step 4: Implement read-only CLI commands**

Commands:

```bash
python -m qwen3_omni_pretrain.cli_profile validate \
  --profile legacy_prototype \
  --config-or-checkpoint configs/model/qwen3_omni_1_3b_moe.yaml

python -m qwen3_omni_pretrain.cli_profile inspect \
  --profile legacy_prototype \
  --config-or-checkpoint configs/model/qwen3_omni_1_3b_moe.yaml \
  --json
```

`validate` prints the manifest and exits before allocating weights. `inspect`
uses meta-device construction. Unsupported capabilities are printed from the
summary and never silently enabled.

- [ ] **Step 5: Document identity and compatibility labels**

Update the README capability table with one row per profile. At this phase:

- legacy: implemented, `legacy-prototype`;
- Qwen3 reference: oracle/config implemented, runtime pending,
  `structure-aligned`;
- Qwen3.5: planned, `paper-inspired`;
- MiMo: planned, `MiMo-style-experiment`.

State that new legacy saves use `qwen3_omni_prototype`; old saves are only read
through the explicit adapter.

- [ ] **Step 6: Run focused and full suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/architecture tests/test_cli_profile.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q
```

Expected: focused tests pass and the full prototype suite has zero failures.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/profiles/registry.py \
  src/qwen3_omni_pretrain/profiles/legacy_prototype/factory.py \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/factory.py \
  src/qwen3_omni_pretrain/profiles/__init__.py \
  src/qwen3_omni_pretrain/cli_profile.py \
  tests/architecture/test_profile_registry.py tests/test_cli_profile.py README.md
git commit -m "feat: register isolated architecture profiles"
```

---

## Plan completion gate

Run all checks from a clean worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q

PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/test_qwen3_omni_config_oracle.py \
  tests/oracle/test_qwen3_omni_processor_oracle.py -q

git diff --check
git status --short
```

The plan is complete only when:

- new configs no longer serialize the official-colliding model type;
- old legacy configs load through one explicit warning-producing adapter;
- every checked-in YAML validates or has an explicit non-sparse routing label;
- architecture summaries report real layer types and total/active parameters;
- official Qwen3 revisions and config contract are pinned;
- ordinary tests do not fetch weights or import the reference backend;
- both test commands have zero failures, except explicit cache-missing skips in
  the reference processor suite;
- the worktree contains only intentional committed changes.
