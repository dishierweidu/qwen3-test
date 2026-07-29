# Stage-Aware Training, Evaluation, and Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为新架构 profile 建立可审计的分阶段训练、确定性续训、原生 Stage-2 DDP、渐进长上下文、统一评测门禁和带媒体缓存的并发服务；蒸馏、偏好与 reward 优化只能在 SFT 门禁通过后启用。

**Architecture:** 新建 `training.program` 控制面，使用不可变的 stage/program/progress 数据结构驱动 profile factory、参数组、数据 mixture、checkpoint、evaluation 和 serving。现有 `trainer_thinker.py` 与 `cli_train_thinker.py` 继续服务 `legacy_prototype`，其 Stage-2 分布式 fail-fast 不修改；新 `cli_train_profile.py` 提供 profile-aware 单进程与 DDP 路径。评测结果与训练 checkpoint 都绑定完整 manifest/hash，服务复用公共 `DecoderState` 并用内容寻址媒体缓存隔离 preprocessing。

**Tech Stack:** Python 3.10, dataclasses, PyYAML, PyTorch 2.10.0, torch.distributed Gloo/NCCL, NumPy, pytest, JSONL/JSON reports, asyncio.

## Global Constraints

- 本计划依赖前六个计划全部完成。
- `legacy_prototype` 的现有训练 CLI、配置和 Stage-2 单卡行为必须保持；新 profile 只能通过 `cli_train_profile.py` 进入新 runner。
- `qwen3_omni_reference` 默认是只读 reference runtime；未显式提供可训练 `nn.Module` artifact 时，训练入口必须在分配 optimizer 前失败。
- 本计划的通用 distributed stage runner 第一版只实现原生 DP（TP/PP/EP
  均为 1）；MiMo 的 EP=2 correctness/gradient 路径由其专用两进程计划
  验证，不在本计划中伪装成可恢复的端到端 stage training。
- Stage 顺序固定为 projector warmup、encoder+projector warmup、general multimodal SFT、long-context SFT；distillation、preference、reward optimization 是后置可选阶段。
- 每个 stage 明确列出可训练参数组、数据 mixture、context length、optimizer/scheduler、进入门禁和退出门禁；缺字段不能使用隐式默认进入训练。
- freeze/unfreeze 由 factory 提供的稳定参数组驱动，禁止用脆弱的名称子串猜测模块身份。
- checkpoint 必须保存 model、optimizer、scheduler、scaler、stage progress、Python/NumPy/PyTorch RNG、sampler cursor、公共 `CheckpointMetadata`（manifest、architecture summary、tokenizer SHA、implementation commit）、program hash、data/collator fingerprint 和 world size。
- 相同 config/data/world size 的中断续训必须产生与不中断训练相同的 sample 顺序、global step 和下一步 loss；world size 改变默认拒绝恢复。
- 新 Stage-2 DDP 第一版支持 `DP>=1, TP=1, PP=1, EP=1`；MiMo EP 使用其独立实验计划，不在本计划中叠加。
- distributed sampler 必须证明每个 epoch 的 global mixture draw position
  在各 rank 间恰好出现一次；加权 mixture 可以按显式 quota 循环某个
  source 的原始样本。为整除 world size 而增加的 padding position 必须
  单独标记并从 coverage 指标排除。
- long-context 默认验证阶梯是 4K、16K、32K；64K、128K、256K/262K 只能在硬件、数据和前一级门禁通过后作为 opt-in 阶段启用。
- manifest 的 `validated_context_length` 只能由评测门禁提升；仅设置更大的 RoPE/context 配置不能提升能力声明。
- 评测门禁同时覆盖文本、图像、音频、视频、cache、媒体 shuffle/ablation、MoE/MTP（适用时）和非有限值回归。
- 服务中的媒体缓存 key 必须包含内容 hash、模态、processor/encoder/projector revision、dtype 和关键 preprocessing 参数；训练路径禁止使用跨 batch 全局媒体缓存。
- 服务 benchmark 必须报告 1/4/8 并发、TTFT、TTFC、RTF、decode tok/s、峰值内存、取消/背压，以及媒体 cache hit/miss 的 P50/P90。
- distillation、DPO 和 reward objective 只能消费已通过 SFT exit gates 的 checkpoint；本计划不声称复现未公开 reward data、生产 RL 系统或论文最终质量。
- 普通测试不得联网、下载大权重或要求 GPU；分布式单测使用两进程 Gloo，硬件 benchmark 显式 opt-in。
- Snippet 中的 `...`/`NotImplementedError` 只表示 Protocol/签名节选；
  任务提交不得保留未实现 stub，必须由该任务列出的行为测试证明可用。
- 每个行为变更使用 red-green-refactor，并形成独立提交。

---

## File responsibility map

- `training/stages.py`: stage kind、context step、parameter policy、stage/program/progress contracts。
- `training/stage_config.py`: strict YAML parser、canonical serialization 和 program hash。
- `training/parameter_policy.py`: factory-owned parameter groups、freeze/unfreeze 和 audit。
- `data/stage_mixture.py`: modality/source mixture 和 deterministic global index stream。
- `data/stage_sampler.py`: rank slicing、padding metadata 和 resumable cursor。
- `training/rng_state.py`: Python/NumPy/PyTorch CPU/CUDA RNG capture/restore。
- `training/stage_checkpoint.py`: 原子 stage checkpoint、manifest/program/data compatibility。
- `training/stage_runner.py`: 单 stage 生命周期、loss step、evaluation gate 和 transition。
- `training/distributed_stage.py`: DP process group、DDP wrapper、collective metrics 和 rank-safe failure。
- `cli_train_profile.py`: profile-aware training CLI；不转发到 legacy CLI。
- `evaluation/registry.py`: metric/evaluator 注册，禁止静默覆盖。
- `evaluation/stage_gates.py`: typed gate comparison、聚合和 transition decision。
- `evaluation/profile_suite.py`: text/media/cache/MoE/MTP 评测编排和 JSON report。
- `serving/contracts.py`: request、event、timing 和 cancellation contracts。
- `serving/media_cache.py`: byte-bounded content-addressed LRU。
- `serving/scheduler.py`: bounded async admission、prefill/decode scheduling。
- `serving/engine.py`: one-time media prefill、token decode 和 stream events。
- `loss/distillation.py`: masked temperature-scaled KL。
- `loss/preference.py`: DPO log-ratio objective。
- `training/reward_protocol.py`: reward provider、group advantage 和 clipped policy objective。

---

### Task 1: Define strict training program and stage progress contracts

**Files:**
- Create: `src/qwen3_omni_pretrain/training/stages.py`
- Create: `src/qwen3_omni_pretrain/training/stage_config.py`
- Create: `tests/training/test_stage_config.py`
- Create: `configs/train/profiles/qwen35_tiny_program.yaml`
- Modify: `src/qwen3_omni_pretrain/training/__init__.py`

- [ ] **Step 1: Write failing stage-config tests**

```python
from dataclasses import replace

import pytest

from qwen3_omni_pretrain.architecture.profiles import ArchitectureProfile
from qwen3_omni_pretrain.training.stage_config import (
    load_training_program,
    training_program_hash,
)
from qwen3_omni_pretrain.training.stages import TrainingStageKind


def test_program_requires_ordered_warmup_sft_and_context_stages(tmp_path):
    path = tmp_path / "program.yaml"
    path.write_text(
        """
profile: qwen35_omni_inspired
data_sources:
  text:
    modality: text
    train_path: data/text.jsonl
    format: jsonl
    fingerprint: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  image_text:
    modality: image_text
    train_path: data/image_text.jsonl
    format: jsonl
    fingerprint: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  audio_text:
    modality: audio_text
    train_path: data/audio_text.jsonl
    format: jsonl
    fingerprint: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  video_text:
    modality: video_text
    train_path: data/video_text.jsonl
    format: jsonl
    fingerprint: dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
stages:
  - name: projector
    kind: projector_warmup
    trainable_groups: [projector]
    max_steps: 2
    context_length: 4096
    micro_batch_size: 1
    gradient_accumulation_steps: 2
    precision: float32
    gradient_clip_norm: 1.0
    checkpoint_interval_steps: 1
    num_workers: 0
    epoch_size: 4
    optimizer: {name: adamw, learning_rate: 1.0e-4, weight_decay: 0.01}
    scheduler: {name: linear, warmup_ratio: 0.0}
    data_mixture: {text: 1, image_text: 1, audio_text: 1}
    objective: null
    entry_gates: []
    exit_gates: [finite_loss]
  - name: encoder_projector
    kind: encoder_projector_warmup
    trainable_groups: [vision_encoder, audio_encoder, projector]
    max_steps: 2
    context_length: 4096
    micro_batch_size: 1
    gradient_accumulation_steps: 2
    precision: float32
    gradient_clip_norm: 1.0
    checkpoint_interval_steps: 1
    num_workers: 0
    epoch_size: 4
    optimizer: {name: adamw, learning_rate: 5.0e-5, weight_decay: 0.01}
    scheduler: {name: linear, warmup_ratio: 0.0}
    data_mixture: {text: 1, image_text: 1, audio_text: 1, video_text: 1}
    objective: null
    entry_gates: [finite_loss]
    exit_gates: [tiny_overfit]
  - name: multimodal_sft
    kind: multimodal_sft
    trainable_groups: [thinker, vision_encoder, audio_encoder, projector]
    max_steps: 4
    context_length: 4096
    micro_batch_size: 1
    gradient_accumulation_steps: 2
    precision: float32
    gradient_clip_norm: 1.0
    checkpoint_interval_steps: 2
    num_workers: 0
    epoch_size: 8
    optimizer: {name: adamw, learning_rate: 1.0e-5, weight_decay: 0.01}
    scheduler: {name: cosine, warmup_ratio: 0.03}
    data_mixture: {text: 4, image_text: 2, audio_text: 2, video_text: 1}
    objective: null
    entry_gates: [tiny_overfit]
    exit_gates: [multimodal_regression, context_4k]
  - name: context_16k
    kind: long_context_sft
    trainable_groups: [thinker, projector]
    max_steps: 2
    context_length: 16384
    micro_batch_size: 1
    gradient_accumulation_steps: 2
    precision: float32
    gradient_clip_norm: 1.0
    checkpoint_interval_steps: 1
    num_workers: 0
    epoch_size: 4
    optimizer: {name: adamw, learning_rate: 5.0e-6, weight_decay: 0.01}
    scheduler: {name: cosine, warmup_ratio: 0.03}
    data_mixture: {text: 4, image_text: 2, audio_text: 2, video_text: 1}
    objective: null
    entry_gates: [multimodal_regression, context_4k]
    exit_gates: [context_16k]
""",
        encoding="utf-8",
    )

    program = load_training_program(path, repository_root=tmp_path)

    assert program.profile is ArchitectureProfile.QWEN35_OMNI_INSPIRED
    assert [stage.kind for stage in program.stages] == [
        TrainingStageKind.PROJECTOR_WARMUP,
        TrainingStageKind.ENCODER_PROJECTOR_WARMUP,
        TrainingStageKind.MULTIMODAL_SFT,
        TrainingStageKind.LONG_CONTEXT_SFT,
    ]
    assert len(training_program_hash(program)) == 64


def test_program_rejects_advanced_stage_without_sft_gate(valid_program):
    advanced = replace(
        valid_program.stages[-1],
        name="preference",
        kind=TrainingStageKind.PREFERENCE,
        entry_gates=(),
    )
    with pytest.raises(ValueError, match="requires an SFT exit gate"):
        replace(valid_program, stages=valid_program.stages + (advanced,)).validate()


def test_program_rejects_context_regression(valid_program):
    regressed = replace(valid_program.stages[-1], context_length=2048)
    with pytest.raises(ValueError, match="non-decreasing"):
        replace(
            valid_program,
            stages=valid_program.stages[:-1] + (regressed,),
        ).validate()
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```bash
pytest -q tests/training/test_stage_config.py
```

Expected: FAIL because `training.stages` and `training.stage_config` do not exist.

- [ ] **Step 3: Implement immutable stage contracts and validation**

```python
class TrainingStageKind(str, Enum):
    PROJECTOR_WARMUP = "projector_warmup"
    ENCODER_PROJECTOR_WARMUP = "encoder_projector_warmup"
    MULTIMODAL_SFT = "multimodal_sft"
    LONG_CONTEXT_SFT = "long_context_sft"
    DISTILLATION = "distillation"
    PREFERENCE = "preference"
    REWARD_OPTIMIZATION = "reward_optimization"


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerSpec:
    name: str
    warmup_ratio: float


@dataclass(frozen=True)
class DataSourceSpec:
    name: str
    modality: str
    train_path: str
    validation_path: str | None
    format: Literal["jsonl"]
    fingerprint: str


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: TrainingStageKind
    trainable_groups: tuple[str, ...]
    max_steps: int
    context_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    precision: Literal["float32", "bfloat16", "float16"]
    gradient_clip_norm: float
    checkpoint_interval_steps: int
    num_workers: int
    epoch_size: int
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec
    data_mixture: Mapping[str, int]
    objective: Mapping[str, object] | None
    entry_gates: tuple[str, ...]
    exit_gates: tuple[str, ...]


@dataclass(frozen=True)
class TrainingProgram:
    profile: ArchitectureProfile
    data_sources: Mapping[str, DataSourceSpec]
    stages: tuple[StageSpec, ...]
    schema_version: int = 1

    def validate(self) -> "TrainingProgram":
        # Check unique names, positive runtime values, source references,
        # known optimizer/scheduler, required warmup/SFT ordering, monotonic
        # context, and advanced-stage gates.
        return self


@dataclass(frozen=True)
class StageProgress:
    stage_index: int
    stage_name: str
    epoch: int
    global_step: int
    stage_step: int
    samples_consumed: int
    best_metric: float | None
```

`load_training_program(path, *, repository_root)` must reject unknown keys,
boolean values where
integers are expected, missing gates, an empty mixture, duplicate stage names,
unsupported trainable groups, a mixture key absent from `data_sources`, an
invalid SHA-256 fingerprint, and a profile name not present in
`ArchitectureProfile`. Paths must be relative to the program file, resolve
inside the explicit canonical `repository_root` selected by the CLI, and are
included in the canonical program hash; machine-specific absolute paths are
rejected. Resolution is
`(path.parent / declared_path).resolve(strict=False)`, followed by an
`is_relative_to(repository_root.resolve())` check before any file is opened.
Tests create `tmp_path/data/*.jsonl` and pass `repository_root=tmp_path`; the
checked-in CLI passes the discovered Git repository root. There is no
current-working-directory fallback.
`training_program_hash()` hashes canonical JSON with sorted mapping keys.

`max_steps` counts optimizer updates, not microbatches.
`gradient_accumulation_steps`, `micro_batch_size`,
`checkpoint_interval_steps`, `epoch_size` and `context_length` are positive;
`gradient_clip_norm` is finite/non-negative. FP16 requires a serialized
`GradScaler`; BF16 and FP32 prohibit one. The deterministic first release
requires `num_workers == 0`; enabling prefetched workers is a future schema
version because worker queue state is not checkpointed.

`StageSpec.__post_init__()` copies `data_mixture` and any objective mapping into
`MappingProxyType`, and
`TrainingProgram.__post_init__()` requires an actual tuple of stages and copies
both `data_sources` and nested source specs before validation. No frozen
dataclass may retain a caller-owned mutable list or mapping that could change
the program hash after checkpoint creation.

- [ ] **Step 4: Add a complete tiny program and make tests pass**

The checked-in YAML must contain all four mandatory phases and explicit metric gate names; it may reference synthetic fixture data, but it must not contain machine-specific absolute paths.

Run:

```bash
pytest -q tests/training/test_stage_config.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qwen3_omni_pretrain/training/stages.py \
  src/qwen3_omni_pretrain/training/stage_config.py \
  src/qwen3_omni_pretrain/training/__init__.py \
  tests/training/test_stage_config.py \
  configs/train/profiles/qwen35_tiny_program.yaml
git commit -m "feat: define stage-aware training programs"
```

---

### Task 2: Make parameter freezing factory-owned and auditable

**Files:**
- Create: `src/qwen3_omni_pretrain/training/parameter_policy.py`
- Create: `src/qwen3_omni_pretrain/profiles/training.py`
- Modify: `src/qwen3_omni_pretrain/profiles/legacy_prototype/factory.py`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/runtime.py`
- Modify: `src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/factory.py`
- Modify: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/modeling_hybrid_swa_moe.py`
- Create: `tests/training/test_parameter_policy.py`

- [ ] **Step 1: Write failing group ownership and audit tests**

```python
import pytest
import torch

from qwen3_omni_pretrain.training.parameter_policy import (
    apply_parameter_policy,
    audit_parameters,
)


class GroupedToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = torch.nn.Linear(2, 2)
        self.projector = torch.nn.Linear(2, 2)

    def named_parameter_groups(self):
        return {
            "thinker": tuple(self.thinker.parameters()),
            "projector": tuple(self.projector.parameters()),
        }


def test_projector_warmup_freezes_every_other_parameter():
    model = GroupedToy()

    audit = apply_parameter_policy(model, ("projector",))

    assert audit.trainable_groups == ("projector",)
    assert all(parameter.requires_grad for parameter in model.projector.parameters())
    assert not any(parameter.requires_grad for parameter in model.thinker.parameters())
    assert audit.trainable_parameters + audit.frozen_parameters == audit.total_parameters


def test_policy_rejects_unowned_or_multiply_owned_parameters():
    model = GroupedToy()
    model.unowned = torch.nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="unowned"):
        audit_parameters(model)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
pytest -q tests/training/test_parameter_policy.py
```

Expected: FAIL because `parameter_policy.py` does not exist.

- [ ] **Step 3: Implement the trainable profile protocol and exact audit**

```python
@runtime_checkable
class GroupedTrainable(Protocol):
    def named_parameters(self, prefix: str = "", recurse: bool = True):
        raise NotImplementedError

    def named_parameter_groups(
        self,
    ) -> Mapping[str, tuple[torch.nn.Parameter, ...]]:
        raise NotImplementedError


@dataclass(frozen=True)
class ProfileTrainingComponents:
    model: GroupedTrainable
    build_collator: Callable[
        [StageSpec],
        Callable[[Sequence[object]], object],
    ]
    loss_keys: tuple[str, ...]
    manifest: ProfileManifest
    architecture_summary: ArchitectureSummary
    tokenizer_sha256: str

    def validate(self) -> None:
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("training model must be an nn.Module")
        if not self.loss_keys:
            raise ValueError("at least one loss key is required")
        if (
            self.architecture_summary.profile
            != self.manifest.architecture_profile.value
        ):
            raise ValueError("architecture summary and manifest profile differ")
        require_sha256(self.tokenizer_sha256, "tokenizer_sha256")
        audit_parameters(self.model)


@runtime_checkable
class TrainableProfileFactory(Protocol):
    def build_training(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileTrainingComponents:
        raise NotImplementedError


@dataclass(frozen=True)
class ParameterAudit:
    trainable_groups: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int
    total_parameters: int
    group_parameters: Mapping[str, int]
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
```

`ProfileTrainingComponents` and `TrainableProfileFactory` live in
`profiles/training.py`; the grouped protocol and audit live in
`training/parameter_policy.py`. `audit_parameters()` must match parameters by
object identity, prove every parameter belongs to exactly one group, and
return stable sorted names. `apply_parameter_policy()` must validate every
requested group, set all `requires_grad` flags in one pass, clear stale
gradients on newly frozen parameters, then return a fresh audit.

`build_collator(stage)` is the sole public collator construction path. It
receives the complete immutable `StageSpec`, must enforce
`stage.context_length` (including the profile's explicit truncation/rejection
policy), and returns the batch type required by that stage kind. Its canonical
fingerprint includes profile, tokenizer SHA, stage kind, context length and
collator revision. This prevents the loader from silently using a 4K collator
for a promoted 16K/32K stage and leaves Task 9 a typed extension point for
advanced batches.

- [ ] **Step 4: Add explicit groups to each trainable factory**

Required group names are:

```text
thinker
vision_encoder
audio_encoder
projector
talker
codec
mtp
```

A profile may omit unsupported groups, but cannot invent aliases in YAML. The
legacy adapter exposes its current modules without renaming checkpoint keys.
The Qwen3 reference factory intentionally does not satisfy
`TrainableProfileFactory`. The Qwen3.5 and MiMo factories implement
`build_training()`; the MiMo generic model maps its decoder to `thinker` and
MTP heads to `mtp`.

The concrete model/runtime classes—not factory monkeypatches—implement
`named_parameter_groups()`: legacy
`Qwen3OmniMoeThinkerVisionAudioModel`, final
`Qwen35InspiredRuntime`, and `HybridSwaMoeForCausalLM`. Each implementation
walks registered submodules, deduplicates shared parameters by object identity,
and calls the common complete/disjoint audit. Factory integration tests build
all three tiny artifacts and prove their unions equal `named_parameters()`
exactly; this catches parameters introduced after factory construction.

- [ ] **Step 5: Test freeze/unfreeze transitions**

Add a test that applies projector-only, encoder+projector, then multimodal-SFT policies to the same object and verifies both `requires_grad` and gradient clearing at every transition.

Run:

```bash
pytest -q tests/training/test_parameter_policy.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_omni_pretrain/training/parameter_policy.py \
  src/qwen3_omni_pretrain/profiles/training.py \
  src/qwen3_omni_pretrain/profiles/legacy_prototype/factory.py \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/runtime.py \
  src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/factory.py \
  src/qwen3_omni_pretrain/models/hybrid_swa_moe/modeling_hybrid_swa_moe.py \
  tests/training/test_parameter_policy.py
git commit -m "feat: audit profile training parameter groups"
```

---

### Task 3: Build deterministic modality mixtures and resumable distributed sampling

**Files:**
- Create: `src/qwen3_omni_pretrain/data/source_registry.py`
- Create: `src/qwen3_omni_pretrain/data/stage_mixture.py`
- Create: `src/qwen3_omni_pretrain/data/stage_sampler.py`
- Create: `src/qwen3_omni_pretrain/data/stage_loader.py`
- Create: `tests/data/test_stage_mixture.py`
- Create: `tests/data/test_stage_sampler.py`
- Create: `tests/data/test_stage_loader.py`
- Create: `tests/fixtures/stage_data/text.jsonl`
- Create: `tests/fixtures/stage_data/image_text.jsonl`
- Create: `tests/fixtures/stage_data/audio_text.jsonl`
- Create: `tests/fixtures/stage_data/video_text.jsonl`
- Modify: `configs/train/profiles/qwen35_tiny_program.yaml`
- Modify: `src/qwen3_omni_pretrain/data/__init__.py`

- [ ] **Step 1: Write failing deterministic mixture tests**

```python
from qwen3_omni_pretrain.data.stage_mixture import (
    MixtureSource,
    StageMixture,
)
from qwen3_omni_pretrain.data.stage_sampler import (
    ResumableDistributedStageSampler,
)


def test_mixture_is_seeded_and_preserves_declared_quota():
    mixture = StageMixture(
        sources=(
            MixtureSource("text", length=5, weight=4),
            MixtureSource("image_text", length=3, weight=2),
            MixtureSource("audio_text", length=2, weight=2),
            MixtureSource("video_text", length=1, weight=1),
        ),
        epoch_size=18,
        seed=17,
    )

    first = tuple(mixture.global_indices(epoch=2))
    second = tuple(mixture.global_indices(epoch=2))

    assert first == second
    assert len(first) == 18
    assert {item.source_name for item in first} == {
        "text", "image_text", "audio_text", "video_text"
    }


def test_rank_slices_reconstruct_global_order_without_padding():
    mixture = StageMixture.single_source(length=10, seed=3)
    rank_zero = ResumableDistributedStageSampler(
        mixture, rank=0, world_size=2, epoch=0
    )
    rank_one = ResumableDistributedStageSampler(
        mixture, rank=1, world_size=2, epoch=0
    )

    interleaved = [
        item
        for pair in zip(tuple(rank_zero), tuple(rank_one))
        for item in pair
    ]
    assert [item.global_position for item in interleaved] == list(range(10))
    assert not any(item.is_padding for item in interleaved)


def test_sampler_state_resumes_at_exact_next_global_position():
    sampler = ResumableDistributedStageSampler(
        StageMixture.single_source(length=12, seed=9),
        rank=0,
        world_size=1,
        epoch=1,
    )
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(5)]
    restored = ResumableDistributedStageSampler.from_state_dict(
        sampler.mixture, sampler.state_dict()
    )

    assert consumed[-1].global_position == 4
    assert next(iter(restored)).global_position == 5
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/data/test_stage_mixture.py tests/data/test_stage_sampler.py \
  tests/data/test_stage_loader.py
```

Expected: FAIL because the mixture and sampler modules do not exist.

- [ ] **Step 3: Implement a global draw stream before rank slicing**

```python
@dataclass(frozen=True)
class MixtureSource:
    name: str
    length: int
    weight: int


@dataclass(frozen=True)
class MixtureIndex:
    source_name: str
    source_index: int
    global_position: int
    is_padding: bool = False


@dataclass(frozen=True)
class SamplerCursor:
    epoch: int
    next_global_position: int
    rank: int
    world_size: int
    mixture_fingerprint: str
```

Use a local `torch.Generator` seeded from `(seed, epoch)`; build a
deterministic weighted source schedule with integer quotas, then independently
permute/cycle each source. Pad only the tail to a world-size multiple. Rank
`r` is issued positions `r, r + world_size, ...`. The sampler maintains
separate issued and committed cursors. A batch carries an opaque ordered
`SamplerBatchToken`; after forward/backward has consumed it, the runner calls
`mark_consumed(token)`. Only then does `next_global_position` advance.
`state_dict()` fails while any issued batch is uncommitted, and checkpoints are
allowed only at optimizer-update boundaries after the last microbatch has been
committed and gradients have been stepped/zeroed. State restoration validates
rank, world size and mixture fingerprint.

Integer quotas use largest-remainder allocation: take
`floor(epoch_size * weight / total_weight)`, then assign remaining draws by
descending fractional remainder with declared source order as the tie-break.
This makes every source count auditable and avoids Python hash-order effects.

`source_registry.py` strictly resolves each `DataSourceSpec`, recomputes the
JSONL content SHA-256 before creating a `torch.utils.data.Dataset`, and rejects
unknown formats/schema fields. `stage_loader.py` constructs a real
`DataLoader` over the mixture dataset with the resumable batch sampler,
the exact `components.build_collator(stage)` result, declared microbatch size,
`shuffle=False` and `num_workers=0`. The loader records and validates the
collator fingerprint, passes `stage.context_length` through that factory, and
rejects any emitted token/position/mask sequence exceeding the declared
length. No CLI creates synthetic samples or an implicit source from a mixture
name. A changed file, source path, stage context length or collator fingerprint
fails before model allocation.
Replace the illustrative repeated-character fingerprints from Task 1 with the
actual fixture SHA-256 values in this same task and add a snapshot test. Thus
Task 1 validates syntax without opening data, while Task 3 makes the program
executable.

- [ ] **Step 4: Add coverage, padding and epoch-change properties**

Tests must cover:

- odd dataset size with world size 2;
- padding rows marked `is_padding=True`;
- no non-padding global position duplicated across ranks;
- changing the epoch changes order but not coverage;
- empty sources and non-positive weights fail at construction;
- resume does not replay the last consumed item.
- saving with an issued-but-unconsumed batch fails, while saving immediately
  after `mark_consumed()` resumes at the exact next batch;
- every checked-in `DataSourceSpec` constructs a real Dataset/DataLoader and a
  changed fixture byte invalidates its fingerprint.
- 4K and 16K `StageSpec` instances build distinct collator fingerprints and
  enforce their exact limits; resuming with a different context/collator
  fingerprint fails before reading model tensors.

Run:

```bash
pytest -q tests/data/test_stage_mixture.py tests/data/test_stage_sampler.py \
  tests/data/test_stage_loader.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qwen3_omni_pretrain/data/source_registry.py \
  src/qwen3_omni_pretrain/data/stage_mixture.py \
  src/qwen3_omni_pretrain/data/stage_sampler.py \
  src/qwen3_omni_pretrain/data/stage_loader.py \
  src/qwen3_omni_pretrain/data/__init__.py \
  configs/train/profiles/qwen35_tiny_program.yaml \
  tests/data/test_stage_mixture.py tests/data/test_stage_sampler.py \
  tests/data/test_stage_loader.py tests/fixtures/stage_data
git commit -m "feat: add deterministic stage data sampling"
```

---

### Task 4: Save all state required for bitwise training resume

**Files:**
- Create: `src/qwen3_omni_pretrain/training/rng_state.py`
- Create: `src/qwen3_omni_pretrain/training/stage_checkpoint.py`
- Modify: `src/qwen3_omni_pretrain/training/checkpoint.py`
- Create: `tests/training/test_stage_checkpoint.py`
- Create: `tests/training/test_deterministic_resume.py`

- [ ] **Step 1: Write failing RNG and checkpoint round-trip tests**

```python
import random

import numpy as np
import torch

from qwen3_omni_pretrain.training.rng_state import (
    capture_rng_state,
    restore_rng_state,
)


def test_rng_restore_reproduces_all_next_draws():
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    state = capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    restore_rng_state(state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
```

The stage checkpoint test must save and restore a tiny model, AdamW, scheduler,
`StageProgress`, sampler state, the common `CheckpointMetadata`
(`ProfileManifest` + `ArchitectureSummary` + tokenizer SHA), program hash and
data fingerprint. It must reject a different program hash, data fingerprint,
profile, architecture summary, tokenizer hash, or world size before mutating
the caller's model.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/training/test_stage_checkpoint.py \
  tests/training/test_deterministic_resume.py
```

Expected: FAIL because the stage checkpoint modules do not exist.

- [ ] **Step 3: Implement complete checkpoint metadata**

```python
@dataclass(frozen=True)
class RngState:
    python_state: object
    numpy_state: tuple[object, ...]
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class StageCheckpointMetadata:
    schema_version: int
    architecture_metadata_sha256: str
    checkpoint_content_digest: str
    program_hash: str
    data_fingerprint: str
    collator_fingerprint: str
    world_size: int
    progress: StageProgress
    rank_sampler_state_files: Mapping[int, str]
    rank_rng_state_files: Mapping[int, str]
    optimizer_update_boundary: bool
```

`save_stage_checkpoint()` accepts an already validated common
`CheckpointMetadata` constructed from `ProfileTrainingComponents`, writes its
`architecture.json` through `write_checkpoint_metadata()`, hashes that exact
file into `architecture_metadata_sha256`, and writes stage metadata plus state
into a sibling temporary directory. It fsyncs metadata and payload files,
validates the temporary artifact, rotates the old target to `.backup`, then
atomically renames the whole directory. `load_stage_checkpoint()` first calls
the common `load_checkpoint_metadata()`, verifies the architecture-sidecar
hash plus every stage compatibility field into CPU memory, and only after all
checks pass may it mutate model/optimizer/scheduler/scaler or restore RNG.
There is no second embedded manifest representation.

`checkpoint_content_digest` is a stable digest over immutable training payload
files only: model, optimizer, scheduler, scaler (when applicable), rank RNG
shards, sampler shards and their canonical file inventory. It deliberately
excludes `architecture.json`, `stage_checkpoint.json`, evaluation reports and
context-promotion receipts. Those mutable metadata files are instead bound by
their own SHA fields. This stable boundary lets Task 7 promote a validated
context claim without pretending that model/optimizer bytes changed.

Reuse low-level atomic helpers from `training/checkpoint.py`, but do not weaken its existing legacy tuple return contract.

Each rank captures its own sampler cursor and Python/NumPy/CPU/CUDA RNG state.
In DDP these small objects are gathered to rank 0 with `gather_object`; rank 0
writes one RNG shard and one canonical sampler-state shard per rank.
On restore, every rank selects only its own entries after validating that keys
are exactly `0..world_size-1`. Saving is rejected unless
`optimizer_update_boundary=true`, there is no outstanding sampler batch,
gradient accumulation is zero, optimizer/scheduler/scaler have completed the
same update, and gradients have been zeroed. Mid-accumulation checkpoints are
not supported in schema v1.

- [ ] **Step 4: Prove uninterrupted and resumed training are identical**

Use a deterministic two-layer CPU model and a 12-example deterministic sampler:

1. run 8 optimizer updates uninterrupted;
2. separately run 5 updates, save, rebuild all objects, restore and run 3;
3. compare every model tensor, optimizer tensor, scheduler step, sample index, global step, and the ninth pre-update loss with `rtol=0, atol=0`.

Run this once with `gradient_accumulation_steps=2`. Assert a save attempt after
the first microbatch fails, while the post-update save is bitwise resumable.

Run:

```bash
pytest -q tests/training/test_stage_checkpoint.py \
  tests/training/test_deterministic_resume.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qwen3_omni_pretrain/training/rng_state.py \
  src/qwen3_omni_pretrain/training/stage_checkpoint.py \
  src/qwen3_omni_pretrain/training/checkpoint.py \
  tests/training/test_stage_checkpoint.py \
  tests/training/test_deterministic_resume.py
git commit -m "feat: checkpoint complete stage training state"
```

---

### Task 5: Implement the profile-aware stage runner and tiny overfit gate

**Files:**
- Create: `src/qwen3_omni_pretrain/training/stage_runner.py`
- Create: `src/qwen3_omni_pretrain/training/gate_protocol.py`
- Create: `src/qwen3_omni_pretrain/cli_train_profile.py`
- Create: `tests/training/test_stage_runner.py`
- Create: `tests/training/test_stage_overfit.py`
- Create: `tests/test_cli_train_profile.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing lifecycle tests**

```python
import pytest

from qwen3_omni_pretrain.training.stage_runner import (
    StageExitRejected,
    StageRunner,
)


def test_runner_applies_policy_runs_exact_steps_and_checks_exit_gate(
    tiny_program,
    fake_trainable_profile,
    deterministic_loader,
    recording_gate_runner,
):
    runner = StageRunner(
        program=tiny_program,
        components=fake_trainable_profile,
        gate_runner=recording_gate_runner,
    )

    result = runner.run_stage(
        stage_index=0,
        loader=deterministic_loader,
        resume=None,
    )

    assert result.progress.stage_step == tiny_program.stages[0].max_steps
    assert result.parameter_audit.trainable_groups == ("projector",)
    assert recording_gate_runner.calls == [("exit", "finite_loss")]


def test_runner_does_not_advance_when_exit_gate_fails(
    tiny_program,
    fake_trainable_profile,
    deterministic_loader,
    failing_gate_runner,
):
    runner = StageRunner(
        program=tiny_program,
        components=fake_trainable_profile,
        gate_runner=failing_gate_runner,
    )

    with pytest.raises(StageExitRejected, match="finite_loss"):
        runner.run_stage(0, deterministic_loader, resume=None)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/training/test_stage_runner.py \
  tests/training/test_stage_overfit.py \
  tests/test_cli_train_profile.py
```

Expected: FAIL because the runner and CLI do not exist.

- [ ] **Step 3: Implement explicit build and stage lifecycle interfaces**

```python
@dataclass(frozen=True)
class StageRunResult:
    progress: StageProgress
    parameter_audit: ParameterAudit
    mean_losses: Mapping[str, float]
    checkpoint_path: Path
    exit_report_path: Path
    next_stage_components: ProfileTrainingComponents


class StageGateRejected(RuntimeError):
    pass


class StageEntryRejected(StageGateRejected):
    pass


class StageExitRejected(StageGateRejected):
    pass


@runtime_checkable
class StageGateRunner(Protocol):
    def require(
        self,
        gate_names: Sequence[str],
        *,
        phase: str,
        checkpoint_path: Path | None,
    ) -> Path:
        """Return immutable evidence or raise the phase-specific rejection."""
        raise NotImplementedError


class StageRunner:
    def __init__(
        self,
        *,
        program: TrainingProgram,
        components: ProfileTrainingComponents,
        gate_runner: StageGateRunner,
    ) -> None:
        ...

    def run_stage(
        self,
        stage_index: int,
        loader: ResumableStageDataLoader,
        resume: Path | None,
    ) -> StageRunResult:
        # Validate entry gates, apply parameter policy, construct optimizer,
        # run exactly max_steps, checkpoint, evaluate exit gates, then return.
        raise NotImplementedError
```

The implementation must:

1. keep registry/factory resolution in the CLI application boundary:
   `build_profile_stage_application()` validates program/source fingerprints,
   resolves the lazy profile, narrows it to `TrainableProfileFactory`, calls
   `build_training()`, and validates `ProfileTrainingComponents`;
2. construct `StageRunner(program=..., components=..., gate_runner=...)` from
   those already validated objects; the runner never imports the registry and
   tests may inject the same protocol-conforming components directly;
3. verify entry gates before optimizer creation;
4. apply the parameter policy and write `parameter_audit.json`;
5. use the existing synchronized non-finite checks;
6. create optimizer/scheduler and the precision context only from the selected
   stage; construct a `GradScaler` for FP16 and prohibit it for BF16/FP32;
7. for each optimizer update, consume exactly
   `gradient_accumulation_steps` microbatches, scale loss accordingly, commit
   each sampler batch only after its backward succeeds, unscale if needed,
   synchronize non-finite status, clip the declared norm, step
   optimizer/scheduler/scaler, zero gradients, then advance `stage_step`;
8. checkpoint only at the resulting clean optimizer boundaries when
   `stage_step % checkpoint_interval_steps == 0`, and on completion;
9. run exit gates on the exact completion checkpoint;
10. return `next_stage_components` carrying the effective frozen manifest for
    the next stage; leave progress at the current stage when any gate fails.

The CLI application boundary resolves `program.data_sources` through
`StageDatasetRegistry` before model allocation, builds the profile/components,
calls `components.build_collator(stage)`, and constructs
`ResumableStageDataLoader`; tests may inject that same concrete loader but no
public path accepts an unrelated `DataLoader`. `num_workers=0` is revalidated
at construction. A factory-level integration test covers the outer
application builder while lifecycle tests instantiate `StageRunner` exactly
as shown above, so there is no hidden alternate constructor.

For resume/new-process execution, the application builder reads and validates
common checkpoint metadata plus any context-promotion receipt **before**
constructing the runner. It builds canonical profile components, then calls
`bind_training_components_to_checkpoint()`: all identity, source,
compatibility, assumption, tokenizer and architecture fields must match
exactly; the only permitted manifest difference is a higher
`validated_context_length` proven by the intact promotion chain. The function
returns a new frozen `ProfileTrainingComponents` with that checkpoint manifest.
Only this rebound object is passed to strict checkpoint loading. An
unreceipted context difference remains a hard mismatch.

Task 5 defines only this injected gate protocol plus a deterministic local
test implementation; it does not import the later evaluation package. Task 7
adds the production `ProfileEvaluationGateRunner` adapter, so every commit up
to that point remains importable and testable.

- [ ] **Step 4: Add a real tiny-set overfit test**

Train a tiny profile fixture on four examples for at most 80 CPU steps. Assert:

```python
assert final_loss < initial_loss * 0.35
assert result.progress.samples_consumed > 0
assert result.parameter_audit.trainable_parameters > 0
```

The test must exercise the real optimizer, scheduler, collator and stage checkpoint, not monkeypatch the training step.

- [ ] **Step 5: Implement a separate CLI without altering legacy dispatch**

CLI contract:

```text
qwen3-omni-train-profile \
  --program configs/train/profiles/qwen35_tiny_program.yaml \
  --stage multimodal_sft \
  --resume outputs/checkpoint-or-empty \
  --report-dir outputs/reports
```

An omitted `--resume` starts cleanly. An unknown stage, non-trainable reference runtime, manifest mismatch, or unsupported TP/PP/EP setting fails before dataset/model allocation. Add the entry point to `pyproject.toml`.

- [ ] **Step 6: Run the runner and CLI tests**

Run:

```bash
pytest -q tests/training/test_stage_runner.py \
  tests/training/test_stage_overfit.py \
  tests/test_cli_train_profile.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/training/stage_runner.py \
  src/qwen3_omni_pretrain/training/gate_protocol.py \
  src/qwen3_omni_pretrain/cli_train_profile.py \
  tests/training/test_stage_runner.py \
  tests/training/test_stage_overfit.py \
  tests/test_cli_train_profile.py \
  pyproject.toml
git commit -m "feat: run profile-aware training stages"
```

---

### Task 6: Add native DP Stage-2 execution with synchronized failures

**Files:**
- Create: `src/qwen3_omni_pretrain/training/distributed_stage.py`
- Modify: `src/qwen3_omni_pretrain/training/stage_runner.py`
- Modify: `src/qwen3_omni_pretrain/cli_train_profile.py`
- Create: `tests/distributed/test_stage_ddp.py`
- Create: `scripts/run_profile_training.sh`

- [ ] **Step 1: Write a failing two-process Gloo integration test**

```python
def _ddp_worker(rank, world_size, init_file, output_dir):
    context = DistributedStageContext.initialize(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
        local_rank=rank,
    )
    try:
        result = run_tiny_distributed_stage(context, output_dir)
        torch.save(result, Path(output_dir) / f"rank-{rank}.pt")
    finally:
        context.close()


def test_two_rank_stage_has_parameter_and_sample_parity(tmp_path):
    spawn_two_processes(_ddp_worker, tmp_path)
    rank_zero = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    rank_one = torch.load(tmp_path / "rank-1.pt", weights_only=True)

    assert rank_zero["parameter_checksum"] == rank_one["parameter_checksum"]
    assert set(rank_zero["global_non_padding_positions"]).isdisjoint(
        rank_one["global_non_padding_positions"]
    )
    assert sorted(
        rank_zero["global_non_padding_positions"]
        + rank_one["global_non_padding_positions"]
    ) == list(range(12))
```

Add cases where rank 1 injects (a) a failure before forward, (b) a failure
after forward/loss construction but before backward, (c) a non-finite loss,
and (d) a failure after backward but before optimizer step. Assert both ranks
raise the same phase-tagged error after the same safe collective boundary and
that no later phase was entered. Add a two-rank interrupted/resumed run and
compare every rank's next sample position/RNG draw plus final
model/optimizer/scheduler tensors bitwise with an uninterrupted run.

- [ ] **Step 2: Run the distributed test and verify failure**

Run:

```bash
pytest -q tests/distributed/test_stage_ddp.py
```

Expected: FAIL because `DistributedStageContext` does not exist.

- [ ] **Step 3: Implement the distributed context**

```python
@dataclass
class DistributedStageContext:
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device
    owns_process_group: bool

    @classmethod
    def from_environment(cls, backend: str | None = None):
        raise NotImplementedError

    @classmethod
    def initialize(
        cls,
        *,
        backend: str,
        rank: int,
        world_size: int,
        init_method: str,
        local_rank: int,
    ):
        """Explicit constructor used by tests and embedded launchers."""
        raise NotImplementedError

    def wrap_model(
        self,
        model: torch.nn.Module,
        *,
        find_unused_parameters: bool,
    ) -> torch.nn.Module:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
```

Rules:

- Gloo permits CPU tests; NCCL requires a valid local CUDA device.
- Initialization validates `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, configures an
  explicit process-group timeout, and refuses TP/PP/EP values other than 1.
- Model wrapping happens after parameter policy and before optimizer creation.
- Only rank 0 writes checkpoints/reports; all ranks barrier before reading the finished artifact.
- Mean losses, sample coverage, gate metrics and bad-state flags use explicit collectives.
- Every step has three declared readiness boundaries: all ranks first
  all-reduce `pre_forward`; after local forward and loss construction they
  all-reduce `pre_backward`; after local backward they all-reduce
  `pre_optimizer_step`. A rank that fails after forward but before backward
  therefore reports failure while peers are still at `pre_backward`, before
  any peer enters DDP backward collectives. Local exception summaries are
  gathered only after the corresponding status collective says a peer failed,
  and all ranks raise one phase-tagged error without stepping.
- DDP uses `broadcast_buffers=False` in schema v1. Profile forward/loss code
  may not introduce undeclared collectives between readiness boundaries.
  Any required collective is wrapped as an explicit distributed phase with
  timeout/abort semantics rather than executed inside arbitrary model code.
- Code must not attempt a status collective after an exception raised from a
  distributed collective itself. A monitored-barrier/timeout path records the
  rank and phase, calls process-group abort/destroy, and raises
  `DistributedCollectiveError`; identical Python exception text is guaranteed
  only for failures injected outside a collective.
- Rank 0 gathers every rank's sampler/RNG state before writing a checkpoint;
  all ranks barrier after the atomic rename, reload their own state, and verify
  the checkpoint digest before continuing.

The two-process test injects failures at all three safe boundaries, including
the critical "forward succeeded, local loss preparation failed, backward not
started" case. It asserts neither peer enters backward or optimizer step and
both report the `pre_backward` phase. A separate injected failure inside a
declared collective exercises timeout/abort cleanup rather than the ordinary
status path.

- [ ] **Step 4: Wire `torchrun` into the new CLI**

`scripts/run_profile_training.sh` must use `set -euo pipefail`, accept program/stage/nproc as positional arguments, and invoke:

```bash
torchrun --standalone --nproc_per_node "${PROFILE_NPROC}" \
  -m qwen3_omni_pretrain.cli_train_profile \
  --program "${PROFILE_PROGRAM}" \
  --stage "${PROFILE_STAGE}"
```

Do not change the legacy Stage-2 DDP rejection in `cli_train_thinker.py`.

- [ ] **Step 5: Run the distributed and legacy regression tests**

Run:

```bash
pytest -q tests/distributed/test_stage_ddp.py \
  tests/test_cli_train_thinker.py \
  tests/test_stage2_config.py \
  tests/test_stage2_runtime.py
```

Expected: PASS, including the legacy Stage-2 fail-fast assertions.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_omni_pretrain/training/distributed_stage.py \
  src/qwen3_omni_pretrain/training/stage_runner.py \
  src/qwen3_omni_pretrain/cli_train_profile.py \
  tests/distributed/test_stage_ddp.py \
  scripts/run_profile_training.sh
git commit -m "feat: support distributed profile stage training"
```

---

### Task 7: Gate long-context progression with unified profile evaluation

**Files:**
- Modify: `src/qwen3_omni_pretrain/evaluation/contracts.py`
- Create: `src/qwen3_omni_pretrain/evaluation/registry.py`
- Create: `src/qwen3_omni_pretrain/evaluation/stage_gates.py`
- Create: `src/qwen3_omni_pretrain/evaluation/profile_suite.py`
- Create: `src/qwen3_omni_pretrain/evaluation/long_context.py`
- Create: `tests/evaluation/test_registry.py`
- Create: `tests/evaluation/test_stage_gates.py`
- Create: `tests/evaluation/test_profile_suite.py`
- Create: `tests/evaluation/test_long_context_progression.py`
- Create: `configs/evaluation/profile_gates.yaml`
- Create: `scripts/evaluate_profile.py`
- Modify: `src/qwen3_omni_pretrain/evaluation/__init__.py`
- Modify: `src/qwen3_omni_pretrain/training/stage_runner.py`

- [ ] **Step 1: Write failing registry and gate tests**

```python
from qwen3_omni_pretrain.evaluation.contracts import (
    Comparison,
    MetricResult,
    StageGate,
)
from qwen3_omni_pretrain.evaluation.stage_gates import evaluate_gate


def test_gate_requires_every_metric_and_preserves_evidence():
    gate = StageGate(
        name="multimodal_regression",
        requirements={
            "text.loss": (Comparison.LE, 2.0),
            "image.shuffle_delta": (Comparison.GE, 0.05),
            "audio.shuffle_delta": (Comparison.GE, 0.05),
            "video.shuffle_delta": (Comparison.GE, 0.05),
            "cache.max_logit_error": (Comparison.LE, 1.0e-5),
        },
    )
    metrics = (
        MetricResult("text.loss", 1.5, sample_count=8),
        MetricResult("image.shuffle_delta", 0.2, sample_count=4),
        MetricResult("audio.shuffle_delta", 0.2, sample_count=4),
        MetricResult("cache.max_logit_error", 0.0, sample_count=8),
    )

    result = evaluate_gate(
        gate,
        metrics,
        evidence=sample_evidence_binding(),
    )

    assert result.passed is False
    assert result.missing_metrics == ("video.shuffle_delta",)
```

Registry tests must reject duplicate metric names and evaluators whose declared profile/capability does not match the requested suite.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/evaluation/test_registry.py \
  tests/evaluation/test_stage_gates.py \
  tests/evaluation/test_profile_suite.py \
  tests/evaluation/test_long_context_progression.py
```

Expected: FAIL because the evaluation modules do not exist.

- [ ] **Step 3: Implement typed metrics, gates and reports**

```python
class Comparison(str, Enum):
    LE = "le"
    GE = "ge"


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    sample_count: int
    unit: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StageGate:
    name: str
    requirements: Mapping[str, tuple[Comparison, float]]


@dataclass(frozen=True)
class EvidenceBinding:
    checkpoint_content_digest: str
    manifest_digest: str
    program_hash: str
    data_fingerprint: str
    requested_context_length: int | None


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: tuple[MetricResult, ...]
    failed_metrics: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    evidence: EvidenceBinding


@dataclass(frozen=True)
class EvaluationReport:
    profile_manifest: ProfileManifest
    architecture_summary: ArchitectureSummary
    evidence: EvidenceBinding
    suite_name: str
    metrics: tuple[MetricResult, ...]
    gates: tuple[GateResult, ...]
    environment: Mapping[str, object]
```

Reports use finite JSON numbers only and include source revision, precision,
device, hardware name, context length, dataset fingerprint and command line.
Every gate in one report carries exactly the report's
`checkpoint_content_digest`, manifest digest, program hash and data
fingerprint; a context gate additionally carries its exact requested length.
Constructors reject mismatched nested bindings. A missing metric fails the
gate; NaN/Inf fails serialization and the gate.

Extend the existing MiMo experiment-report types in
`evaluation/contracts.py`; do not rename or duplicate
`RuntimeEnvironment`, `CorrectnessGate`, `BenchmarkMeasurement`, or
`ExperimentReport`.

`ProfileEvaluationGateRunner` in `stage_gates.py` implements the
`StageGateRunner` protocol from Task 5. It resolves gate names from the strict
YAML registry, evaluates or loads evidence for the exact checkpoint digest,
writes one immutable JSON report, and returns that report path. Evidence for a
different checkpoint/program/data fingerprint is stale and must be rejected.

- [ ] **Step 4: Implement the mandatory profile suite**

The suite must invoke applicable evaluators for:

- text loss/perplexity and a fixed greedy-generation fixture;
- image/audio/video loss plus media ablation and deterministic shuffle delta;
- cached/uncached max logit error and exact greedy token parity;
- media encoder call count during prefill/decode;
- router entropy/load and active parameter ratio for MoE profiles;
- teacher-forced MTP losses and rejection repair for MTP profiles;
- synchronized non-finite fixture.

Capability absence is recorded as `not_applicable` metadata only when the
validated `ArchitectureSummary.capabilities` says the component does not
exist. `ProfileManifest` provides identity/provenance, not a second capability
map. Absence cannot convert a capability declared true into a skip, and a
summary/manifest profile mismatch rejects the report.

- [ ] **Step 5: Implement progressive context validation**

```python
@dataclass(frozen=True)
class ContextValidation:
    requested_length: int
    previous_validated_length: int
    gate_name: str
    checkpoint_content_digest: str
    manifest_digest_before: str
    program_hash: str
    data_fingerprint: str
    experimental_override: bool


def validate_context_transition(
    transition: ContextValidation,
    gate_result: GateResult,
) -> int:
    if transition.experimental_override:
        raise ValueError("experimental override cannot promote a manifest")
    allowed_edges = {
        0: {4096},
        4096: {16384},
        16384: {32768},
        32768: {65536},
        65536: {131072},
        131072: {262144},
    }
    if transition.requested_length not in allowed_edges.get(
        transition.previous_validated_length, set()
    ):
        raise ValueError("context transition is not an allowed next edge")
    if not gate_result.passed:
        raise ValueError(f"context gate failed: {gate_result.name}")
    expected = EvidenceBinding(
        checkpoint_content_digest=transition.checkpoint_content_digest,
        manifest_digest=transition.manifest_digest_before,
        program_hash=transition.program_hash,
        data_fingerprint=transition.data_fingerprint,
        requested_context_length=transition.requested_length,
    )
    if gate_result.name != transition.gate_name:
        raise ValueError("context gate name does not match transition")
    if gate_result.evidence != expected:
        raise ValueError("context gate evidence does not match transition")
    return transition.requested_length
```

Default config contains 4096, 16384 and 32768 gates. Add opt-in definitions
for 65536, 131072 and 262144; 262000/262144 naming must be explicit rather than
silently rounded. Each gate definition declares the exact predecessor,
required metrics/datasets and hardware/memory evidence. Skipping an edge or
using `experimental_override` may run an experiment but may not change the
manifest.

`GateResult` evidence must match all four transition identities and the exact
requested length. After `validate_context_transition()` succeeds,
`StageRunner` creates a new frozen manifest with
`validated_context_length=requested_length` and performs a whole-checkpoint
metadata transaction:

1. copy/link the immutable payload into a sibling staging directory and verify
   its unchanged `checkpoint_content_digest`;
2. write the promoted common `architecture.json`;
3. rewrite `stage_checkpoint.json` with the new
   `architecture_metadata_sha256` while retaining the same content digest;
4. write a promotion receipt containing the predecessor/new manifest digests,
   evaluation report digest, program/data identities and requested length;
5. validate all cross-references, fsync, then atomically replace the checkpoint
   directory.

The original evaluation report remains immutable and bound to the predecessor
manifest; the promotion receipt is the auditable edge to the new manifest.
Failure before rename leaves the old checkpoint intact. Merely setting
RoPE/max-position configuration or passing another checkpoint's report cannot
update this field.

After the rename, the runner creates
`promoted_components = dataclasses.replace(self.components,
manifest=promoted_manifest)` and returns it as
`StageRunResult.next_stage_components`; it never mutates the frozen old object.
The same-process stage orchestrator must build the next runner from that
returned object. A new process uses the receipt-aware
`bind_training_components_to_checkpoint()` path defined in Task 5. Tests cover
both paths through 0→4K→16K and prove strict checkpoint loading sees the
promoted manifest rather than the factory's original zero-context declaration.

Tests independently mutate each evidence field and gate name and require
rejection. Promotion fault-injection tests fail after each staged write and
prove readers see either the complete predecessor or complete promoted
metadata set, never a mixed `architecture.json`/stage sidecar. A successful
promotion proves payload digests are unchanged and all three new metadata
cross-references validate.

- [ ] **Step 6: Add the evaluation CLI and run tests**

CLI:

```bash
python scripts/evaluate_profile.py \
  --profile qwen35_omni_inspired \
  --checkpoint outputs/profile/checkpoint \
  --suite multimodal_sft \
  --gates configs/evaluation/profile_gates.yaml \
  --output outputs/reports/multimodal_sft.json
```

Run:

```bash
pytest -q tests/evaluation
python scripts/evaluate_profile.py --help
```

Expected: all tests PASS and help exits 0 without importing optional model backends.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/evaluation \
  src/qwen3_omni_pretrain/training/stage_runner.py \
  tests/evaluation \
  configs/evaluation/profile_gates.yaml \
  scripts/evaluate_profile.py
git commit -m "feat: gate profile stages with unified evaluation"
```

---

### Task 8: Add request-safe serving with content-addressed media caching

**Files:**
- Create: `src/qwen3_omni_pretrain/serving/__init__.py`
- Create: `src/qwen3_omni_pretrain/serving/contracts.py`
- Create: `src/qwen3_omni_pretrain/serving/media_cache.py`
- Create: `src/qwen3_omni_pretrain/serving/scheduler.py`
- Create: `src/qwen3_omni_pretrain/serving/backends.py`
- Create: `src/qwen3_omni_pretrain/serving/engine.py`
- Create: `src/qwen3_omni_pretrain/serving/metrics.py`
- Create: `tests/serving/test_media_cache.py`
- Create: `tests/serving/test_scheduler.py`
- Create: `tests/serving/test_engine.py`
- Create: `scripts/benchmark_profile_serving.py`

- [ ] **Step 1: Write failing media-cache and engine tests**

```python
from dataclasses import replace

from qwen3_omni_pretrain.serving.media_cache import (
    MediaCacheKey,
    ProcessedMediaCache,
)


def test_cache_key_changes_for_every_semantic_input():
    base = MediaCacheKey.from_bytes(
        payload=b"image",
        modality="image",
        processor_revision="processor-a",
        encoder_revision="encoder-a",
        projector_revision="projector-a",
        dtype="float32",
        preprocessing={"resize": [224, 224]},
    )

    assert base != replace(base, dtype="bfloat16")
    assert base != replace(base, encoder_revision="encoder-b")
    assert base != replace(base, preprocessing_digest="different")


def test_decode_never_reencodes_media(fake_common_backend, image_request):
    engine = ServingEngine(backend=fake_common_backend, max_concurrency=1)

    events = list(engine.generate_sync(image_request))

    assert fake_common_backend.media_encode_calls == 1
    assert [event.kind for event in events][-1] == "completed"
    assert all(
        event.request_id == image_request.request_id for event in events
    )
```

Scheduler tests must cover bounded admission, FIFO within equal priority, cancellation during queue/prefill/decode, and a slow consumer causing backpressure without unbounded event growth.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/serving
```

Expected: FAIL because the serving package does not exist.

- [ ] **Step 3: Implement immutable serving contracts and byte-bounded LRU**

```python
@dataclass(frozen=True)
class InferenceMedia:
    source_id: str
    modality: MediaModality
    payload: bytes | None
    absolute_path: Path | None
    media_type: str
    timeline_offset_seconds: float
    preprocessing: Mapping[str, object]


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    profile: ArchitectureProfile
    input_ids: torch.Tensor
    media: tuple[InferenceMedia, ...]
    max_new_tokens: int
    temperature: float
    seed: int


@dataclass(frozen=True)
class StreamEvent:
    request_id: str
    kind: str
    sequence_number: int
    token_id: int | None
    audio: torch.Tensor | None
    timing_ns: int
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class MediaCacheKey:
    content_digest: str
    modality: str
    processor_revision: str
    encoder_revision: str
    projector_revision: str
    dtype: str
    preprocessing_digest: str
```

`ProcessedMediaCache` owns detached CPU tensors, has a fixed byte budget, updates recency under one lock, never returns mutable internal storage, and reports hit/miss/eviction/bytes. `enabled=False` bypasses both lookup and insertion; the stage runner always constructs it disabled.

`InferenceMedia` requires exactly one of immutable `payload` or
`absolute_path`. Payload size, item count, decoded pixels/frames/audio duration
and aggregate request bytes have configured hard limits. Paths must be
absolute, resolve under an allowlisted media root, refer to a regular file and
must not escape through symlinks. `media_type`, modality and decoder output
must agree. The strict media loader converts this request into
`MediaRequest`/`MediaSource`; provenance output types are never used as if they
contained raw bytes. A path-backed item is opened/read once under the byte
limit; the exact resulting bytes are both hashed and decoded, preventing a
hash/decode time-of-check/time-of-use mismatch.

- [ ] **Step 4: Implement bounded scheduler and generation engine**

Define explicit backend and handle boundaries:

```python
@dataclass(frozen=True)
class PreparedPrefill:
    inputs: ModelPrefillInputs
    media_cache_keys: tuple[MediaCacheKey, ...]
    diagnostics: Mapping[str, object]


@runtime_checkable
class ProfilePrefillAdapter(Protocol):
    def prepare(
        self,
        request: InferenceRequest,
        cache: ProcessedMediaCache,
    ) -> PreparedPrefill:
        ...


@dataclass(frozen=True)
class BackendPrefillResult:
    state: object | None
    events: tuple[StreamEvent, ...]
    terminal: bool


@dataclass(frozen=True)
class BackendDecodeResult:
    state: object | None
    events: tuple[StreamEvent, ...]
    terminal: bool


class ServingBackend(Protocol):
    capabilities: Mapping[str, bool]

    async def prefill(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> BackendPrefillResult:
        ...

    async def decode(
        self, state: object, cancellation: CancellationToken
    ) -> BackendDecodeResult:
        ...

    async def close_request(self, request_id: str) -> None:
        ...


class RequestHandle:
    request_id: str

    def cancel(self) -> None:
        ...

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        ...
```

`ServingEngine.submit(request) -> RequestHandle` is the public asynchronous
API; `generate_sync()` is a test/CLI adapter over it. The handle owns the
request-local cancellation token and bounded event queue. Cancellation before
admission, during prefill, during decode and after a terminal event is
idempotent; exactly one terminal `cancelled` or `completed` event is emitted.
Result constructors require at least one event, consistent request IDs and
strictly increasing sequence numbers. A non-terminal result requires a
non-null state owned by that request; a terminal result prohibits a reusable
state and must end in exactly one terminal event.

`CommonStateBackend` adapts `CacheCapableModel.prefill/decode` and supports
token/cache metrics. It owns the selected factory artifact, a matching
`ProfilePrefillAdapter` and the serving-only cache. The adapter is the single
raw-media boundary: it enforces limits, reads the stable bytes, decodes them
to typed `DecodedMedia`, runs the profile's
`MultimodalPrefillPipeline`, and returns `ModelPrefillInputs`. Cache values are
detached immutable processed-media features with provenance, never module
objects or request state; cache hits still pass through sample-level sequence
assembly so placeholder ownership/order is revalidated. A text-only MiMo
profile rejects non-empty media before decode; Qwen3.5 selects its public
audio/video adapters explicitly.

`Qwen3OfflineBackend` adapts the official Transformers
facade as one offline operation and emits final text/audio plus completion; it
does not claim common KV state, token streaming, media-embedding cache parity
or TTFT/TTFC. `VllmRealtimeBackend` adapts the separately configured
WebSocket client and may report realtime deltas. Backend capabilities select
valid metrics/tests explicitly; no profile-name conditional casts the Qwen3
facade to `CacheCapableModel`.

The engine flow is:

```text
admit -> cache lookup/strict media load -> prefill once
      -> copy-on-write DecoderState -> one-token decode loop
      -> optional codec chunks -> completed/cancelled/error
```

Rules:

- duplicate active `request_id` fails;
- raw media is accepted only during prefill;
- each decode call receives only the newest token and immutable state;
- every non-terminal `BackendPrefillResult`/`BackendDecodeResult` contains a
  non-null next state; the engine atomically replaces its request-local state
  before the next token. A terminal result contains no reusable state.
- cancellation drops uncommitted state and releases all request references;
- event queues have a fixed maximum and `await put()` provides backpressure;
- one request's failure cannot cancel or mutate another request;
- errors are structured terminal events without source file contents.

- [ ] **Step 5: Add correctness and concurrency tests**

Tests must prove:

- cached and uncached media embeddings are identical;
- two sessions never share `DecoderState`;
- cache revision/dtype/preprocessing changes force a miss;
- LRU eviction respects the byte budget;
- cancellation during prefill and decode releases scheduler capacity;
- four concurrent requests complete with monotonic per-request sequence numbers;
- slow consumption never exceeds the configured event queue size.
- `RequestHandle.cancel()` at each lifecycle phase emits exactly one terminal
  event and never affects a second handle;
- common, Qwen3-offline and external-realtime backend capability matrices
  reject unsupported metrics rather than filling them with zeros.
- a two-token decode test proves the second call receives the first call's
  returned immutable state, while the original object is unchanged;
- raw bytes traverse strict load → `DecodedMedia` → profile prefill adapter →
  `ModelPrefillInputs` exactly once on a miss, while a hit reuses only detached
  processed features and still re-runs placeholder assembly.

Run:

```bash
pytest -q tests/serving
```

Expected: PASS.

- [ ] **Step 6: Add a correctness-gated benchmark**

`benchmark_profile_serving.py` runs cache parity before timing and writes
capability-aware JSON. Every metric is either measured with strictly positive
timing/throughput samples, or explicitly not applicable with a reason:

```json
{
  "profile": "qwen35_omni_inspired",
  "precision": "bfloat16",
  "context_length": 4096,
  "output_length": 128,
  "concurrency": 4,
  "cache_status": "hit",
  "ttft_ms": {"status": "measured", "p50": 12.4, "p90": 15.8},
  "ttfc_ms": {
    "status": "not_applicable",
    "reason": "backend has no streaming audio capability"
  },
  "decode_tokens_per_second": {"status": "measured", "value": 41.2},
  "rtf": {
    "status": "not_applicable",
    "reason": "request has no audio output"
  },
  "peak_device_memory_bytes": 0,
  "cancelled_requests": 0,
  "backpressure_events": 0
}
```

Zero counters/memory are valid observations; zero measured latency,
throughput, TTFC or RTF is not. Define one strict metric union
(`MeasuredScalar`/`MeasuredPercentiles`/`NotApplicable`) and validate it against
backend capabilities and request modalities before JSON serialization. A
text-only MiMo row therefore marks TTFC/RTF not applicable rather than
inventing values. The script must run separate hit/miss cases at concurrency
1, 4 and 8 and record hardware, source revision and command line. Its
common-state mode accepts only
Qwen3.5-inspired or MiMo experimental checkpoints implementing
`CommonStateBackend`; Qwen3 official service measurements require
`--backend vllm-realtime` and a configured service. Offline Qwen3 is limited to
whole-request latency/RTF and is never mixed into token-cache benchmark rows.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/serving \
  tests/serving \
  scripts/benchmark_profile_serving.py
git commit -m "feat: serve profiles with isolated media caching"
```

---

### Task 9: Add gated distillation, preference, and local reward objectives

**Files:**
- Create: `src/qwen3_omni_pretrain/loss/distillation.py`
- Create: `src/qwen3_omni_pretrain/loss/preference.py`
- Create: `src/qwen3_omni_pretrain/training/advanced_batches.py`
- Create: `src/qwen3_omni_pretrain/training/reference_models.py`
- Create: `src/qwen3_omni_pretrain/training/reward_protocol.py`
- Modify: `src/qwen3_omni_pretrain/profiles/training.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py`
- Modify: `src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/factory.py`
- Modify: `src/qwen3_omni_pretrain/training/stages.py`
- Modify: `src/qwen3_omni_pretrain/training/stage_config.py`
- Create: `tests/loss/test_distillation.py`
- Create: `tests/loss/test_preference.py`
- Create: `tests/training/test_reward_protocol.py`
- Create: `tests/training/test_advanced_stage_runner.py`
- Modify: `src/qwen3_omni_pretrain/loss/__init__.py`
- Modify: `src/qwen3_omni_pretrain/training/stage_runner.py`

- [ ] **Step 1: Write failing numerical objective tests**

```python
import torch

from qwen3_omni_pretrain.loss.distillation import masked_distillation_kl
from qwen3_omni_pretrain.loss.preference import dpo_loss


def test_distillation_is_zero_for_identical_logits_and_ignores_masked_rows():
    logits = torch.tensor([[[1.0, 2.0], [9.0, -9.0]]])
    mask = torch.tensor([[True, False]])

    loss = masked_distillation_kl(
        student_logits=logits,
        teacher_logits=logits.clone(),
        loss_mask=mask,
        temperature=2.0,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1.0e-7, rtol=0)


def test_dpo_prefers_a_larger_policy_preference_margin():
    lower = dpo_loss(
        policy_chosen_logps=torch.tensor([0.2]),
        policy_rejected_logps=torch.tensor([0.1]),
        reference_chosen_logps=torch.tensor([0.0]),
        reference_rejected_logps=torch.tensor([0.0]),
        beta=0.1,
    )
    higher = dpo_loss(
        policy_chosen_logps=torch.tensor([1.0]),
        policy_rejected_logps=torch.tensor([-1.0]),
        reference_chosen_logps=torch.tensor([0.0]),
        reference_rejected_logps=torch.tensor([0.0]),
        beta=0.1,
    )

    assert higher < lower
```

Reward tests must verify per-group zero-mean advantages, masking, clipping, and refusal to enter reward optimization without both an SFT gate artifact and a reward-evaluator gate artifact.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/loss/test_distillation.py \
  tests/loss/test_preference.py \
  tests/training/test_reward_protocol.py
```

Expected: FAIL because the new objective modules do not exist.

- [ ] **Step 3: Implement stable masked objectives**

```python
def masked_distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_p = F.softmax(
        teacher_logits.detach().float() / temperature, dim=-1
    )
    per_token = F.kl_div(
        student_logp, teacher_p, reduction="none"
    ).sum(dim=-1)
    return per_token[loss_mask].mean() * (temperature ** 2)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = (
        reference_chosen_logps.detach() - reference_rejected_logps.detach()
    )
    logits = beta * (policy_margin - reference_margin)
    positive = -F.logsigmoid(logits)
    negative = -F.logsigmoid(-logits)
    return (
        (1.0 - label_smoothing) * positive
        + label_smoothing * negative
    ).mean()
```

Both functions must validate shapes, finite inputs, non-empty masks and scalar ranges. Teacher/reference tensors are detached inside the function. Tests include FP32 gradcheck-scale finite gradients and padding invariance.

- [ ] **Step 4: Define complete advanced-stage batch and model inputs**

```python
@dataclass(frozen=True)
class DistillationBatch:
    student_inputs: ModelPrefillInputs
    teacher_inputs: ModelPrefillInputs
    loss_mask: torch.BoolTensor
    sample_ids: tuple[str, ...]
    tokenizer_sha256: str


@dataclass(frozen=True)
class PreferenceBatch:
    chosen_inputs: ModelPrefillInputs
    rejected_inputs: ModelPrefillInputs
    reference_chosen_inputs: ModelPrefillInputs
    reference_rejected_inputs: ModelPrefillInputs
    chosen_loss_mask: torch.BoolTensor
    rejected_loss_mask: torch.BoolTensor
    pair_ids: tuple[str, ...]
    tokenizer_sha256: str


@dataclass(frozen=True)
class RewardBatch:
    samples: tuple["RewardSample", ...]
    policy_inputs: ModelPrefillInputs
    group_ids: torch.LongTensor
    old_logps: torch.Tensor
    loss_mask: torch.BoolTensor
```

Add strict collator adapters selected by stage kind; an SFT batch cannot be
silently reused for these objectives. Distillation objective config requires
local teacher checkpoint, manifest digest, temperature and loss weight.
Preference config requires local reference checkpoint, manifest digest, beta,
label smoothing and data fingerprint. Reward config requires provider revision,
clip epsilon and evaluator evidence. Unknown or missing keys fail in
`load_training_program()`.

The raw canonical sample is collated independently through the student and
teacher/reference profile pipelines—student `inputs_embeds` or processed media
are never handed to another model. Schema v1 permits token-level KL/DPO only
when the pinned artifacts have the exact same tokenizer SHA, vocabulary size,
special-token mapping, emitted `input_ids` and supervised token alignment.
`DistillationBatch`/`PreferenceBatch` store that SHA and constructors assert
the paired masks/token axes match. Cross-tokenizer distillation requires a
future explicit alignment/mapping schema and fails before model allocation in
this release.

`ProfileTrainingComponents.build_collator(stage)` dispatches to a concrete SFT,
distillation, preference or reward collator and returns the corresponding
typed batch above. `StageRunner` pattern-matches the stage kind and exact batch
class; a generic mapping or a batch from another objective is rejected before
forward. The collator fingerprint includes both pinned artifact/tokenizer
digests for dual-model stages.

Task 9 updates the real Qwen3.5 and MiMo `build_training()` implementations,
not only fake test components. Each factory composes its existing
profile-specific SFT/media collator with the common advanced batch adapters;
unsupported objectives fail during `build_collator(stage)`, before optimizer
creation. A parametrized factory integration test builds both tiny profiles,
requests every supported advanced stage, feeds canonical raw samples, and
asserts the exact typed batch plus context/tokenizer fingerprints. The
text-only MiMo path rejects media samples, while Qwen3.5 independently collates
teacher/reference media inputs under the strict tokenizer-alignment rule.

`reference_models.py` loads teacher/reference artifacts locally through their
profile factories, checks exact manifest/checkpoint digests, calls `.eval()`,
freezes parameters and returns detached logits/log-probabilities under
`torch.no_grad()`. It never downloads or defaults to the current trainable
model.

- [ ] **Step 5: Define the local reward boundary and objective**

```python
@runtime_checkable
class RewardProvider(Protocol):
    @property
    def revision(self) -> str:
        raise NotImplementedError

    def score(self, samples: Sequence["RewardSample"]) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class RewardSample:
    prompt_id: str
    candidate_id: str
    token_ids: tuple[int, ...]
    metadata: Mapping[str, object]


def leave_one_out_group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    advantages = torch.empty_like(rewards)
    for group_id in torch.unique(group_ids):
        selected = group_ids == group_id
        count = int(selected.sum())
        if count < 2:
            raise ValueError("each reward group requires at least two samples")
        group_rewards = rewards[selected]
        baseline = (group_rewards.sum() - group_rewards) / (count - 1)
        advantages[selected] = group_rewards - baseline
    return advantages


def clipped_policy_objective(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    if log_ratio.shape != loss_mask.shape or loss_mask.dtype is not torch.bool:
        raise ValueError("log_ratio and bool loss_mask must have the same shape")
    if advantages.ndim != 1 or advantages.shape[0] != log_ratio.shape[0]:
        raise ValueError("advantages must have shape [batch]")
    token_advantages = advantages.unsqueeze(-1).expand_as(log_ratio)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    per_token = -torch.minimum(
        ratio * token_advantages,
        clipped * token_advantages,
    )
    return per_token[loss_mask].mean()
```

The production function additionally rejects non-finite inputs, an empty mask,
rank other than `[B,T]`, and `clip_epsilon` outside `(0,1)`. Tests cover
`B != T` so accidental `[B]`→trailing-axis broadcasting cannot pass.

This boundary accepts only local/injected providers with a non-empty immutable
revision; it does not call a remote service. Persist reward revision, sample
IDs and aggregate score statistics, but never raw credentials or private
prompt contents.

- [ ] **Step 6: Dispatch complete advanced-stage training steps**

`StageRunner` must require:

- distillation: a passing multimodal-SFT report and pinned teacher manifest;
- preference: a passing multimodal-SFT report, pinned reference manifest and preference-data fingerprint;
- reward optimization: both a passing multimodal-SFT report and a passing reward-evaluator calibration report.

The runner rejects missing/stale reports before loading teacher/reference/reward models.

After those checks, `StageRunner` dispatches by `TrainingStageKind`:

- distillation: student forward + pinned teacher forward +
  `masked_distillation_kl`;
- preference: chosen/rejected policy log-probabilities + frozen reference
  log-probabilities + `dpo_loss`;
- reward optimization: validate grouped candidates, obtain local provider
  scores, compute leave-one-out advantages and the clipped policy objective.

All paths use the same accumulation, precision, synchronized non-finite,
checkpoint and resume lifecycle as SFT and emit named component losses. Add
one tiny real optimizer-step/resume test per stage. If its required batch
adapter or model/provider is absent, the stage is rejected before optimizer
creation; there is no executable loss-only skeleton.

- [ ] **Step 7: Run focused and full training tests**

Run:

```bash
pytest -q tests/loss/test_distillation.py \
  tests/loss/test_preference.py \
  tests/training/test_reward_protocol.py \
  tests/training/test_advanced_stage_runner.py \
  tests/training/test_stage_runner.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qwen3_omni_pretrain/loss/distillation.py \
  src/qwen3_omni_pretrain/loss/preference.py \
  src/qwen3_omni_pretrain/loss/__init__.py \
  src/qwen3_omni_pretrain/training/advanced_batches.py \
  src/qwen3_omni_pretrain/training/reference_models.py \
  src/qwen3_omni_pretrain/training/reward_protocol.py \
  src/qwen3_omni_pretrain/profiles/training.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py \
  src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/factory.py \
  src/qwen3_omni_pretrain/training/stages.py \
  src/qwen3_omni_pretrain/training/stage_config.py \
  src/qwen3_omni_pretrain/training/stage_runner.py \
  tests/loss/test_distillation.py \
  tests/loss/test_preference.py \
  tests/training/test_reward_protocol.py \
  tests/training/test_advanced_stage_runner.py
git commit -m "feat: gate advanced post-sft objectives"
```

---

## Final verification

- [ ] **Step 1: Run all ordinary CPU tests**

```bash
pytest -q
```

Expected: the original 111 tests plus all new ordinary tests pass.

- [ ] **Step 2: Run the two-process distributed suite**

```bash
pytest -q tests/distributed
```

Expected: PASS with no leaked process group or child process.

- [ ] **Step 3: Run static profile and CLI checks**

```bash
python scripts/inspect_architecture.py \
  configs/model/qwen3_omni_7b_moe.yaml
python -m qwen3_omni_pretrain.cli_train_profile --help
python scripts/evaluate_profile.py --help
python scripts/benchmark_profile_serving.py --help
```

Expected: every command exits 0 without downloading weights.

- [ ] **Step 4: Run opt-in hardware checks in the matching dependency environment**

```bash
pytest -q -m large_model --run-large-model-tests
python scripts/benchmark_profile_serving.py \
  --backend common-state \
  --profile mimo_v25_experimental \
  --checkpoint /absolute/path/to/local/mimo-experiment-checkpoint \
  --context-length 4096 \
  --output-length 128 \
  --concurrency 1 4 8 \
  --output outputs/reports/mimo-serving.json

python scripts/benchmark_profile_serving.py \
  --backend vllm-realtime \
  --profile qwen3_omni_reference \
  --service-url "${QWEN3_OMNI_REALTIME_URL}" \
  --concurrency 1 4 8 \
  --output outputs/reports/qwen3-realtime-serving.json
```

Expected: common-state correctness gates pass before timing and its report
contains hit/miss rows for all concurrency levels. The Qwen3 command runs only
when the external service is configured and reports realtime-service metrics,
not common-cache parity.

- [ ] **Step 5: Update capability documentation only from passed artifacts**

Modify `README.md` and the three research reports only when a checked-in or archived evaluation report supports the new claim. Keep Qwen3.5 at `paper-inspired` and MiMo at `MiMo-style-experiment`; do not promote either to checkpoint-compatible.

- [ ] **Step 6: Commit only evidence-backed documentation changes**

```bash
if ! git diff --quiet -- README.md docs/research; then
  git add README.md docs/research
  git commit -m "docs: record stage training and evaluation gates"
fi
```

If no capability claim changed, no documentation commit is required; do not
create an empty commit.
