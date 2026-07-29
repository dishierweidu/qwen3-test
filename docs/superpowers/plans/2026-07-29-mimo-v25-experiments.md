# MiMo-V2.5-Inspired Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立 `mimo_v25_experimental` profile 中实现可归因的小规模 SWA/GA、routed-only SwiGLU MoE、expert parallel 和 speculative MTP 实验。

**Architecture:** 新建通用 `hybrid_swa_moe` 模型树，不修改 legacy Qwen 模型类。显式 per-layer attention/FFN lists 取代相互覆盖的 boolean/index 配置。SWA/GA 复用公共 immutable cache；MoE 层与 dense FFN 互斥；MTP correctness 与 speculative verification 先于性能优化。

**Tech Stack:** Python 3.10, PyTorch 2.10.0 SDPA/eager attention, torch.distributed Gloo/NCCL, dataclasses, PyYAML, pytest, JSON benchmark reports.

## Global Constraints

- 本计划依赖 profile/contracts、media 和 cache 三个公共计划完成；它不依赖
  Qwen3 reference runtime 或 Qwen3.5 profile 的实现。
- architecture profile 固定为 `mimo_v25_experimental`，compatibility 固定为 `MiMo-style-experiment`。
- 模型 class/model type 固定使用通用实验命名；不得使用官方 `mimo_v2` 或 MiMo checkpoint class 名。
- 本计划不加载 MiMo checkpoint，不声明 state-dict 或 numerical checkpoint compatibility。
- Task 2 的第一版 tiny config 使用 6 层、5 个 SWA + 1 个 full
  attention、window 128、8 experts/top-2，并显式设置
  `mtp_num_predictors: 0`；只有 Task 6 在 MTP 模块可用后才把它改为 1。
- SWA 和 full attention 独立 RoPE base；第一版不混入 DeltaNet/GDN。
- `value_scale` 只能为 `None` 或有限正数；`0`、负数、NaN 和 Inf
  必须在模型分配前失败。
- SWA persistent KV token 数始终不超过 window size；sink 是额外 softmax logit，不占 KV token。
- 当 sink 关闭且 `T <= W` 时，同权重 SWA 与 full attention FP32 误差必须 `<=1e-5`。
- 第 0 层 dense SwiGLU；后续配置为 routed-MoE 的层不得持有或执行 shared/dense FFN。
- `top_k >= num_experts` 在构建前失败；不做 clamp。
- 第一版 router 使用可训练 softmax、top-k renormalization 和 aux loss；
  总训练损失固定为
  `ce_loss + router_aux_loss_weight * aux_loss + mtp_loss_weight * mtp_loss`；
  `noaux_tc` 暂不实现。
- `EP=1` 时保留项目既有 DP/PP/TP 拓扑语义，不额外收窄组合；只有
  `EP>1` 时第一版严格要求 `DP=1,PP=1,EP=2,TP=1`，且必须在创建
  process group 或分配模型参数前 fail-fast。
- speculative sampling 使用标准接受/拒绝校正；拒绝后必须从 immutable committed state 恢复。
- 只有 correctness gates 通过后才运行或报告 performance benchmark。
- `mimo_v25_experimental` 本身暂不实现 DeltaNet；D6 报告必须把现有
  legacy DeltaNet 作为独立、不同随机权重的参考行，显式标记
  `weight_comparable=false`，不得计算或宣称同权重收益。
- 暂不实现 256/384 experts、1M context、FP4/DFlash、GCache/RDMA 或 Talker。
- Snippet 中的 `...` 只表示 Protocol/签名节选；任务提交不得保留未实现
  stub，必须由该任务列出的行为测试证明可用。
- 每个行为变更使用 red-green-refactor，并形成独立提交。

---

## File responsibility map

- `models/hybrid_swa_moe/configuration_hybrid_swa_moe.py`: explicit layer lists and validation。
- `models/hybrid_swa_moe/attention.py`: QK/V-asymmetric full/SWA attention、sink、cache。
- `models/hybrid_swa_moe/moe.py`: dense/routed SwiGLU、router stats。
- `models/hybrid_swa_moe/mtp.py`: future-token heads/losses。
- `models/hybrid_swa_moe/modeling_hybrid_swa_moe.py`: decoder layers and causal LM output。
- `profiles/mimo_v25_experimental/factory.py`: manifest、generic model build、capabilities。
- `parallel/expert_parallel.py`: EP groups、dispatch、combine。
- `generation/speculative.py`: greedy and sampling verification/rejection。
- `evaluation/contracts.py` and `utils/profiling.py`: experiment reports required before ablations。
- `scripts/benchmark_hybrid_attention.py`, `benchmark_moe.py`,
  `benchmark_mtp.py`: correctness-gated、同 profile 机制测量。
- `scripts/benchmark_legacy_deltanet.py`: 独立 legacy DeltaNet 参考测量，
  不与 Hybrid SWA 模型宣称权重可比。
- `configs/model/legacy_deltanet_tiny_benchmark.yaml`: 与 Hybrid tiny
  基线同 hidden width/layer count 的 CPU DeltaNet 工程测量配置。

---

### Task 1: Define experiment reporting before implementing mechanisms

**Files:**
- Create: `src/qwen3_omni_pretrain/evaluation/contracts.py`
- Modify: `src/qwen3_omni_pretrain/utils/profiling.py`
- Create: `tests/evaluation/test_experiment_report.py`

**Interfaces:**
- Consumes: profile manifest and architecture summary.
- Produces: `RuntimeEnvironment`, `CorrectnessGate`, `BenchmarkMeasurement`, `ExperimentReport`, and `measure_peak_memory()`.

- [ ] **Step 1: Write failing report-completeness tests**

```python
def test_report_rejects_performance_without_correctness_gate():
    report = ExperimentReport(
        manifest=mimo_experiment_manifest(),
        architecture=tiny_architecture_summary(),
        environment=cpu_environment(),
        router_aux_loss_weight=0.01,
        mtp_loss_weight=0.0,
        correctness_gates=(),
        measurements=(
            BenchmarkMeasurement(
                name="decode_tokens_per_second",
                value=10.0,
                unit="tokens/s",
                dimensions={"context": 128, "output": 32},
            ),
        ),
        comparison_metadata={},
    )
    with pytest.raises(ValueError, match="correctness"):
        report.validate()


def test_report_round_trips_raw_samples_and_loss_weights():
    report = valid_cpu_report(
        router_aux_loss_weight=0.02,
        mtp_loss_weight=0.1,
        raw_samples=(1.0, 2.0, 4.0),
    )
    restored = ExperimentReport.from_json(report.to_json())
    assert restored.router_aux_loss_weight == 0.02
    assert restored.mtp_loss_weight == 0.1
    assert restored.measurements[0].raw_samples == (1.0, 2.0, 4.0)
```

- [ ] **Step 2: Run and observe missing contracts**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/evaluation/test_experiment_report.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement immutable report types**

```python
@dataclass(frozen=True)
class CorrectnessGate:
    name: str
    passed: bool
    tolerance: float | None
    observed: float | None


@dataclass(frozen=True)
class BenchmarkMeasurement:
    name: str
    value: float
    unit: str
    dimensions: Mapping[str, int | float | str | bool]
    raw_samples: tuple[float, ...] = ()


@dataclass(frozen=True)
class ExperimentReport:
    manifest: ProfileManifest
    architecture: ArchitectureSummary
    environment: RuntimeEnvironment
    router_aux_loss_weight: float
    mtp_loss_weight: float
    correctness_gates: tuple[CorrectnessGate, ...]
    measurements: tuple[BenchmarkMeasurement, ...]
    comparison_metadata: Mapping[str, int | float | str | bool]

    def validate(self) -> None:
        if (
            not math.isfinite(self.router_aux_loss_weight)
            or self.router_aux_loss_weight < 0
            or not math.isfinite(self.mtp_loss_weight)
            or self.mtp_loss_weight < 0
        ):
            raise ValueError("loss weights must be finite and non-negative")
        if self.measurements and (
            not self.correctness_gates
            or not all(gate.passed for gate in self.correctness_gates)
        ):
            raise ValueError(
                "performance measurements require passing correctness gates"
            )
        for measurement in self.measurements:
            if is_timing_or_throughput(measurement) and not measurement.raw_samples:
                raise ValueError(
                    "timing/throughput measurements require raw samples"
                )
```

Environment records commit, Python/PyTorch versions, device, precision,
hardware, kernels/fallbacks and seed. JSON uses sorted keys and rejects NaN.
Measurement validation requires finite `value` and finite non-negative timing
samples; P50/P90 are recomputed from `raw_samples` when samples are present
rather than accepted from an unauditable scalar. Serialization must preserve
every raw sample plus the exact `router_aux_loss_weight` and
`mtp_loss_weight`; tests round-trip non-zero and zero weights. A report that
mentions more than one architecture must include `weight_comparable` and a
non-empty reason when that value is false in `comparison_metadata`.

- [ ] **Step 4: Implement profiling helpers**

Provide:

- synchronized wall-clock measurement;
- CUDA peak memory reset/read when available;
- CPU RSS measurement with explicit platform field;
- percentile calculation from raw samples;
- no import-time CUDA initialization.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/evaluation/test_experiment_report.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/evaluation/contracts.py \
  src/qwen3_omni_pretrain/utils/profiling.py \
  tests/evaluation/test_experiment_report.py
git commit -m "feat: define architecture experiment reports"
```

---

### Task 2: Define the isolated Hybrid SWA-MoE configuration

**Files:**
- Create: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/__init__.py`
- Create: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/configuration_hybrid_swa_moe.py`
- Create: `src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/__init__.py`
- Create: `src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/manifest.py`
- Create: `configs/model/hybrid_swa_moe_tiny.yaml`
- Create: `tests/hybrid_swa_moe/test_config.py`

**Interfaces:**
- Consumes: profile manifest contracts.
- Produces: `HybridSwaMoeConfig` and `mimo_experiment_manifest()`.

- [ ] **Step 1: Write failing explicit-layer config tests**

```python
def test_tiny_config_has_explicit_layer_types():
    config = load_tiny_hybrid_config()
    assert config.model_type == "hybrid_swa_moe_experimental"
    assert config.mtp_num_predictors == 0
    assert config.mtp_loss_weight == 0.0
    assert config.attention_layer_types == [
        "swa", "swa", "swa", "swa", "swa", "full"
    ]
    assert config.ffn_layer_types == [
        "dense",
        "routed_moe", "routed_moe", "routed_moe",
        "routed_moe", "routed_moe",
    ]


def test_config_rejects_deltanet_and_non_sparse_router():
    raw = tiny_hybrid_config_dict()
    raw["num_experts_per_token"] = raw["num_experts"]
    with pytest.raises(ValueError, match="top_k < num_experts"):
        HybridSwaMoeConfig(**raw)


@pytest.mark.parametrize("value_scale", [0.0, -1.0, math.nan, math.inf])
def test_config_rejects_invalid_value_scale(value_scale):
    raw = tiny_hybrid_config_dict()
    raw["value_scale"] = value_scale
    with pytest.raises(ValueError, match="value_scale"):
        HybridSwaMoeConfig(**raw)
```

- [ ] **Step 2: Run and observe missing model package**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/hybrid_swa_moe/test_config.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement the config**

```python
class HybridSwaMoeConfig(PretrainedConfig):
    model_type = "hybrid_swa_moe_experimental"

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        full_num_key_value_heads: int,
        swa_num_key_value_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        rotary_dim: int,
        attention_layer_types: Sequence[str],
        ffn_layer_types: Sequence[str],
        swa_window_size: int,
        full_rope_theta: float,
        swa_rope_theta: float,
        attention_sink: bool,
        num_experts: int,
        num_experts_per_token: int,
        expert_intermediate_size: int,
        dense_intermediate_size: int,
        value_scale: float | None,
        router_aux_loss_weight: float,
        mtp_num_predictors: int,
        mtp_loss_weight: float,
        **kwargs: object,
    ) -> None:
        ...
```

Validate list lengths, allowed values, layer 0 dense, no DeltaNet fields,
head dimensions, rotary evenness, positive window, `top_k < experts`, MTP
predictors in `[0,3]`, and profile manifest.
`value_scale` must be `None` or finite and strictly positive.
`router_aux_loss_weight` and `mtp_loss_weight` must be finite and
non-negative. Reject `mtp_num_predictors == 0` with non-zero
`mtp_loss_weight`, because that configuration silently reports a loss term
that cannot exist.

- [ ] **Step 4: Add the tiny YAML**

Use:

```yaml
model_type: hybrid_swa_moe_experimental
architecture_profile: mimo_v25_experimental
vocab_size: 256
hidden_size: 128
num_hidden_layers: 6
num_attention_heads: 8
full_num_key_value_heads: 2
swa_num_key_value_heads: 4
qk_head_dim: 24
v_head_dim: 16
rotary_dim: 8
attention_layer_types: [swa, swa, swa, swa, swa, full]
ffn_layer_types: [dense, routed_moe, routed_moe, routed_moe, routed_moe, routed_moe]
swa_window_size: 128
full_rope_theta: 10000000.0
swa_rope_theta: 10000.0
attention_sink: true
value_scale: null
num_experts: 8
num_experts_per_token: 2
expert_intermediate_size: 128
dense_intermediate_size: 512
router_aux_loss_weight: 0.01
mtp_num_predictors: 0
mtp_loss_weight: 0.0
```

- [ ] **Step 5: Implement and test experiment provenance**

`mimo_experiment_manifest()` reports `MiMo-style-experiment`, identifies the
pinned MiMo config/source revisions as inspiration, records every scale
reduction in assumptions, uses the generic
`hybrid_swa_moe_experimental` model type, and always sets exact checkpoint
compatibility false. Add tests that official MiMo model/checkpoint names are
rejected from the experimental config and manifest.

- [ ] **Step 6: Run configuration and provenance tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/hybrid_swa_moe/test_config.py -q
```

Expected: all config tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/models/hybrid_swa_moe \
  src/qwen3_omni_pretrain/profiles/mimo_v25_experimental \
  configs/model/hybrid_swa_moe_tiny.yaml \
  tests/hybrid_swa_moe/test_config.py
git commit -m "feat: define Hybrid SWA MoE experiment config"
```

---

### Task 3: Implement full/SWA attention and strict window cache

**Files:**
- Create: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/attention.py`
- Create: `tests/hybrid_swa_moe/test_attention.py`
- Create: `tests/hybrid_swa_moe/test_cache.py`

**Interfaces:**
- Consumes: `AttentionKV`, `SlidingWindowKV`, per-layer config.
- Produces: `HybridSelfAttention.forward()`.

- [ ] **Step 1: Write failing SWA/full equivalence tests**

```python
def test_swa_matches_full_when_sequence_fits_window_and_sink_is_off():
    full, swa = tied_attention_pair(
        sequence_length=32,
        window_size=128,
        attention_sink=False,
    )
    hidden = torch.randn(2, 32, 128)
    positions = torch.arange(32).expand(2, -1)
    expected, _ = full(hidden, position_ids=positions)
    actual, _ = swa(hidden, position_ids=positions)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
```

`tied_attention_pair()` is an attention-only correctness fixture: it gives
both modules the same KV-head count/projection shapes, initializes the full
module once, and copies an exact name/shape inventory into SWA. It does not
reuse the production tiny YAML, whose full/SWA KV-head counts intentionally
differ.

- [ ] **Step 2: Run and observe missing attention**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/hybrid_swa_moe/test_attention.py \
  tests/hybrid_swa_moe/test_cache.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement asymmetric fused QKV and RoPE**

`HybridSelfAttention`:

- one fused QKV projection;
- Q/K head dim and V head dim may differ;
- Q uses all heads, K/V use per-layer KV heads;
- only `rotary_dim` prefix of Q/K rotates;
- full and SWA use separate theta;
- optional value scale;
- output projection maps `num_q_heads * v_head_dim` to hidden.

Public signature:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor,
    current_key_valid_mask: torch.BoolTensor | None = None,
    cache: AttentionKV | SlidingWindowKV | None = None,
    use_cache: bool = False,
) -> tuple[
    torch.Tensor,
    AttentionKV | SlidingWindowKV | None,
]:
    ...
```

- [ ] **Step 4: Implement exact attention-sink semantics**

In eager attention:

```python
scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
scores = scores + causal_or_window_mask
if self.attention_sink_bias is not None:
    sink = self.attention_sink_bias.view(1, -1, 1, 1).expand(
        scores.shape[0], -1, scores.shape[2], 1
    )
    scores = torch.cat((scores, sink), dim=-1)
probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
if self.attention_sink_bias is not None:
    probabilities = probabilities[..., :-1]
output = torch.matmul(probabilities, value)
```

The sink receives probability mass but has no value vector. If an optimized
kernel cannot reproduce this, use eager attention and record the fallback.

- [ ] **Step 5: Implement strict SWA cache**

SWA attention:

- reads only the last `window_size` keys;
- appends current unrepeated KV through `SlidingWindowKV.append`;
- appends the current key-valid mask with KV and positions;
- persistent key/value/position/mask token axis never exceeds window;
- full attention grows normally;
- rectangular mask uses absolute positions;
- cache storage bytes are exposed.

For a prefill chunk with query length `Q > window_size`, do **not** truncate
the concatenated prefix/current KV before computing attention. Compute each
current query against its own absolute-position window over
`persistent_prefix + all_current_kv`; only after all `Q` outputs are produced
may `SlidingWindowKV.append` truncate the next persistent state to its final
`window_size` valid tokens. This prevents early queries in a large chunk from
incorrectly attending to late current-chunk tokens or losing their own prefix.

- [ ] **Step 6: Test causality, cache, sink, and memory**

Cover:

- equivalence when `T<=W`, sink off;
- sink output against explicit eager formula;
- changing a token outside one SWA window cannot change the current output of
  that layer;
- cached/uncached FP32 `<=1e-5`;
- padded two-row cached/uncached parity with different valid lengths;
- a multi-chunk padded two-row case in which the shorter row's right padding
  never evicts valid KV; assert the common `SlidingWindowKV` per-row compaction
  contract after every chunk;
- greedy parity;
- prefill and 4×window decode keep SWA cache `<=W`;
- one-shot and irregular chunked prefill, including at least one
  `Q > window_size` chunk, match token-by-token SWA output/cache in FP32 while
  the returned persistent cache is still `<=W`;
- full cache grows with sequence length;
- different full/SWA RoPE bases change only their selected layers;
- `value_scale=None` is identical to scale 1, while a valid positive scale is
  applied to V exactly once before attention output;
- no full `T×T` allocation during one-token cached decode.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/hybrid_swa_moe/test_attention.py \
  tests/hybrid_swa_moe/test_cache.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/models/hybrid_swa_moe/attention.py \
  tests/hybrid_swa_moe/test_attention.py \
  tests/hybrid_swa_moe/test_cache.py
git commit -m "feat: add Hybrid full and sliding attention"
```

---

### Task 4: Implement routed-only SwiGLU MoE and router observability

**Files:**
- Create: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/moe.py`
- Create: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/modeling_hybrid_swa_moe.py`
- Create: `src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/factory.py`
- Modify: `src/qwen3_omni_pretrain/profiles/registry.py`
- Modify: `src/qwen3_omni_pretrain/utils/model_stats.py`
- Create: `tests/hybrid_swa_moe/test_moe.py`
- Create: `tests/hybrid_swa_moe/test_factory.py`
- Modify: `tests/test_model_stats.py`

**Interfaces:**
- Consumes: explicit FFN type config.
- Produces: `SwiGLU`, `RouterStats`, `RoutedMoeOutput`, `RoutedSwiGLUMoE`.

- [ ] **Step 1: Write failing routed-only module-tree tests**

```python
def test_routed_layer_has_no_dense_ffn():
    model = tiny_hybrid_model()
    assert isinstance(model.layers[0].ffn, SwiGLU)
    for layer in model.layers[1:]:
        assert isinstance(layer.ffn, RoutedSwiGLUMoE)
        assert not hasattr(layer, "shared_mlp")


def test_router_counts_include_every_selected_route():
    moe = tiny_routed_moe(experts=8, top_k=2)
    output = moe(torch.randn(2, 5, 128), collect_stats=True)
    assert output.stats.expert_token_counts.sum().item() == 2 * 5 * 2


def test_forced_routes_give_every_expert_finite_gradients():
    moe = tiny_routed_moe(experts=8, top_k=2)
    hidden = torch.randn(1, 8, 128, requires_grad=True)
    primary = torch.arange(8)
    logits = torch.full((8, 8), -20.0)
    logits[torch.arange(8), primary] = 20.0
    logits[torch.arange(8), (primary + 1) % 8] = 10.0
    with patch.object(
        moe.router,
        "forward",
        return_value=logits.view(1, 8, 8),
    ):
        output = moe(hidden, collect_stats=True)
    output.hidden_states.square().mean().backward()
    assert output.stats.expert_token_counts.tolist() == [2] * 8
    for expert in moe.experts:
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in expert.parameters()
        )
```

- [ ] **Step 2: Run and observe missing MoE**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/hybrid_swa_moe/test_moe.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement dense and expert SwiGLU**

```python
class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

`RoutedSwiGLUMoE` contains only router and expert SwiGLUs. Normalize selected
weights per token, execute selected experts, scatter-add outputs and compute
Switch-style aux loss from full router probabilities and selected load.
The gradient test patches the router output with a fixed `[B,T,E]` tensor so
that it deterministically selects every expert; production forward has no
test-only routing override.

- [ ] **Step 4: Define no-grad router stats**

```python
@dataclass(frozen=True)
class RouterStats:
    expert_token_counts: torch.Tensor
    router_entropy: torch.Tensor
    max_mean_load: torch.Tensor


@dataclass(frozen=True)
class RoutedMoeOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    stats: RouterStats | None
```

Stats are detached and never participate in the forward loss. Distributed
aggregation occurs outside the module.

- [ ] **Step 5: Assemble the first complete model and register its factory**

`HybridDecoderLayer` selects its attention and FFN exclusively from the two
explicit per-layer lists. `HybridSwaMoeForCausalLM` constructs embeddings,
decoder layers, final norm and LM head, and exposes the backward-compatible
forward signature:

```python
def forward(
    self,
    input_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    labels: torch.LongTensor | None = None,
    *,
    decoder_state: DecoderState | None = None,
    request_id: str | None = None,
    use_cache: bool = False,
    collect_router_stats: bool = False,
) -> dict[str, object]:
    ...
```

Reserved keys are `logits`, `loss`, `ce_loss`, `aux_loss`, `mtp_loss`,
`router_stats`, and `decoder_state`. The implementation must provide a finite
forward/backward pass before the factory is registered; there is no
error-raising intermediate model. The lazy factory returns the model in
`ProfileBuildResult.artifact` with `mimo_experiment_manifest()` and its
architecture summary.

At this task `mtp_num_predictors == 0`, so `mtp_loss` is `None`. When labels
are present, the exact loss equation is:

```python
loss = ce_loss + config.router_aux_loss_weight * aux_loss
```

The saved config and `ExperimentReport` record
`router_aux_loss_weight`, `mtp_num_predictors`, and `mtp_loss_weight`.
`ArchitectureSummary` keeps its common strict schema and exposes only boolean
mechanism capabilities such as `mtp_enabled`; it does not grow
profile-specific numeric top-level fields.
`router_aux_loss_weight=0.0` must make `loss` bitwise equal to `ce_loss`
without suppressing detached router observability. Implement an explicit
zero-weight branch instead of evaluating `ce_loss + 0 * aux_loss`, so a
non-finite unused auxiliary value cannot contaminate the total and the identity
is exact.

The model also implements the common typed
the exact common keyword-only
`CacheCapableModel.prefill(inputs=ModelPrefillInputs, request_id=...,
use_cache=...)` and `decode(token_ids=..., current_attention_mask=...,
decoder_state=..., request_id=...)` methods,
normalizing mappings to `CausalLMOutput`. Text-only MiMo uses
`ModelPrefillInputs.input_ids`; any later multimodal adapter supplies already
assembled embeddings. Decode has no arbitrary media/kwargs channel. Add
protocol-level cached parity tests so the serving plan can construct a
`CommonStateBackend` without casting the model's raw `forward`.

- [ ] **Step 6: Make parameter stats protocol-based**

Replace the hard-coded `isinstance(Qwen3OmniMoeMLP)` check with a protocol:

```python
class RoutedParameterInfo(Protocol):
    num_experts: int
    num_experts_per_token: int

    def expert_parameter_groups(
        self,
    ) -> Sequence[Sequence[nn.Parameter]]:
        ...
```

Implement it for legacy and hybrid modules. Preserve all existing stats fields
and add routed/shared/dense counts to the architecture summary.

- [ ] **Step 7: Test routing, complete forward, dtype, gradients, and load**

Cover:

- selected weights sum to one;
- top-k is never clamped;
- output dtype for BF16;
- a deterministic override routes at least one token through every expert,
  and every forced-selected expert receives a finite, non-zero gradient;
- stats count and entropy;
- active/total parameter ratio below one;
- module tree has dense layer 0 and routed-only later layers;
- full tiny-model forward/backward returns every reserved key with finite
  logits/loss and a request-owned cache when requested;
- registry construction returns a `ProfileBuildResult` whose manifest cannot
  claim official MiMo checkpoint compatibility;
- with identical weights and inputs, setting `router_aux_loss_weight=0.0`
  gives `loss == ce_loss` and the same logits/CE gradients as computing base
  CE without adding aux loss;
- existing legacy MoE tests still pass.

- [ ] **Step 8: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/hybrid_swa_moe/test_moe.py \
  tests/hybrid_swa_moe/test_factory.py \
  tests/test_moe.py tests/test_model_stats.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/models/hybrid_swa_moe/moe.py \
  src/qwen3_omni_pretrain/models/hybrid_swa_moe/modeling_hybrid_swa_moe.py \
  src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/factory.py \
  src/qwen3_omni_pretrain/profiles/registry.py \
  src/qwen3_omni_pretrain/utils/model_stats.py \
  tests/hybrid_swa_moe/test_moe.py \
  tests/hybrid_swa_moe/test_factory.py tests/test_model_stats.py
git commit -m "feat: add routed-only SwiGLU experts"
```

---

### Task 5: Add two-rank expert parallel dispatch and combine

**Files:**
- Create: `src/qwen3_omni_pretrain/parallel/expert_parallel.py`
- Modify: `src/qwen3_omni_pretrain/parallel/initialize.py:49-207`
- Modify: `src/qwen3_omni_pretrain/parallel/__init__.py`
- Modify: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/moe.py`
- Modify: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/modeling_hybrid_swa_moe.py`
- Modify: `src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/factory.py`
- Create: `tests/distributed/run_expert_parallel_smoke.py`
- Create: `tests/distributed/test_expert_parallel.py`
- Modify: `tests/hybrid_swa_moe/test_factory.py`

**Interfaces:**
- Consumes: selected expert IDs/weights and routed tokens.
- Produces: `ExpertParallelContext`, `DispatchedTokens`,
  `ParallelTopology`, `dispatch_to_experts()`, `combine_from_experts()`, and
  an EP-aware Hybrid model/factory path.

- [ ] **Step 1: Write a failing unsupported-topology test**

```python
def test_expert_parallel_rejects_tp_combination():
    with pytest.raises(ValueError, match="EP.*TP"):
        validate_parallel_topology(
            world_size=4,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=2,
            tensor_parallel_size=2,
        )


def test_ep_one_preserves_existing_tp_pp_topology():
    topology = validate_parallel_topology(
        world_size=4,
        data_parallel_size=1,
        pipeline_parallel_size=2,
        expert_parallel_size=1,
        tensor_parallel_size=2,
    )
    assert topology.expert_parallel_size == 1
    assert topology.pipeline_parallel_size == 2
    assert topology.tensor_parallel_size == 2
```

- [ ] **Step 2: Add a two-rank parity smoke script**

The script initializes Gloo with EP=2, constructs identical router weights and
sharded experts, runs the same tokens as a single-process unsharded reference,
backpropagates a non-constant scalar loss, and has rank zero assert
output/loss/input-grad, router-grad and every actually selected expert-grad
parity. Deterministic router weights plus basis-vector inputs route at least
one token to every global expert while keeping logits connected to router
parameters, so neither router- nor expert-gradient assertions can pass
vacuously.

Run before implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/torchrun \
  --standalone --nproc-per-node=2 \
  tests/distributed/run_expert_parallel_smoke.py
```

Expected: FAIL because EP helpers do not exist.

- [ ] **Step 3: Extend topology explicitly**

Define and return an immutable topology rather than passing four unrelated
integers:

```python
@dataclass(frozen=True)
class ParallelTopology:
    data_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    tensor_parallel_size: int
```

Logical topology is `[DP,PP,EP,TP]`. Add expert group/rank/size getters.
Validation has two branches:

- `EP == 1`: require only the project's pre-existing world-size/divisibility
  constraints and leave existing DP/PP/TP combinations unchanged;
- `EP > 1`: accept exactly `DP=1,PP=1,EP=2,TP=1`.

Invalid `EP>1` combinations fail before any `dist.new_group` call. Add
regression tests for at least an existing TP=2 path and PP=2 path with EP=1,
plus parameterized failures for DP>1, PP>1, TP>1, EP not equal to 2, and world
size mismatch when EP>1.

- [ ] **Step 4: Implement all-to-all dispatch/combine**

`dispatch_to_experts()`:

1. flatten selected token routes;
2. compute owning rank by contiguous expert partition;
3. stable-sort routes by destination;
4. exchange integer split sizes and route metadata under `torch.no_grad()`
   with raw `torch.distributed` collectives;
5. exchange floating token payloads with the autograd-enabled
   `torch.distributed.nn.functional.all_to_all`;
6. return token values, local expert IDs, source token/route indices and
   routing weights.

Concretely, allocate one receive tensor per source count and invoke:

```python
from torch.distributed.nn import functional as dist_nn

received_chunks = dist_nn.all_to_all(
    output_tensor_list=receive_buffers,
    input_tensor_list=list(sorted_tokens.split(send_counts)),
    group=context.group,
)
```

`combine_from_experts()` sends expert outputs back with the same autograd-
enabled API, then multiplies by the source-local differentiable routing
weights and scatter-adds into original token order. Raw
`dist.all_to_all_single` is forbidden for floating token/expert-output
payloads because it would sever autograd. Support empty destinations and
non-divisible token counts. A source-inspection unit test prevents a future
regression to a non-differentiable payload collective.

- [ ] **Step 5: Integrate only the hybrid MoE**

Legacy MoE remains replicated. Hybrid `RoutedSwiGLUMoE` uses EP only when an
initialized `ExpertParallelContext` is explicitly passed. Each rank constructs
only its expert shard.

Define the context and thread it without reading hidden globals:

```python
@dataclass(frozen=True)
class ExpertParallelContext:
    group: dist.ProcessGroup
    rank: int
    world_size: int
    local_expert_start: int
    local_expert_end: int


class HybridSwaMoeForCausalLM(nn.Module):
    def __init__(
        self,
        config: HybridSwaMoeConfig,
        *,
        expert_parallel_context: ExpertParallelContext | None = None,
    ) -> None:
        ...

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        *,
        decoder_state: DecoderState | None = None,
        request_id: str | None = None,
        use_cache: bool = False,
        collect_router_stats: bool = False,
        expert_parallel_context: ExpertParallelContext | None = None,
    ) -> dict[str, object]:
        ...
```

The optional forward context must be the same context stored at construction;
passing a different context fails instead of silently mixing expert shards.
`MimoV25ExperimentalFactory` receives immutable `topology` and optional
`expert_parallel_context` constructor fields. Its registry-created defaults
are the existing EP=1 topology. For distributed builds,
`factory.validate(request)` first validates topology/context consistency, and
only then parses the model config and allocates
`HybridSwaMoeForCausalLM(config, expert_parallel_context=...)`. A spy test
patches the model constructor and proves invalid topology raises before that
constructor is called.

The factory surface is exact:

```python
class MimoV25ExperimentalFactory:
    def __init__(
        self,
        *,
        topology: ParallelTopology | None = None,
        expert_parallel_context: ExpertParallelContext | None = None,
    ) -> None:
        ...

    def validate(self, request: ProfileBuildRequest) -> None:
        ...

    def build(self, request: ProfileBuildRequest) -> ProfileBuildResult:
        ...
```

`topology=None` means “read and validate the project's existing DP/PP/TP
topology with EP fixed to 1”; it never infers EP>1 from world size.

- [ ] **Step 6: Run distributed and local tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/torchrun \
  --standalone --nproc-per-node=2 \
  tests/distributed/run_expert_parallel_smoke.py

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/distributed/test_expert_parallel.py \
  tests/hybrid_swa_moe/test_moe.py \
  tests/hybrid_swa_moe/test_factory.py -q
```

Expected: all commands pass. The two-rank script explicitly asserts finite,
non-zero input gradients and finite, non-zero gradients for every forced-
selected local expert on both ranks, in addition to unsharded-reference
parity.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/parallel \
  src/qwen3_omni_pretrain/models/hybrid_swa_moe/moe.py \
  src/qwen3_omni_pretrain/models/hybrid_swa_moe/modeling_hybrid_swa_moe.py \
  src/qwen3_omni_pretrain/profiles/mimo_v25_experimental/factory.py \
  tests/distributed/run_expert_parallel_smoke.py \
  tests/distributed/test_expert_parallel.py \
  tests/hybrid_swa_moe/test_factory.py
git commit -m "feat: dispatch routed experts across ranks"
```

---

### Task 6: Add teacher-forced MTP heads and losses

**Files:**
- Create: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/mtp.py`
- Modify: `src/qwen3_omni_pretrain/models/hybrid_swa_moe/modeling_hybrid_swa_moe.py`
- Modify: `configs/model/hybrid_swa_moe_tiny.yaml`
- Modify: `tests/hybrid_swa_moe/test_config.py`
- Create: `tests/hybrid_swa_moe/test_mtp.py`

**Interfaces:**
- Consumes: final hidden states and labels.
- Produces: `MtpTrainingOutput` and `MultiTokenPredictor.forward()`.

- [ ] **Step 1: Write failing future-offset tests**

```python
def test_predictor_j_targets_offset_j_plus_two():
    predictor = tiny_mtp(num_predictors=2, vocab_size=16)
    hidden = torch.randn(1, 6, 8)
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = predictor(hidden, labels=labels)
    assert output.target_offsets == (2, 3)
    assert output.valid_token_counts == (4, 3)


def test_tiny_config_enables_completed_mtp_module():
    config = load_tiny_hybrid_config()
    assert config.mtp_num_predictors == 1
    assert config.mtp_loss_weight == 0.1
```

Replace the two Task 2 bootstrap assertions (`0` and `0.0`) with this Task 6
test; do not leave mutually contradictory assertions in
`test_tiny_config_has_explicit_layer_types`.

- [ ] **Step 2: Run and observe missing MTP**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/hybrid_swa_moe/test_mtp.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement predictor outputs**

```python
@dataclass(frozen=True)
class MtpTrainingOutput:
    logits: tuple[torch.Tensor, ...]
    losses: tuple[torch.Tensor, ...]
    loss: torch.Tensor | None
    target_offsets: tuple[int, ...]
    valid_token_counts: tuple[int, ...]
```

First change the tiny YAML from the Task 2 bootstrap values to exactly:

```yaml
mtp_num_predictors: 1
mtp_loss_weight: 0.1
```

Do this in Task 6, not Task 2, so every earlier commit constructs a complete
model without referring to a missing MTP module.

Each predictor has its own norm, projection/decoder block and LM head. Predictor
index 0 targets two tokens ahead; index `j` targets `j+2`. Shift hidden/labels
exactly, preserve `-100`, normalize each loss by its own valid count and average
only non-empty losses.

- [ ] **Step 4: Integrate model loss without changing base distribution**

The main model:

- computes base next-token CE unchanged;
- computes MTP loss only during training or explicit request;
- returns separate `mtp_loss`;
- combines losses with the single exact equation:

  ```python
  loss = (
      ce_loss
      + config.router_aux_loss_weight * aux_loss
      + config.mtp_loss_weight * mtp_loss
  )
  ```

  A missing aux or MTP term contributes a scalar zero on the same
  device/dtype; it is still returned as `None` when the mechanism was not
  executed;
- each zero weight is handled by an explicit branch that omits the addition
  entirely; do not rely on multiplication by zero;
- reads that weight from `HybridSwaMoeConfig` and records it in the checkpoint
  config and `ExperimentReport`; the common architecture summary records only
  `mtp_enabled: true`;
- does not append draft tokens during ordinary generation.

- [ ] **Step 5: Test masks, zero-token cases, and gradients**

Cover 1–3 predictors, short sequences, all-ignored labels, padding, finite
gradients, deterministic resume and unchanged base logits when MTP is disabled.
With identical weights/input/RNG, test both zero-weight identities:

- `router_aux_loss_weight=0` excludes aux loss exactly and leaves
  `loss == ce_loss + mtp_loss_weight * mtp_loss`;
- `mtp_loss_weight=0` excludes MTP exactly and leaves
  `loss == ce_loss + router_aux_loss_weight * aux_loss`.

Also assert logits and the corresponding unweighted CE gradients are unchanged
by enabling MTP, and that report JSON records the effective MTP weight `0.1`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/hybrid_swa_moe/test_mtp.py \
  tests/hybrid_swa_moe/test_config.py \
  tests/evaluation/test_experiment_report.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/models/hybrid_swa_moe/mtp.py \
  src/qwen3_omni_pretrain/models/hybrid_swa_moe/modeling_hybrid_swa_moe.py \
  configs/model/hybrid_swa_moe_tiny.yaml \
  tests/hybrid_swa_moe/test_config.py \
  tests/hybrid_swa_moe/test_mtp.py
git commit -m "feat: train multi-token predictors"
```

---

### Task 7: Implement correct speculative verification and rollback

**Files:**
- Create: `src/qwen3_omni_pretrain/generation/__init__.py`
- Create: `src/qwen3_omni_pretrain/generation/speculative.py`
- Create: `tests/hybrid_swa_moe/test_speculative.py`

**Interfaces:**
- Consumes: base target model, MTP draft, immutable `DecoderState`.
- Produces: `DraftProposal`, `SpeculativeStepOutput`,
  `verify_greedy_proposal()`, and `verify_sampled_proposal()`.

- [ ] **Step 1: Write failing greedy rollback tests**

```python
def test_greedy_rejection_restores_committed_state():
    target = scripted_target(tokens=[5, 6, 9])
    draft = scripted_draft(tokens=[5, 7, 8])
    initial = tiny_decoder_state(
        seen_tokens=torch.tensor([4], dtype=torch.long)
    )
    result = speculative_step_greedy(
        target=target,
        draft=draft,
        state=initial,
        max_draft_tokens=3,
    )
    assert result.committed_token_ids.tolist() == [5, 6]
    assert result.accepted == 1
    assert result.rejected == 1
    assert result.decoder_state.seen_tokens.tolist() == [6]
    assert initial.seen_tokens.tolist() == [4]
```

- [ ] **Step 2: Run and observe missing generation module**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/hybrid_swa_moe/test_speculative.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Define proposals and outputs**

```python
@dataclass(frozen=True)
class DraftProposal:
    token_ids: torch.LongTensor
    probabilities: tuple[torch.Tensor, ...]
    state: DecoderState


@dataclass(frozen=True)
class SpeculativeStepOutput:
    committed_token_ids: torch.LongTensor
    decoder_state: DecoderState
    proposed: int
    accepted: int
    rejected: int
```

- [ ] **Step 4: Implement greedy verification**

Accept consecutive draft tokens while they equal target argmax. At first
mismatch, commit target argmax. Rebuild returned state by replaying accepted
tokens plus correction from the last immutable committed state. Never return
the draft state after rejection.

- [ ] **Step 5: Implement sampled verification**

For draft token `x` with target probability `p(x)` and draft probability
`q(x)`:

```python
acceptance = min(1.0, p_x / q_x)
```

If rejected, sample from normalized `clamp(p-q, min=0)`. If all proposals are
accepted, sample one additional token from target distribution. Handle zero
draft probability and zero residual mass with exact documented fallbacks.

- [ ] **Step 6: Verify distributions and state repair**

Tests:

- greedy output equals ordinary greedy decode;
- exhaustive tiny categorical p/q cases match analytic distribution;
- Monte Carlo seeded frequency within tolerance;
- forced rejection state equals ordinary target replay;
- session ownership;
- proposed/accepted/rejected counts;
- cache bytes and media state restored;
- random-token and natural-token acceptance reported separately.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/hybrid_swa_moe/test_speculative.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/generation \
  tests/hybrid_swa_moe/test_speculative.py
git commit -m "feat: verify speculative MTP proposals"
```

---

### Task 8: Run isolated SWA, MoE, and MTP benchmarks

**Files:**
- Create: `scripts/benchmark_hybrid_attention.py`
- Create: `scripts/benchmark_moe.py`
- Create: `scripts/benchmark_mtp.py`
- Create: `scripts/benchmark_legacy_deltanet.py`
- Create: `configs/model/legacy_deltanet_tiny_benchmark.yaml`
- Create: `tests/evaluation/test_benchmark_contracts.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: experiment report types and completed mechanisms.
- Produces: three same-profile mechanism benchmark CLIs plus one explicitly
  non-weight-comparable legacy DeltaNet reference CLI.

- [ ] **Step 1: Write failing output-contract tests**

Each script must expose a pure `run_benchmark(args) -> ExperimentReport` tested
with tiny CPU inputs. Tests reject reports missing manifest, architecture,
correctness gates, active parameters, raw timing samples, exact
`router_aux_loss_weight`/`mtp_loss_weight`, or environment.

Add a contract test that loads the ordinary Hybrid, explicitly paired
equal-KV-head, and legacy DeltaNet reports and
asserts:

```python
assert hybrid.comparison_metadata["weight_comparable"] is False
assert hybrid.comparison_metadata["reason"] == (
    "full and SWA KV projection shapes differ"
)
assert paired.comparison_metadata["weight_comparable"] is True
assert paired.comparison_metadata["kv_heads"] == 4
assert paired.comparison_metadata["copied_parameter_digest"]
assert legacy.comparison_metadata["weight_comparable"] is False
assert legacy.comparison_metadata["reason"] == (
    "different model class and independently initialized weights"
)
assert legacy.mtp_loss_weight == 0.0
```

- [ ] **Step 2: Implement attention benchmark**

Command:

```bash
PYTHONPATH=src .venv-prototype/bin/python \
  scripts/benchmark_hybrid_attention.py \
  --config configs/model/hybrid_swa_moe_tiny.yaml \
  --attention-modes full swa \
  --prompt-lengths 128 512 2048 4096 \
  --output-lengths 32 128 \
  --dtype bfloat16 --device cuda \
  --output results/d6-attention.json
```

Report cached/uncached parity, latency, tokens/s, cache bytes and peak memory.
The command above preserves the checked-in full/SWA KV-head counts and marks
the comparison non-weight-comparable. Run a second microbenchmark with
`--paired-kv-heads 4`; it constructs two attention-only configs with identical
projection shapes, initializes one module, copies all parameters by exact
name/shape into the other, verifies the inventory/digest, and only then emits
the `paired` relative-ratio row. It does not replace or relabel the ordinary
D6 model rows.

- [ ] **Step 3: Implement MoE benchmark**

Command:

```bash
PYTHONPATH=src .venv-prototype/bin/python \
  scripts/benchmark_moe.py \
  --config configs/model/hybrid_swa_moe_tiny.yaml \
  --experts 8 --top-k 2 --baseline dense \
  --sequence-lengths 128 1024 \
  --output results/d6-moe.json
```

Report matched dense baseline, total/active parameters, router entropy,
expert counts, max/mean load, latency and memory.

- [ ] **Step 4: Implement MTP benchmark**

Command:

```bash
PYTHONPATH=src .venv-prototype/bin/python \
  scripts/benchmark_mtp.py \
  --config configs/model/hybrid_swa_moe_tiny.yaml \
  --prompt-sources natural random \
  --draft-lengths 1 3 --batch-sizes 1 4 \
  --output results/d6-mtp.json
```

Report distribution gate, greedy parity, proposed/accepted/rejected,
acceptance length and end-to-end speedup. Predictor count remains one unless a
reviewed report shows correctness and speedup above one.

- [ ] **Step 5: Add the independent legacy DeltaNet baseline**

Create a CPU-runnable legacy text-model config with this exact mechanism
shape:

```yaml
model_type: qwen3_omni_prototype
architecture_profile: legacy_prototype
vocab_size: 256
hidden_size: 128
intermediate_size: 512
num_hidden_layers: 6
num_attention_heads: 8
num_key_value_heads: 8
max_position_embeddings: 512
rope_theta: 10000.0
rope_partial_factor: 1.0
thinker_config:
  hidden_size: 128
  intermediate_size: 512
  num_hidden_layers: 6
  num_attention_heads: 8
  num_key_value_heads: 8
  max_position_embeddings: 512
  use_moe: false
  routing_kind: dense
  use_flash_attention: false
  use_deltanet: true
  deltanet_layer_indices: "0,1,2,3,4"
  deltanet_kernel_size: 3
  deltanet_num_heads: 8
  deltanet_chunk_size: 0
bos_token_id: 1
eos_token_id: 2
pad_token_id: 0
```

Run it separately:

```bash
PYTHONPATH=src .venv-prototype/bin/python \
  scripts/benchmark_legacy_deltanet.py \
  --config configs/model/legacy_deltanet_tiny_benchmark.yaml \
  --sequence-lengths 128 512 \
  --dtype float32 --device cpu \
  --output results/d6-legacy-deltanet.json
```

The legacy report uses the `legacy_prototype` manifest, records absolute
latency, tokens/s, peak memory and parameter counts, and gates measurement on
finite forward/backward, causal-prefix invariance and seeded-repeatability
tests. It must set:

```python
comparison_metadata={
    "primary": "mimo_v25_experimental",
    "subject": "legacy_deltanet",
    "weight_comparable": False,
    "reason": "different model class and independently initialized weights",
}
```

The D6 matrix has separate rows for Hybrid full attention, Hybrid SWA and
legacy DeltaNet. The checked-in tiny model deliberately uses different
`full_num_key_value_heads` and `swa_num_key_value_heads`, so its ordinary
full/SWA rows are **not** weight-comparable and report absolute measurements
only. A separate paired attention microbenchmark may report a relative ratio
only after constructing two configs with the same KV-head count and projection
shapes, copying every compatible parameter by exact name/shape, and recording
the copied-parameter inventory and digest. The legacy row also reports
absolute measurements and is never used for numerical-parity, quality, or
checkpoint-compatibility claims.

- [ ] **Step 6: Run tests and full regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/hybrid_swa_moe \
  tests/evaluation/test_benchmark_contracts.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q
```

Expected: zero failures.

- [ ] **Step 7: Document experiment labels and commit**

README must state that results are mechanism ablations on a generic tiny model,
not MiMo quality or checkpoint reproduction. It must also state that the
ordinary Hybrid full/SWA rows have different KV projection shapes and the
legacy DeltaNet row uses a different class and independently initialized
weights, so both are absolute engineering baselines. Only the separately
constructed equal-KV-head pair may be called a tied-weight ablation.

```bash
git add scripts/benchmark_hybrid_attention.py \
  scripts/benchmark_moe.py scripts/benchmark_mtp.py \
  scripts/benchmark_legacy_deltanet.py \
  configs/model/legacy_deltanet_tiny_benchmark.yaml \
  tests/evaluation/test_benchmark_contracts.py README.md
git commit -m "bench: compare Hybrid SWA MoE mechanisms"
```

---

## Plan completion gate

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/hybrid_swa_moe \
  tests/evaluation tests/distributed/test_expert_parallel.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/torchrun \
  --standalone --nproc-per-node=2 \
  tests/distributed/run_expert_parallel_smoke.py

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m compileall -q src tests scripts

git diff --check
git status --short
```

The plan is complete only when:

- the generic profile never uses official MiMo class/checkpoint identity;
- SWA/full, sink and strict `O(W)` cache tests pass;
- routed layers contain no dense/shared FFN and report real routing load;
- EP=2 matches the single-process reference for output/loss/gradients;
- MTP offsets, masks and losses are correct;
- greedy speculation is identical and sampling preserves the target
  distribution;
- benchmarks cannot emit performance without passing correctness gates;
- the full prototype suite remains green;
- all changes are intentional and committed.
