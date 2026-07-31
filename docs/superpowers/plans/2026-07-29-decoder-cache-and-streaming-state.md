# Decoder Cache and Incremental Decode State Implementation Plan

**Goal:** 为 legacy full-attention Thinker 建立请求隔离、可恢复且不别名的
prefill/decode KV state，使 cached 与 uncached greedy 输出一致，并保证 legacy
媒体 encoder 每个请求最多执行一次。

尽管文件名保留了 `streaming-state`，本计划中的 “streaming” 仅指模型内部增量
decode state。本计划不提供公开 iterator/session streaming API，不实现 beam、
speculative rejection 或通用 state truncate。

**Architecture:** 公共 `runtime` 包定义 typed position、KV、owner、checkpoint 和
model protocol。legacy standard/TP full-attention 实现该 protocol；现有训练
`forward()` 保持兼容。model boundary 是 rectangular causal bias 的唯一 owner。
当前 DeltaNet 没有经过验证的 recurrent cache，因此 cache 请求必须在任何 compute
前失败；只有 CLI 显式允许时才回到有结构化告警的既有 uncached loop。

**Dependencies:** profile/oracle 与 media-sequence/TM-RoPE 计划已经完成。公共位置
合同必须直接复用 `multimodal.types.PositionBatch`，不得复制较弱版本。

**Tech stack:** Python 3.10、PyTorch 2.10、frozen dataclass、runtime-checkable
Protocol、CPU Gloo、pytest、JSON benchmark。

---

## Support matrix and non-goals

后续任务和测试不得在没有新设计审查的情况下扩大下表范围。

| Capability | 本计划结果 |
| --- | --- |
| legacy standard、全 MHA、prefill/decode KV | 支持 |
| legacy TP、全 MHA、local-shard KV | 仅在 2-rank CPU Gloo parity 后支持 |
| legacy explicit/implicit DeltaNet hybrid cache | 不支持；compute 前失败 |
| legacy DeltaNet uncached fallback | 仅显式 CLI flag，输出结构化 warning |
| Qwen3 official reference weight runtime | 不支持；仍是 oracle/facade |
| Qwen3Disjoint position/state round-trip | common typed contract 支持；不宣称 reference runtime |
| Qwen3.5/MiMo native cache | 只保留 typed adapter 边界，由下游计划实现 |
| engine batch | 仅 batch size 1 |
| beam | `beam_search=false`；`num_beams != 1` prefill 前失败 |
| generic state truncate | `state_truncate=false`；无 `truncate()` API |
| speculative decode | `speculative_decode=false`；不实现 rejection engine |
| public iterator/session streaming | `streaming_generation=false` |
| internal cached greedy decode | capability 名为 `incremental_decode_state` |

稳定的 unsupported/error code 至少包含：

- `DELTA_NET_UNSUPPORTED`
- `BEAM_UNSUPPORTED`
- `TRUNCATE_UNSUPPORTED`
- `SPECULATIVE_UNSUPPORTED`
- `PROFILE_RUNTIME_UNSUPPORTED`
- `CONTEXT_OVERFLOW`
- `STATE_OWNER_MISMATCH`

---

## Global correctness contracts

- 每个 model call 接收一个 committed snapshot，并产生独立 candidate snapshot。
  异常或取消时丢弃 candidate；旧 snapshot 的内容、fingerprint 和 storage pointer
  不变。该能力可供未来 snapshot/replay speculation 使用，但本计划不实现
  speculation 或 rollback。
- state 的授权身份是至少 128-bit 随机 owner nonce。display request ID 只用于日志，
  同名 session 不能互相使用 state。
- state 是 **structurally immutable, clone-detached, non-aliasing snapshot**。
  frozen dataclass 本身不等价于 tensor value immutable。
- full-attention KV 保存 RoPE 后、尚未 repeat 的 local K/V，形状
  `[B,Hkv,S,Dk]` 与 `[B,Hkv,S,Dv]`；允许 `Dk != Dv`。
- `seen_tokens` 是 `[B] long` 有效 token 计数；storage position 由独立 typed
  cursor 管理。padding、legacy 两个固定 prefix slot 与显式 position gap 不能由
  `seen_tokens` 反推。
- `DecoderPositionState` 是 full-attention history position 的唯一真源；每层 KV
  不重复保存 position/delta/axis metadata。
- `PositionBatch.validate(key_valid_mask)` 是公共位置验证的唯一基础合同。legacy
  RoPE adapter 只接受一轴、有效值为整数且未越界的位置。
- model boundary 每次调用只构造一份 `[B,1,Q,P+Q]` rectangular additive bias；
  attention 层不得再次生成 causal mask。
- `use_cache=True` 时，owner、batch、dtype/device、position、context、training mode、
  DeltaNet capability 和 exact layer set 全部在 embedding/QKV/media compute 前验证。
- cache model call 全部运行在 `torch.inference_mode()`；返回 state 的 tensor
  `requires_grad=False` 且 `grad_fn is None`。
- FP32 cached/uncached logits 的 `max_abs` 必须直接计算并满足 `<=1e-5`；随后再
  检查 greedy token exact。
- raw legacy media 只允许出现在 `decoder_state is None` 的 prefill。任何 supplied
  state 都表示 decode，即使它是 empty/partial state。
- snippet 中的 `...` 只表示签名节选。任务提交不得保留 stub，并且每项行为必须按
  red-green-refactor 实现后独立提交。

---

## File responsibility map

- `runtime/capabilities.py`: stable capability/error values 和实际 layer scan。
- `runtime/state.py`: owner、position cursor、KV/state snapshot、字节统计。
- `runtime/protocols.py`: typed prefill/decode input 与 cache-capable model protocol。
- `runtime/generation.py`: legacy-only greedy pending-token checkpoint engine。
- `modeling_thinker_text.py`: standard MHA cache、mask、model adapter。
- `modeling_thinker_text_tp.py`: TP local-shard cache 与相同 contract。
- `modeling_thinker_vision_audio.py`: legacy media prefill-only typed union。
- `architecture/summary.py` 与 legacy factory: capability publication。
- `cli_infer_thinker.py`: Stage1/Stage2 cache selection 和 explicit fallback。
- `benchmark_decode_cache.py`: correctness-gated JSON latency/memory benchmark。

---

### Task 0: Freeze capability, error and operation scope

**Files:**

- Create: `src/qwen3_omni_pretrain/runtime/__init__.py`
- Create: `src/qwen3_omni_pretrain/runtime/capabilities.py`
- Create: `tests/runtime/test_cache_capabilities.py`

**Produces:** `CacheErrorCode`, `CacheCapabilityError`, `CacheSupport`, and a
strict actual-layer scanner. This task publishes no true cache capability yet.

- [ ] **Step 1: Write failing capability tests**

Test exact enum/string values, structured `code`/`reason`, and the support
matrix default:

```python
support = CacheSupport.unimplemented()
assert support.incremental_decode_state is False
assert support.streaming_generation is False
assert support.beam_search is False
assert support.state_truncate is False
assert support.speculative_decode is False
```

Reject bool-as-int/zero/negative `num_beams`, `num_beams != 1`, truncate and
speculative requests with stable codes. These helpers must be callable before
tokenizer/model/media work.

- [ ] **Step 2: Implement strict error and capability values**

`CacheCapabilityError` is a typed runtime error with public `code` and `reason`;
its string is diagnostic only. Add strict helpers for beam/truncate/speculative
operation gates. Do not add `DecoderState.truncate()` or batch reorder APIs.

Define actual-layer inspection around constructed decoder layers, not
`ArchitectureSummary.cache_type`. The scanner must identify both explicit
DeltaNet indices and the current implicit 3:1 default by inspecting every
constructed layer's block implementation/type. Until Tasks 2–3 implement the
protocol, it reports `incremental_decode_state=False` even for all-MHA models.

- [ ] **Step 3: Prove fail-before-compute**

Use call-count fakes to show beam/truncate/speculative rejection occurs before
tokenizer, embedding, attention, DeltaNet and media calls. Add fixtures for
explicit DeltaNet and `use_deltanet=true` with empty indices (implicit 3:1).

- [ ] **Step 4: Run and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime/test_cache_capabilities.py -q

git add src/qwen3_omni_pretrain/runtime tests/runtime/test_cache_capabilities.py
git commit -m "feat: define incremental decode capabilities"
```

---

### Task 1: Define typed non-aliasing decoder state and protocols

**Files:**

- Create: `src/qwen3_omni_pretrain/runtime/state.py`
- Create: `src/qwen3_omni_pretrain/runtime/protocols.py`
- Create: `tests/runtime/test_decoder_state.py`
- Modify: `src/qwen3_omni_pretrain/runtime/__init__.py`

**Produces:** `StateOwner`, `AttentionKV`, `SlidingWindowKV`, position cursors,
`DecoderPositionState`, `LegacyProcessedPrefix`, `DecoderState`, model input
types, `CausalLMOutput`, and `CacheCapableModel`.

- [ ] **Step 1: Write failing owner and snapshot-isolation tests**

Cover two concurrent sessions with display ID `"r1"`, a later new `"r1"`
session, nonce mismatch, and exact owner preservation in derived snapshots.
Mutate every caller tensor after construction and prove state tensor values are
unchanged and storage pointers do not alias. Candidate append must not alter the
old snapshot, including on injected exception.

- [ ] **Step 2: Implement owner and position state**

```python
@dataclass(frozen=True)
class StateOwner:
    display_request_id: str
    nonce: bytes

    @classmethod
    def fresh(cls, display_request_id: str) -> StateOwner:
        ...  # secrets.token_bytes(16) or stronger


@dataclass(frozen=True)
class LegacyPositionCursor:
    next_storage_position: torch.LongTensor  # [B]


@dataclass(frozen=True)
class Qwen3DisjointPositionCursor:
    next_text_position: torch.Tensor  # [B], profile dtype/device
    rope_deltas: torch.Tensor          # [B,1]
    axis_names: tuple[str, ...]


@dataclass(frozen=True)
class DecoderPositionState:
    cached: PositionBatch
    key_valid_mask: torch.BoolTensor
    continuation: LegacyPositionCursor | Qwen3DisjointPositionCursor
```

Rules:

- `display_request_id` is a nonblank exact string; nonce is exact bytes with at
  least 16 bytes. Authorization compares nonce, never display text.
- every public ingress clone+detaches tensors before storing;
- position construction and append call
  `PositionBatch.validate(key_valid_mask)` and validate cursor batch/device/type;
- legacy cursor cannot be reconstructed from valid count. It preserves explicit
  offsets/gaps, padding and the wrapper's two storage prefix slots;
- Qwen3 cursor is created/advanced only by the Qwen3-disjoint adapter/builder and
  retains axis/delta metadata. This plan supports only previously approved
  non-joint contiguous semantics; split/interleaved/joint AV fails fast.

- [ ] **Step 3: Implement strict KV snapshots**

```python
@dataclass(frozen=True)
class AttentionKV:
    key: torch.Tensor
    value: torch.Tensor
    key_valid_mask: torch.BoolTensor
```

Validate rank-4 K/V with common B/Hkv/S/device/dtype, independent Dk/Dv, and
same-device bool `[B,S]` mask. `append()` validates the complete current chunk
before allocating and returns clone-detached non-aliasing storage. It never
uses `copy_`, in-place masking or mutable capacity buffers.

`SlidingWindowKV` uses the same K/V fields plus `window_size` and a validated
`PositionBatch` window. Append compacts each row independently by valid mask,
gathers K/V/all position axes with the same indices, keeps the last
`window_size` valid entries, then right-pads a rectangle. An irregular two-row
oracle must prove padded storage slots neither evict nor advance valid tokens.

- [ ] **Step 4: Implement typed state partitions**

```python
@runtime_checkable
class StatePartition(Protocol):
    @property
    def batch_size(self) -> int: ...
    def clone_detached(self) -> StatePartition: ...
    def logical_tensor_bytes(self) -> int: ...


@dataclass(frozen=True)
class LegacyProcessedPrefix:
    has_image: tuple[bool, ...]
    has_audio: tuple[bool, ...]
    prefix_storage_length: int  # exactly 2


@dataclass(frozen=True)
class DecoderState:
    owner: StateOwner
    seen_tokens: torch.LongTensor
    position: DecoderPositionState | None
    full_attention_kv: Mapping[int, AttentionKV]
    swa_kv: Mapping[int, SlidingWindowKV]
    processed_media: LegacyProcessedPrefix | None
    gdn_state: StatePartition | None
    talker_state: StatePartition | None
    mtp_state: StatePartition | None
    codec_state: StatePartition | None
```

No `Mapping[str, object]` or `object | None` is permitted. Unimplemented
partitions remain `None`. Validate `[B] long` non-negative `seen_tokens`, exact
batch/device agreement across position/cache/per-batch partitions, integer
non-negative layer keys, immutable copied mappings, and consistent KV lengths,
masks, dtype/device.

`advance_seen_tokens(valid_count_delta)` requires exact `[B] long`, non-negative,
same-device input and returns a new state. It rejects bool and does not store
caller storage. There is no reorder/truncate method.

Byte metrics are separate:

- `logical_tensor_bytes(partition=None)` counts logical tensor payload;
- `unique_allocated_bytes(partition=None)` is explicitly best-effort, dedupes
  underlying storage and reports request-shared media separately.

- [ ] **Step 5: Define strict model inputs and protocol**

```python
@dataclass(frozen=True)
class LegacyMediaPrefillInputs:
    pixel_values: torch.Tensor | None
    audio_values: torch.Tensor | None
    has_image: torch.BoolTensor
    has_audio: torch.BoolTensor


@dataclass(frozen=True)
class ModelPrefillInputs:
    input_ids: torch.LongTensor | None
    inputs_embeds: torch.Tensor | None
    key_valid_mask: torch.BoolTensor
    position_batch: PositionBatch | None
    media: LegacyMediaPrefillInputs | None = None


@dataclass(frozen=True)
class ModelDecodeInputs:
    token_ids: torch.LongTensor
    current_key_valid_mask: torch.BoolTensor
    position_batch: PositionBatch
    decoder_state: DecoderState


@dataclass(frozen=True)
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    ce_loss: torch.Tensor | None
    aux_loss: torch.Tensor | None
    decoder_state: DecoderState | None
    hidden_states: tuple[torch.Tensor, ...] | None


class CacheCapableModel(Protocol):
    def prefill(
        self,
        *,
        inputs: ModelPrefillInputs,
        owner: StateOwner,
        use_cache: bool,
    ) -> CausalLMOutput: ...

    def decode(
        self,
        *,
        inputs: ModelDecodeInputs,
        owner: StateOwner,
    ) -> CausalLMOutput: ...
```

`ModelPrefillInputs` requires exactly one of IDs/embeddings, nonempty `[B,Q]`
shape, same-device bool mask and validates supplied position via
`position_batch.validate(key_valid_mask)`. `ModelDecodeInputs` performs the
same coupled checks and exact owner/batch match. Decode has no raw-media field.
`CausalLMOutput` validates floating rank-3 logits, coupled batch/token axes,
optional scalar losses, tuple hidden states and detached owned state when cache
is enabled; malformed adapters fail at this boundary rather than later sampling.

Legacy position adapter accepts only `axis_names == ("sequence",)`,
`[1,B,Q]`, finite integral valid values, masked zeros, and valid max below
`max_position_embeddings`; it then converts to `[B,Q] long`. Fractional,
three-axis or overflow positions fail before QKV.

- [ ] **Step 6: Add exhaustive state tests**

Cover:

- float32 three-axis fractional/non-monotonic positions, negative deltas and
  masked-zero round-trip without semantic weakening;
- valid legacy integral adapter and pre-compute rejection of fractional,
  three-axis and boundary overflow inputs;
- absent/image/audio/both prefix, explicit gaps, padding, `Q=1` and `Q=3`;
- `Dk != Dv`, mixed dtype/device, missing/extra layer and divergent mask;
- typed partition rejection and logical/allocated byte accounting.

- [ ] **Step 7: Run and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime/test_decoder_state.py -q

git add src/qwen3_omni_pretrain/runtime tests/runtime/test_decoder_state.py
git commit -m "feat: define request-owned decoder state"
```

---

### Task 2: Add full-attention KV and one rectangular mask owner

**Files:**

- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py`
- Create: `tests/runtime/test_attention_cache.py`
- Modify: `tests/test_tp_moe.py`
- Modify: `tests/test_multimodal_attention_mask.py`

**Produces:** standard/TP MHA `(hidden, present_kv)` contract while preserving
DeltaNet's existing no-cache tensor contract.

- [ ] **Step 1: Write failing attention parity and allocation tests**

For standard and TP fixtures compare full sequence with `Q=5` prefill followed
by `Q=1` and `Q=3` decode. Compute `max_abs` directly and require `<=1e-5`, then
compare greedy tokens. Add GQA, `Dk != Dv`, BF16 dtype and two-row padded cases.

- [ ] **Step 2: Migrate MHA and callers atomically**

Standard/TP MHA accepts current hidden, already validated current position,
`AttentionKV | None`, current bool mask, one final additive bias, and
`use_cache`; it returns `(hidden, present_kv)`.

Apply RoPE to current Q/K, append persistent unrepeated local K/V, then repeat
only for attention compute. When `use_cache=False`, return `present_kv=None`.

DeltaNet remains a single tensor when cache is disabled. Decoder layer/model
caller branches on the actual block implementation/type; it must never blindly
unpack a DeltaNet tensor. Commit MHA signature and all callers together so the
default implicit 3:1 hybrid no-cache logits/output keys remain unchanged.

- [ ] **Step 3: Move causal bias ownership to model boundary**

Replace current standard/TP square-mask construction with one helper per model
call. Given past mask `[B,P]` and current mask `[B,Q]`, create exactly one
`[B,1,Q,P+Q]` additive bias using the `P+q` causal rule. Invalid keys are
masked. A valid query with no valid key fails before attention; invalid current
queries are explicitly zeroed after output projection, remain false in cache
mask and do not advance valid count.

Attention consumes that final bias and never layers a second causal mask. Eager
and SDPA paths use identical semantics and query-compatible dtype/device.

- [ ] **Step 4: Prove allocation and failure order**

Spies must show:

- one rectangular bias allocation per entire multi-layer model call;
- no `[P+1,P+1]` allocation during `Q=1` decode;
- explicit and implicit DeltaNet cache requests fail before embedding, QKV,
  DeltaNet or collective calls;
- no-cache mixed DeltaNet results remain exact regressions.

- [ ] **Step 5: Add real TP local-shard parity**

Keep world-size-one smoke and add mandatory 2-rank CPU Gloo. Each rank stores
`Hkv/world_size` unrepeated local KV; owner/mask/position/seen are identical
across ranks. Compare gathered FP32 logits and greedy token to a standard model
with the same weights (`max_abs <=1e-5`). Synchronize any validation failure
before collectives so one-rank failure cannot hang peers.

- [ ] **Step 6: Run and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/runtime/test_attention_cache.py tests/test_tp_moe.py \
  tests/test_multimodal_attention_mask.py -q

git add \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py \
  tests/runtime/test_attention_cache.py tests/test_tp_moe.py \
  tests/test_multimodal_attention_mask.py
git commit -m "feat: cache full-attention key values"
```

---

### Task 3: Propagate atomic cache state through Thinker

**Files:**

- Modify: `modeling_thinker_text.py`
- Modify: `modeling_thinker_text_tp.py`
- Create: `tests/runtime/test_thinker_cache.py`

**Produces:** typed standard/TP `.prefill()` and `.decode()` adapters plus
atomic `DecoderState` candidates.

- [ ] **Step 1: Write failing model-level cached/uncached parity tests**

Use tiny all-MHA standard/TP models. Cover one- and three-token chunks,
different per-row valid counts, explicit position gaps, maximum boundary and
padded batch. Check all affected logits, direct `max_abs <=1e-5`, then tokens.

- [ ] **Step 2: Add a single pre-compute validation phase**

Before embedding or any layer:

- validate owner nonce, batch, device/dtype and `model.training is False`;
- reject every `use_cache=True and model.training` combination, regardless of
  labels or gradient checkpointing;
- scan every actual layer and reject DeltaNet (explicit or implicit default);
- require state KV layer keys to equal the model's exact cacheable layer set;
- require all KV lengths/masks/device/dtype and position history to agree;
- validate legacy position adapter and true maximum RoPE position;
- reject zero-valid rows and context/output-budget overflow before compute.

Missing layers are never interpreted as empty past; extra layers are never
ignored.

- [ ] **Step 3: Implement typed prefill/decode adapters**

Training-compatible `forward()` may retain legacy two-dimensional
`position_ids`, but engine-facing methods accept only `ModelPrefillInputs` and
`ModelDecodeInputs` with `PositionBatch`.

Prefill creates empty per-layer candidates, one position history and an owner.
Decode consumes only current token IDs/mask/position plus the owned state. It
does not infer position from `seen_tokens + cumsum`.

Legacy cursor advances storage positions, including fixed masked/absent media
slots and explicit gaps; `seen_tokens` advances only by current valid counts.
Qwen3-disjoint position can round-trip through the common protocol but is not
wired to this legacy-only model runtime.

Write every full-attention layer candidate first. Only after all layers
succeed, atomically create the new `DecoderPositionState` and advance seen
counts. A layer exception returns no partial state.

- [ ] **Step 4: Enforce detached inference output**

Engine calls model only inside `torch.inference_mode()`. Adapters clone-detach
candidate state before returning `CausalLMOutput`; all state tensors have no
grad, and no old/caller/candidate storage aliases.

- [ ] **Step 5: Add state and bounds tests**

Cover exact layer set, owner mismatch before attention, `Q=1/Q=3`, per-row
padding, context boundary success and overflow failure, candidate exception,
old-state fingerprint/pointer stability, cache-disabled legacy output keys,
mixed-DeltaNet no-cache regression and TP parity.

- [ ] **Step 6: Enable capability only after protocol works**

The actual-layer scanner may report `incremental_decode_state=True` only when
the concrete model implements the typed protocol and every layer is validated
full MHA. All public streaming/beam/truncate/speculation values remain false.

- [ ] **Step 7: Run and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/runtime/test_thinker_cache.py \
  tests/runtime/test_attention_cache.py \
  tests/test_multimodal_attention_mask.py -q

git add \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py \
  src/qwen3_omni_pretrain/runtime \
  tests/runtime/test_thinker_cache.py
git commit -m "feat: propagate decoder cache through Thinker"
```

---

### Task 4: Encode legacy media only during prefill

**Files:**

- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py`
- Create: `tests/runtime/test_multimodal_cache.py`
- Modify: `tests/test_multimodal_attention_mask.py`

**Produces:** strict `LegacyMediaPrefillInputs`, fixed two-slot processed prefix,
and raw-media-free decode.

- [ ] **Step 1: Write failing encoder call-count and phase tests**

Cover absent/image/audio/both plus a mixed two-row batch. Present encoder count
is exactly one per request; absent encoder count is zero. Supplied state, raw
decode media, partial state or union contradiction all fail with both encoder
counts still zero.

- [ ] **Step 2: Validate the raw media union before encoder work**

Only `decoder_state is None` is prefill. Any supplied state is decode.

`has_image`/`has_audio` must be same-device bool `[B]`. When any flag is true,
the corresponding raw floating tensor is required with matching B/device and
valid rank/dtype. When all flags are false, the corresponding tensor must be
`None`. Mixed rows index-select only present items, call the encoder once and
scatter results back into the fixed storage slot.

Direct legacy `forward()` may normalize the old 0/1 long flags and dummy absent
tensors for backward compatibility; the typed protocol never accepts them.

- [ ] **Step 3: Separate prefill and decode**

Prefill writes concrete `LegacyProcessedPrefix` with exactly two storage slots.
Its key-valid mask follows media presence, while `LegacyPositionCursor` advances
over both storage slots even if masked. Decode requires a complete prefix marker
and accepts only new text input; it cannot receive or re-encode raw media.

- [ ] **Step 4: Add parity and rejection tests**

For every modality combination compare uncached full input with cached prefill
plus decode (`max_abs <=1e-5`, token exact). Test wrong flag dtype/value,
tensor/flag contradiction, wrong batch/device/dtype, raw decode media,
empty/partial state, fixed slot positions and padding. All rejections precede
vision/audio/text encoder calls.

- [ ] **Step 5: Run and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/runtime/test_multimodal_cache.py \
  tests/test_multimodal_attention_mask.py -q

git add \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py \
  tests/runtime/test_multimodal_cache.py \
  tests/test_multimodal_attention_mask.py
git commit -m "feat: cache legacy multimodal prefill state"
```

---

### Task 5: Add a legacy-only pending-token greedy engine and CLI wiring

**Files:**

- Create: `src/qwen3_omni_pretrain/runtime/generation.py`
- Modify: `src/qwen3_omni_pretrain/runtime/__init__.py`
- Modify: `src/qwen3_omni_pretrain/cli_infer_thinker.py`
- Modify: `src/qwen3_omni_pretrain/architecture/summary.py`
- Modify: legacy profile factory/capability tests
- Create: `tests/runtime/test_generation_engine.py`
- Modify: `tests/test_stage2_inference.py`
- Modify: `tests/test_cli_profile.py`
- Modify: `README.md`

**Produces:** `GenerationRequest`, `GenerationCheckpoint`, internal
`GenerationStep`, `GenerationResult`, `CancellationToken`, and
`LegacyGreedyPrefillDecodeEngine`.

- [ ] **Step 1: Write failing pending-token and resume tests**

The single invariant is:

```text
checkpoint.decoder_state caches prompt + generated_ids[:-1]
if generated_ids is non-empty:
    checkpoint.pending_token_id == generated_ids[-1:]
else:
    checkpoint.pending_token_id is None
```

Generate at least three tokens and assert exactly one prefill plus
`generated_count - 1` decodes; each decode consumes only the previous pending
token. Test pause after every step, resume from the prior `GenerationResult`,
and exact parity with uninterrupted generation.

- [ ] **Step 2: Implement strict request/checkpoint values**

```python
@dataclass(frozen=True)
class GenerationRequest:
    display_request_id: str
    prefill_inputs: ModelPrefillInputs
    max_new_tokens: int
    eos_token_id: int | None
    num_beams: int = 1
    cancellation: CancellationToken | None = None


@dataclass(frozen=True)
class GenerationCheckpoint:
    decoder_state: DecoderState
    pending_token_id: torch.LongTensor | None


@dataclass(frozen=True)
class GenerationStep:
    emitted_token_id: torch.LongTensor
    emitted_index: int
    checkpoint: GenerationCheckpoint
    finished: torch.BoolTensor


@dataclass(frozen=True)
class GenerationResult:
    prompt_ids: torch.LongTensor
    generated_ids: torch.LongTensor
    checkpoint: GenerationCheckpoint
    finish_reason: Literal["eos", "length", "cancelled"]
    prefill_calls: int
    decode_calls: int
```

Strictly reject bool/non-integer/`<=0` length or beams and all
`num_beams != 1` before tokenizer/model/media. The engine requires batch one,
input IDs, at least one valid token and a valid final query slot; it rejects a
right-padded prompt instead of sampling `logits[:, -1]` from padding. Tokenizer
stays in the CLI adapter, not the engine.

Every fresh generation creates `StateOwner.fresh(display_request_id)`. Explicit
resume accepts a prior `GenerationResult`, reuses that exact owner and validates
the pending invariant; a fresh same-name session gets a different nonce.

- [ ] **Step 3: Implement candidate/commit ordering**

Prefill caches the prompt, samples the first emitted token and stores it as
pending without decoding it. Each later call consumes exactly the old pending
token, obtains a candidate state, samples the next pending token, then commits.

After every model call, check in this order:

1. model exception/result contract;
2. cancellation;
3. owner;
4. candidate commit and token emission.

Post-call cancellation wins over commit/emit. Without cancellation, EOS wins
over length; emitted EOS remains the pending last token. Attempt counters include
post-call-cancelled calls, while checkpoint state remains the previous committed
snapshot.

Pre-cancel performs zero model calls and returns empty generated IDs with an
owned empty checkpoint. In-flight cancel or exception exposes the latest
committed checkpoint, never a partial candidate. This plan has no iterator;
`GenerationStep` is an internal test value and `streaming_generation=false`.

- [ ] **Step 4: Wire both CLI paths by actual protocol/capability**

Update current Stage1 and Stage2 loops, not only a shared-looking helper. Add
`--num-beams` and `--allow-uncached-fallback`, and update every handwritten
`argparse.Namespace` fixture.

Selection uses concrete model protocol plus full actual-layer scan. It never
uses summary `cache_type` text:

- standard all-MHA and validated TP use the cache engine;
- explicit/implicit DeltaNet without fallback raises pre-compute
  `DELTA_NET_UNSUPPORTED`;
- with fallback, call the existing uncached loop and emit exactly one JSON
  warning per request containing stable `code`, `profile`, display ID, `reason`
  and `semantic_change: false`.

Stage2 media is passed only during prefill. A protocol fake must fail if decode
receives raw media, full token history or wrong owner.

- [ ] **Step 5: Publish truthful profile capability**

`incremental_decode_state` is true only when the built concrete model exposes
the protocol and the all-layer scan succeeds. `streaming_generation`,
`beam_search`, `state_truncate`, and `speculative_decode` remain false.

Fix legacy requested-capability handling: append to `unsupported` only when
`capabilities.get(name) is not True`, rather than unconditionally. Do not claim
Qwen3/Qwen3.5/MiMo native runtime support.

- [ ] **Step 6: Add engine, cancellation and CLI proof**

Cover prefill-EOS, `max_new_tokens=1`, EOS/length precedence, every-step resume,
same display/fresh nonce, pre-set/prefill-in-flight/decode-in-flight/post-call
cancellation, prefill/decode exception, snapshot fingerprints, context boundary,
Stage1/Stage2 cache invocation, media once, fallback flag on/off, exact-one
warning and implicit DeltaNet.

- [ ] **Step 7: Run and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/runtime/test_generation_engine.py \
  tests/test_stage2_inference.py tests/test_cli_profile.py -q

git add \
  src/qwen3_omni_pretrain/runtime \
  src/qwen3_omni_pretrain/cli_infer_thinker.py \
  src/qwen3_omni_pretrain/architecture/summary.py \
  tests/runtime/test_generation_engine.py \
  tests/test_stage2_inference.py tests/test_cli_profile.py README.md
git commit -m "feat: decode legacy requests with owned cache"
```

---

### Task 6: Add correctness-gated cache benchmark and completion gates

**Files:**

- Create: `scripts/benchmark_decode_cache.py`
- Create: `configs/model/legacy_full_attention_tiny.yaml`
- Create: `tests/runtime/test_benchmark_decode_cache.py`
- Modify: `README.md`

- [ ] **Step 1: Write a failing deterministic benchmark smoke test**

Run prompt length 8, output length 3, warmup 1 and repetitions 2. Validate the
complete JSON schema and require parity failure to happen before any timing.

- [ ] **Step 2: Implement benchmark truthfully**

Before timing, calculate:

```python
max_abs = (cached.float() - uncached.float()).abs().max().item()
assert max_abs <= 1e-5
assert torch.equal(cached_tokens, uncached_tokens)
```

JSON contains at least:

- architecture manifest and implementation commit;
- random seed and actual prompt/output lengths;
- dtype/device and synchronization method;
- warmup/repetition counts and raw latency samples;
- aggregate latency and tokens/second;
- peak memory;
- logical cache bytes and best-effort unique allocated bytes by partition;
- `max_abs`, token-exact flag, fallback status and stable fallback code.

Use a checked-in valid all-MHA tiny config. Do not silently rewrite a hybrid
config. Longer 128/512/2048/4096 runs are optional performance jobs; the tiny
smoke is mandatory CI.

- [ ] **Step 3: Run focused, distributed and full gates**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/test_stage2_inference.py \
  tests/test_cli_profile.py \
  tests/test_tp_moe.py \
  tests/test_multimodal_attention_mask.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m compileall -q src tests scripts

git diff --check
git status --short
```

The 2-rank CPU Gloo gate is mandatory whenever `torch.distributed` is present;
it cannot silently degrade to world size one.

- [ ] **Step 4: Completion checklist**

The plan is technically complete only when all are proven:

- typed `PositionBatch` prefill/state/decode round-trip and strict legacy
  position adapter;
- explicit/implicit DeltaNet fails before compute;
- nonce owner isolation across same display IDs;
- clone-detached non-aliasing state and cancellation/exception isolation;
- exact missing/extra cache-layer rejection;
- one model-level rectangular bias and no decode square allocation;
- pending-token checkpoint resume parity;
- `num_beams != 1` and generic truncate/speculation rejection;
- legacy media encoder once for present modalities and zero for absent ones;
- 2-rank TP local-shard parity;
- direct FP32 `max_abs <=1e-5` plus exact greedy token;
- benchmark JSON smoke, full tests, compileall and diff checks pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_decode_cache.py \
  configs/model/legacy_full_attention_tiny.yaml \
  tests/runtime/test_benchmark_decode_cache.py README.md
git commit -m "perf: gate incremental decode cache benchmark"
```
