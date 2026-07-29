# Decoder Cache and Streaming State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 full-attention 模型建立请求隔离、copy-on-write 的 prefill/decode cache，使 cached 与 uncached 输出一致，并确保媒体 encoder 每个请求只执行一次。

**Architecture:** 公共 `runtime.state` 定义所有 profile 可消费的 state partitions；当前计划只写入经过验证的 full-attention KV 和 processed-media partitions。legacy 标准/TP attention 增加可选 cache，但默认 forward 行为保持不变。当前简化 DeltaNet 没有正确 conv/recurrent cache，因此 cache 请求必须显式失败或由 CLI 记录后使用全量重算。

**Tech Stack:** Python 3.10, dataclasses, immutable mappings, PyTorch 2.10.0 SDPA, pytest, torch.profiler-compatible timing.

## Global Constraints

- 本计划依赖 profile/oracle 和 media-sequence 两个计划完成。
- `DecoderState` 属于单个 `request_id`；跨 session 复用必须失败。
- state 更新采用 copy-on-write；forward 不得原地修改传入 state。
- full-attention cache 保存未 repeat 的 KV，形状固定为 `[B,Hkv,S,D]`。
- K/V 只要求 batch、KV-head 和 token 轴一致，最后一维可分别为
  `Dk`/`Dv`，以支持 MiMo 的 QK/V 非对称 head。
- 每个 KV partition 保存 `[B,S]` key-valid mask；`seen_tokens` 是
  `[B]` 的有效 token 计数，不是把 padded batch 压成一个标量。
- cache position IDs 支持 `[B,S]` 一轴或 `[A,B,S]` 多轴；batch/token
  轴必须与 KV 一致。一轴位置必须在有效 token 上单调；多轴 TM-RoPE
  可合法重置空间轴或复用跨模态时间 ID，cache 只保存并验证 shape/dtype，
  语义由 `PositionBuilder` 测试。
- FP32 cached/uncached logits 的初始最大误差阈值为 `1e-5`，greedy token 必须完全一致。
- media encoder 只在 prefill 执行一次；decode 不得重新接收或编码原始媒体。
- SWA、Qwen3.5 GDN、Talker、MTP 和 codec state 只预留 partition；本计划不伪造其更新语义。
- 当前 legacy DeltaNet cache 不受支持；调用者要么 fail-fast，要么显式选择有记录的 uncached fallback。
- session 取消、异常和 speculative rejection 不能污染其他请求状态。
- 性能 benchmark 必须同时报告正确性、上下文、输出长度、dtype、device 和峰值内存。
- Snippet 中的 `...` 只表示 Protocol/签名节选；任务提交不得保留未实现
  stub，必须由该任务列出的行为测试证明可用。
- 每个行为变更使用 red-green-refactor，并形成独立提交。

---

## File responsibility map

- `src/qwen3_omni_pretrain/runtime/state.py`: immutable KV/state dataclasses、ownership 和字节统计。
- `src/qwen3_omni_pretrain/runtime/protocols.py`: cache-capable model 和 generation output protocols。
- `src/qwen3_omni_pretrain/runtime/generation.py`: framework-independent greedy prefill/decode engine。
- `modeling_thinker_text.py`: legacy standard full-attention cache 计算。
- `modeling_thinker_text_tp.py`: TP local-shard cache 计算和相同输出 contract。
- `modeling_thinker_vision_audio.py`: legacy media prefill 与 decode 分离。
- `cli_infer_thinker.py`: 薄 CLI；选择 cached engine 或记录明确 fallback。
- `scripts/benchmark_decode_cache.py`: correctness-gated latency/memory benchmark。

---

### Task 1: Define immutable decoder state partitions

**Files:**
- Create: `src/qwen3_omni_pretrain/runtime/__init__.py`
- Create: `src/qwen3_omni_pretrain/runtime/state.py`
- Create: `src/qwen3_omni_pretrain/runtime/protocols.py`
- Create: `tests/runtime/test_decoder_state.py`

**Interfaces:**
- Consumes: `MediaSequence`.
- Produces: `AttentionKV`, `SlidingWindowKV`, `DecoderState`,
  `CacheCapabilityError`, `LegacyMediaPrefillInputs`, `ModelPrefillInputs`,
  `CacheCapableModel` protocol, and `CausalLMOutput`.

- [ ] **Step 1: Write failing ownership and copy-on-write tests**

```python
def test_state_update_does_not_mutate_previous_state():
    empty = DecoderState.empty(request_id="r1")
    cache = AttentionKV(
        key=torch.zeros(1, 2, 3, 4),
        value=torch.ones(1, 2, 3, 4),
        position_ids=torch.arange(3).view(1, 3),
        key_valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    updated = empty.with_full_attention(layer_idx=0, cache=cache)
    assert dict(empty.full_attention_kv) == {}
    assert updated.full_attention_kv[0] is cache
    assert updated.request_id == "r1"


def test_state_rejects_cross_request_owner():
    state = DecoderState.empty(request_id="r1")
    with pytest.raises(ValueError, match="request_id"):
        state.assert_owner("r2")
```

- [ ] **Step 2: Run and verify the runtime package is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime/test_decoder_state.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement validated full and sliding KV types**

```python
@dataclass(frozen=True)
class AttentionKV:
    key: torch.Tensor
    value: torch.Tensor
    position_ids: torch.LongTensor
    key_valid_mask: torch.BoolTensor

    def __post_init__(self) -> None:
        if self.key.ndim != 4:
            raise ValueError("key must have shape [B, Hkv, S, Dk]")
        if self.value.ndim != 4:
            raise ValueError("value must have shape [B, Hkv, S, Dv]")
        if self.value.shape[:3] != self.key.shape[:3]:
            raise ValueError("key/value B, Hkv, and S axes must match")
        valid_position_shape = (
            self.position_ids.ndim == 2
            and self.position_ids.shape
            == (self.key.shape[0], self.key.shape[2])
        ) or (
            self.position_ids.ndim == 3
            and self.position_ids.shape[1:]
            == (self.key.shape[0], self.key.shape[2])
        )
        if not valid_position_shape:
            raise ValueError("position_ids must have shape [B,S] or [A,B,S]")
        if self.key_valid_mask.shape != (
            self.key.shape[0],
            self.key.shape[2],
        ):
            raise ValueError("key_valid_mask must have shape [B, S]")
        if self.key_valid_mask.dtype is not torch.bool:
            raise TypeError("key_valid_mask must be boolean")
        if self.position_ids.dtype != torch.long:
            raise TypeError("position_ids must be torch.long")
        valid_positions = (
            self.position_ids
            if self.position_ids.ndim == 2
            else self.position_ids.flatten(0, 1)
        )
        expanded_valid_mask = (
            self.key_valid_mask
            if self.position_ids.ndim == 2
            else self.key_valid_mask.repeat(self.position_ids.shape[0], 1)
        )
        if torch.any(valid_positions[expanded_valid_mask] < 0):
            raise ValueError("valid cached position IDs must be non-negative")
        valid_pairs = (
            self.key_valid_mask[:, 1:] & self.key_valid_mask[:, :-1]
        )
        if self.position_ids.ndim == 2 and torch.any(
            (
                self.position_ids[..., 1:]
                < self.position_ids[..., :-1]
            )
            & valid_pairs
        ):
            raise ValueError("valid cached position IDs must be monotonic")

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.LongTensor,
        key_valid_mask: torch.BoolTensor,
    ) -> "AttentionKV":
        if position_ids.ndim != self.position_ids.ndim:
            raise ValueError("position axis count cannot change while appending")
        return AttentionKV(
            key=torch.cat((self.key, key), dim=2),
            value=torch.cat((self.value, value), dim=2),
            position_ids=torch.cat(
                (self.position_ids, position_ids), dim=-1
            ),
            key_valid_mask=torch.cat(
                (self.key_valid_mask, key_valid_mask), dim=1
            ),
        )

    def storage_bytes(self) -> int:
        tensors = (
            self.key,
            self.value,
            self.position_ids,
            self.key_valid_mask,
        )
        return sum(t.numel() * t.element_size() for t in tensors)
```

For the one-axis case, the production validation compares consecutive valid
positions after compacting each row, so a padding gap cannot hide a decrease.
For `[A,B,S]`, validate only dtype, shape and non-negativity; do not impose
per-axis monotonicity because image width IDs reset and temporal IDs may be
reused by later spans.

Define `SlidingWindowKV` now with the same fields plus `window_size`. Its
`append()` compacts **each batch row independently by `key_valid_mask`**, keeps
that row's last `window_size` valid key/value/position entries, and then
right-pads the rectangular result back to
`[B, H_kv, max_kept, D]`/`[B, max_kept]`. Padding slots have a false mask and
cannot advance another row's window or evict valid KV. For multi-axis
positions, compaction applies the same row/token gather to every axis. This
type is tested here but first consumed by the MiMo experimental plan.

Add an irregular two-row test in which the short row contains right padding
in every one of several appended chunks. Compare the compacted valid
key/value/position sequence against a per-row Python oracle after every append
and prove the short row retains its last `window_size` **valid** tokens rather
than the last rectangular storage slots.

- [ ] **Step 4: Implement the common state**

```python
@dataclass(frozen=True)
class DecoderState:
    request_id: str
    seen_tokens: torch.LongTensor
    full_attention_kv: Mapping[int, AttentionKV]
    swa_kv: Mapping[int, SlidingWindowKV]
    gdn_convolution_state: Mapping[int, torch.Tensor]
    gdn_recurrent_matrix_state: Mapping[int, torch.Tensor]
    talker_state: object | None
    mtp_state: object | None
    codec_state: object | None
    processed_media_cache: Mapping[str, object]

    @classmethod
    def empty(
        cls,
        request_id: str,
        *,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> "DecoderState":
        if not request_id.strip():
            raise ValueError("request_id must be non-empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return cls(
            request_id=request_id,
            seen_tokens=torch.zeros(
                batch_size, dtype=torch.long, device=device
            ),
            full_attention_kv=MappingProxyType({}),
            swa_kv=MappingProxyType({}),
            gdn_convolution_state=MappingProxyType({}),
            gdn_recurrent_matrix_state=MappingProxyType({}),
            talker_state=None,
            mtp_state=None,
            codec_state=None,
            processed_media_cache=MappingProxyType({}),
        )
```

Add `assert_owner()`, `with_full_attention()`, `with_processed_media(key,
value: object)`, `with_seen_tokens()`, and `storage_bytes(partition=None)`.
Each method creates a new mapping and wraps it in `MappingProxyType`; it never
mutates tensors or input mappings. The object-valued partition is deliberate:
common profile prefill stores `MediaSequence`, while the legacy adapter stores
a frozen processed-prefix marker. Profile code must validate its own value
type when reading a key.

`DecoderState.__post_init__()` requires non-negative rank-1 long
`seen_tokens`, and every cache batch size must match it.
`with_seen_tokens(current_valid_counts)` validates an identically shaped
non-negative long tensor and returns a state whose counter is
`self.seen_tokens + current_valid_counts`; it never stores the caller's tensor
without cloning. Add a two-row `[3,1] -> [4,2]` test so padded batches cannot
regress to scalar accounting.

- [ ] **Step 5: Define output and model protocols**

```python
@dataclass(frozen=True)
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    ce_loss: torch.Tensor | None
    aux_loss: torch.Tensor | None
    decoder_state: DecoderState | None
    hidden_states: tuple[torch.Tensor, ...] | None


@dataclass(frozen=True)
class LegacyMediaPrefillInputs:
    pixel_values: torch.Tensor | None
    audio_values: torch.Tensor | None
    has_image: torch.Tensor | None
    has_audio: torch.Tensor | None


@dataclass(frozen=True)
class ModelPrefillInputs:
    input_ids: torch.LongTensor | None
    inputs_embeds: torch.Tensor | None
    attention_mask: torch.Tensor
    position_ids: torch.LongTensor | None
    media: LegacyMediaPrefillInputs | None = None


class CacheCapableModel(Protocol):
    def prefill(
        self,
        *,
        inputs: ModelPrefillInputs,
        request_id: str,
        use_cache: bool,
    ) -> CausalLMOutput:
        ...

    def decode(
        self,
        *,
        token_ids: torch.LongTensor,
        current_attention_mask: torch.BoolTensor,
        decoder_state: DecoderState,
        request_id: str,
    ) -> CausalLMOutput:
        ...
```

`ModelPrefillInputs` requires exactly one of `input_ids`/`inputs_embeds`, a
rank-2 key-valid mask and matching batch/token axes. It contains the only raw
legacy media union supported by this plan; common sequence-based profiles pass
already assembled embeddings and positions. `decode()` has no raw-media
parameter by construction. Concrete legacy models may keep `forward()` for
backward compatibility, but expose these two thin typed methods for the engine
instead of accepting arbitrary `**model_kwargs`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime/test_decoder_state.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/runtime tests/runtime/test_decoder_state.py
git commit -m "feat: define request-owned decoder state"
```

---

### Task 2: Add full-attention KV cache to standard and TP attention

**Files:**
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py:148-289`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py:55-226`
- Create: `tests/runtime/test_attention_cache.py`
- Modify: `tests/test_tp_moe.py`

**Interfaces:**
- Consumes: `AttentionKV`.
- Produces: standard and TP attention signature returning `(hidden_states, present_kv)`.

- [ ] **Step 1: Write a failing prefill/decode parity test**

```python
def test_full_attention_cached_logits_match_full_sequence():
    torch.manual_seed(7)
    attention = tiny_attention()
    hidden = torch.randn(1, 6, 8)
    positions = torch.arange(6).view(1, 6)
    full, _ = attention(
        hidden,
        position_ids=positions,
        use_cache=False,
    )
    prefill, state = attention(
        hidden[:, :5],
        position_ids=positions[:, :5],
        use_cache=True,
    )
    step, next_state = attention(
        hidden[:, 5:],
        position_ids=positions[:, 5:],
        past_key_value=state,
        use_cache=True,
    )
    torch.testing.assert_close(prefill, full[:, :5], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(step, full[:, 5:], atol=1e-5, rtol=1e-5)
    assert next_state.key.shape[2] == 6
```

- [ ] **Step 2: Run and observe the unexpected keyword failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime/test_attention_cache.py -q
```

Expected: FAIL because attention does not accept `past_key_value` or
`use_cache`.

- [ ] **Step 3: Change the attention contract**

Both standard and TP attention use:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    rotary_emb: RotaryEmbedding | None = None,
    past_key_value: AttentionKV | None = None,
    current_key_valid_mask: torch.BoolTensor | None = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, AttentionKV | None]:
    ...
```

Compute Q/K/V for current tokens. Apply RoPE to current Q/K. Build or append
the cache before `_repeat_kv`; persistent keys and values remain at KV-head
count. `current_key_valid_mask` defaults to all true for an unpadded current
chunk, is appended to `AttentionKV.key_valid_mask`, and masks invalid cached
and current keys. Use all valid cached K/V for attention and return a new
cache only when `use_cache=True`.

In this same task, update standard/TP decoder-layer and model internal call
sites to unpack `(attention_output, present_kv)`. With cache disabled they
discard `present_kv` and preserve the exact existing outer model output. Do
not commit the tuple-returning attention before its callers are migrated;
Task 3 only adds request-owned state propagation.

The decoder-layer contract established here is:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor,
    current_key_valid_mask: torch.BoolTensor,
    past_key_value: AttentionKV | None = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, AttentionKV | None]:
    ...
```

- [ ] **Step 4: Build the correct rectangular causal mask**

For `query_length=Q`, `past_length=P`, `key_length=P+Q`, a current query at
local index `q` may attend through key index `P+q`. Build `[1,1,Q,P+Q]`:

```python
query_positions = torch.arange(Q, device=device) + P
key_positions = torch.arange(P + Q, device=device)
blocked = key_positions[None, :] > query_positions[:, None]
causal = torch.zeros(Q, P + Q, device=device, dtype=dtype)
causal.masked_fill_(blocked, torch.finfo(dtype).min)
```

Combine with `past_key_value.key_valid_mask` plus the current key-valid mask.
Never materialize `[total,total]` during single-token decode. A valid query
with zero valid keys is an error; an invalid/padded query returns a zeroed
attention output and does not contribute to state token counts.

- [ ] **Step 5: Verify standard/TP world-size-one parity**

Add tests for:

- prefill `Q=5,P=0`;
- one-token decode `Q=1,P=5`;
- multi-token chunk decode `Q=3,P=5`;
- GQA cache keeps KV-head count;
- padded two-row batch, including cached key-mask preservation and different
  valid-token counts;
- BF16 dtype preservation;
- TP world size one matches standard output and state shapes.

The parity fixture checks every causal prefill token, not only the final token,
and a spy asserts that prefill/decode each construct exactly one rectangular
`[1,1,Q,P+Q]` mask. No outer model path may also add a causal mask.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/runtime/test_attention_cache.py tests/test_tp_moe.py \
  tests/test_multimodal_attention_mask.py -q
```

Expected: all selected tests pass with FP32 `max_abs <= 1e-5`.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py \
  tests/runtime/test_attention_cache.py tests/test_tp_moe.py
git commit -m "feat: cache full-attention key values"
```

---

### Task 3: Thread cache through decoder layers and the text model

**Files:**
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py:460-812`
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py:404-802`
- Create: `tests/runtime/test_thinker_cache.py`

**Interfaces:**
- Consumes: `DecoderState` and cached attention contract.
- Produces: optional `decoder_state` in the existing output mapping.

- [ ] **Step 1: Write a failing model-level parity test**

```python
def test_thinker_cached_decode_matches_uncached_logits():
    torch.manual_seed(11)
    model = tiny_full_attention_thinker().eval()
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8]])
    full = model(input_ids=tokens, labels=None)["logits"]
    prefill = model(
        input_ids=tokens[:, :5],
        labels=None,
        request_id="r1",
        use_cache=True,
    )
    step = model(
        input_ids=tokens[:, 5:],
        labels=None,
        decoder_state=prefill["decoder_state"],
        request_id="r1",
        use_cache=True,
    )
    torch.testing.assert_close(
        step["logits"][:, -1],
        full[:, -1],
        atol=1e-5,
        rtol=1e-5,
    )
```

- [ ] **Step 2: Run and observe missing model arguments**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime/test_thinker_cache.py -q
```

Expected: FAIL with unexpected `request_id`.

- [ ] **Step 3: Extend the model signature and propagate common state**

The decoder-layer cache signature and no-cache caller migration were completed
atomically in Task 2. Extend only the public model boundary here:

```python
def forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    labels: torch.LongTensor | None = None,
    output_hidden_states: bool = False,
    inputs_embeds: torch.Tensor | None = None,
    decoder_state: DecoderState | None = None,
    request_id: str | None = None,
    use_cache: bool = False,
):
    ...
```

When cache is requested:

- require non-empty `request_id`;
- create an empty state with the input batch size or assert existing ownership
  and matching batch size;
- require a rank-2 external attention mask; on cached calls it covers the
  complete cached-plus-current key length, and its cached prefix must exactly
  equal every layer cache's `key_valid_mask`;
- slice the current `[B,Q]` valid mask and, when `position_ids` is absent,
  infer each row's new positions from rank-1 `state.seen_tokens` plus its
  current valid-token cumulative sum;
- obtain each full-attention layer's cache by layer index;
- write returned cache into a new state;
- increment each row of `seen_tokens` by its current valid-token count;
- return new state as `output["decoder_state"]`.

Also implement the typed `CacheCapableModel.prefill()` and `.decode()` methods
from Task 1 as thin calls into this boundary. `prefill()` expands
`ModelPrefillInputs` once; `decode()` accepts only the new token/mask and owned
state. `decode()` first asserts all populated attention layers agree on their
cached `key_valid_mask`, appends `current_attention_mask`, and supplies that
complete mask to legacy `forward()`; callers never know media-prefix/cache
lengths. Both normalize the existing mapping into `CausalLMOutput`. The
legacy public `forward()` remains available for training and old callers, but
the generation engine never passes arbitrary keyword mappings to it.

- [ ] **Step 4: Fail before compute on unsupported recurrent layers**

If any layer is `deltanet` and `use_cache=True`, raise:

```text
CacheCapabilityError: legacy DeltaNet has no validated recurrent cache;
select a full-attention profile or use the explicit uncached fallback
```

Do not reuse the vector output as a recurrent matrix state. Gradient
checkpointing plus cache is invalid during training and must also fail early.

- [ ] **Step 5: Add state, mask, and training tests**

Cover:

- cached/uncached logits and greedy parity;
- chunks of size 1 and 3;
- `attention_mask` includes past and new tokens;
- wrong request ID fails before attention;
- input state remains byte-for-byte unchanged;
- cache disabled returns no state and preserves old output keys;
- cache with labels during training raises a clear error;
- unsupported DeltaNet cache raises exact capability error;
- TP world size one produces the same next token.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/runtime/test_thinker_cache.py \
  tests/runtime/test_attention_cache.py \
  tests/test_multimodal_attention_mask.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py \
  tests/runtime/test_thinker_cache.py
git commit -m "feat: propagate decoder cache through Thinker"
```

---

### Task 4: Encode legacy media only during prefill

**Files:**
- Modify: `src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py:68-123`
- Create: `tests/runtime/test_multimodal_cache.py`
- Modify: `tests/test_multimodal_attention_mask.py`

**Interfaces:**
- Consumes: `DecoderState.processed_media_cache` and cached Thinker.
- Produces: prefill-only legacy media prefix and token-only decode.

- [ ] **Step 1: Write a failing encoder-call-count test**

```python
def test_media_encoders_run_once_across_prefill_and_decode(monkeypatch):
    model = tiny_full_attention_multimodal_model().eval()
    vision_calls = count_forward_calls(monkeypatch, model.vision_encoder)
    audio_calls = count_forward_calls(monkeypatch, model.audio_encoder)
    prefill = model(
        input_ids=torch.tensor([[3, 4, 5]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        labels=None,
        pixel_values=torch.zeros(1, 3, 224, 224),
        audio_values=torch.zeros(1, 32_000),
        has_image=torch.tensor([1]),
        has_audio=torch.tensor([1]),
        request_id="r1",
        use_cache=True,
    )
    model(
        input_ids=torch.tensor([[6]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        labels=None,
        pixel_values=None,
        audio_values=None,
        has_image=None,
        has_audio=None,
        decoder_state=prefill["decoder_state"],
        request_id="r1",
        use_cache=True,
    )
    assert vision_calls.value == 1
    assert audio_calls.value == 1
```

- [ ] **Step 2: Run and observe required-media argument failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime/test_multimodal_cache.py -q
```

Expected: FAIL because decode still requires media tensors.

- [ ] **Step 3: Separate prefill and decode branches**

Add optional arguments:

```python
pixel_values: torch.Tensor | None = None
audio_values: torch.Tensor | None = None
has_image: torch.Tensor | None = None
has_audio: torch.Tensor | None = None
decoder_state: DecoderState | None = None
request_id: str | None = None
use_cache: bool = False
```

Rules:

- empty/no state is prefill and may encode media;
- non-empty state is decode and rejects newly supplied media;
- prefill writes a small immutable marker under
  `processed_media_cache["legacy-prefix"]`;
- decode passes only new text embedding to Thinker;
- key-side mask covers cached prefix plus all text tokens;
- labels are allowed only in uncached training.

The new typed `prefill()` reads raw tensors only from
`ModelPrefillInputs.media: LegacyMediaPrefillInputs`; `decode()` has no such
field and therefore cannot accidentally re-encode media. Keep direct
`forward()` coverage as a legacy regression, and add protocol-level tests used
by `GreedyPrefillDecodeEngine`.

- [ ] **Step 4: Test absent, image-only, audio-only, and both media**

For each case compare the last-token logits of:

- full uncached input;
- cached prefill plus one-token decode.

Assert FP32 max error `<=1e-5`, exact greedy token equality, one encoder call
per present modality, zero calls for absent modality, and state ownership.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/runtime/test_multimodal_cache.py \
  tests/test_multimodal_attention_mask.py -q
```

Expected: all selected tests pass.

```bash
git add \
  src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py \
  tests/runtime/test_multimodal_cache.py \
  tests/test_multimodal_attention_mask.py
git commit -m "feat: cache multimodal prefill state"
```

---

### Task 5: Use a correctness-gated prefill/decode engine in the CLI

**Files:**
- Create: `src/qwen3_omni_pretrain/runtime/generation.py`
- Modify: `src/qwen3_omni_pretrain/cli_infer_thinker.py:192-336`
- Create: `scripts/benchmark_decode_cache.py`
- Create: `configs/model/legacy_full_attention_tiny.yaml`
- Create: `tests/runtime/test_generation_engine.py`
- Modify: `tests/test_stage2_inference.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `CacheCapableModel`, tokenizer, typed `ModelPrefillInputs`, and
  `DecoderState`.
- Produces: `GenerationRequest`, `GenerationStep`, `GenerationResult`, and `GreedyPrefillDecodeEngine`.

- [ ] **Step 1: Write a failing engine parity test**

```python
def test_engine_matches_uncached_greedy_tokens():
    model = tiny_full_attention_thinker().eval()
    tokenizer = TinyTokenizer()
    request = GenerationRequest(
        request_id="r1",
        prefill_inputs=ModelPrefillInputs(
            input_ids=torch.tensor([[3, 4, 5]]),
            inputs_embeds=None,
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            position_ids=None,
        ),
        max_new_tokens=4,
        eos_token_id=2,
    )
    cached = GreedyPrefillDecodeEngine(model).generate(request)
    uncached = uncached_greedy(model, request)
    assert torch.equal(cached.generated_ids, uncached.generated_ids)
    assert cached.prefill_calls == 1
    assert cached.decode_calls <= 4
```

- [ ] **Step 2: Run and observe missing engine failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime/test_generation_engine.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement prefill and token-only decode**

```python
@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prefill_inputs: ModelPrefillInputs
    max_new_tokens: int
    eos_token_id: int | None
    cancellation: "CancellationToken | None" = None


@dataclass(frozen=True)
class GenerationStep:
    next_token_id: torch.LongTensor
    next_token_logits: torch.Tensor
    decoder_state: DecoderState
    finished: torch.BoolTensor


@dataclass(frozen=True)
class GenerationResult:
    prompt_ids: torch.LongTensor
    generated_ids: torch.LongTensor
    decoder_state: DecoderState
    finish_reason: Literal["eos", "length", "cancelled"]
    prefill_calls: int
    decode_calls: int


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class GreedyPrefillDecodeEngine:
    def prefill(self, request: GenerationRequest) -> GenerationStep:
        ...

    def decode_step(
        self,
        *,
        token_id: torch.LongTensor,
        attention_mask: torch.Tensor,
        state: DecoderState,
    ) -> GenerationStep:
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
```

Prefill sends the full prompt and media once. Every decode call sends only the
previous token and the returned state. The engine owns no global state; the
caller owns each returned immutable state.

Validate all request/result shapes, non-empty unique request ID, positive
length and ownership before model compute. This first greedy engine explicitly
requires `prefill_inputs.input_ids` and batch size one; callers with batched
prompts must submit independent requests. This makes the single
`finish_reason` unambiguous while the underlying cache remains batch-aware.
Check the request-local
`CancellationToken` before prefill and between committed decode steps. On
cancellation, return only committed tokens/state with
`finish_reason="cancelled"`; never mutate the input state or another request.
If cancellation is already set before prefill, return an empty
`generated_ids`, `prefill_calls=decode_calls=0`, and
`DecoderState.empty(request_id, batch_size=1, device=input_ids.device)`; a
result never carries a null state.
Add a two-request test that cancels one after its first decode while the other
reaches EOS unchanged.

- [ ] **Step 4: Add an explicit uncached fallback**

`CacheCapabilityError` is handled only when the CLI flag
`--allow-uncached-fallback` is present. Emit one JSON warning containing:

```json
{
  "cache_fallback": {
    "profile": "legacy_prototype",
    "reason": "legacy DeltaNet has no validated recurrent cache",
    "semantic_change": false
  }
}
```

Without the flag, fail before generation. The fallback calls the existing
uncached loop and never claims cache performance.

- [ ] **Step 5: Add the benchmark script**

Command:

```bash
PYTHONPATH=src .venv-prototype/bin/python \
  scripts/benchmark_decode_cache.py \
  --model-config configs/model/legacy_full_attention_tiny.yaml \
  --prompt-lengths 128 512 2048 4096 \
  --output-lengths 32 128 \
  --dtype float32 \
  --device cpu \
  --output results/cache-benchmark.json
```

Before timing, the script runs one cached/uncached parity case and aborts on
error above `1e-5` or token mismatch. JSON records manifest, commit, device,
dtype, lengths, prefill latency, decode tokens/s, peak memory, cache bytes, and
fallback status.

The checked-in benchmark config is a small valid legacy config with
`use_deltanet: false`, empty DeltaNet indices and full attention in every
layer. Do not use comment-only `configs/model/thinker_text.yaml`, which parses
to `None`, or silently override one of the existing hybrid configs.

- [ ] **Step 6: Run engine, CLI, and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/runtime/test_generation_engine.py \
  tests/test_stage2_inference.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/runtime/generation.py \
  src/qwen3_omni_pretrain/cli_infer_thinker.py \
  scripts/benchmark_decode_cache.py \
  configs/model/legacy_full_attention_tiny.yaml \
  tests/runtime/test_generation_engine.py \
  tests/test_stage2_inference.py README.md
git commit -m "feat: decode with request-owned cache"
```

---

## Plan completion gate

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/runtime -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m compileall -q src tests scripts

git diff --check
git status --short
```

The plan is complete only when:

- FP32 cached/uncached logits differ by at most `1e-5`;
- greedy tokens are identical;
- input state remains unchanged after every forward;
- request ownership and cancellation isolation are tested;
- media encoders execute once per request;
- single-token decode does not materialize a total-length square mask;
- unsupported DeltaNet cache is explicit and never mislabeled;
- benchmark output records cache bytes, correctness and fallback metadata;
- full prototype tests and compileall pass;
- all changes are intentional and committed.
