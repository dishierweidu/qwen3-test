# Media Sequences and TM-RoPE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可变长图像、视频和音频序列，按文本中的媒体 sentinel 原位替换 embedding，并生成 profile-specific 的 1D/TM-RoPE 位置。

**Architecture:** legacy 单 token wrapper 保留为回归基线；新 profile 使用独立的媒体类型、严格 loader、序列 encoder、`SequenceAssembler` 和 `PositionBuilder`。媒体 encoder 只产生带 mask/grid/timestamp 的 `MediaSequence`；assembler 先验证 sentinel/item 一一对应，再允许样本级 policy 联合展开相邻 AV 占位符；position builder 只消费 span metadata，三者互不修改 tokenizer 状态。

**Tech Stack:** Python 3.10, dataclasses, PyTorch 2.10.0, TorchVision 0.25.0, TorchAudio 2.10.0, Pillow, FFmpeg/ffprobe for optional video decoding, pytest.

## Global Constraints

- 本计划依赖 `2026-07-29-profile-contracts-and-qwen3-oracle.md` 全部完成。
- `MediaSequence.embeddings` 固定为 `[batch, media_tokens, hidden]`，`attention_mask` 固定为 `[batch, media_tokens]`。
- media token 数必须可变；缺失媒体不产生伪 summary token。
- 被引用但损坏的媒体默认抛出结构化错误；quarantine 必须显式并记录 sample、path、modality 和 error。
- 文本中的媒体 sentinel 必须与媒体 item 一一对应后才能调用扩展策略；
  数量或模态不匹配立即失败。样本级策略可以联合展开一组相邻 AV
  sentinel，但必须恰好消费同一组 source、保留逐 source span，并禁止静默
  丢弃或复制 item。
- 媒体位置的 labels 必须为 `-100`；文本 padding 和原始监督边界必须保持。
- Qwen3 disjoint-media TM-RoPE (`use_audio_in_video=False`) 使用 T/H/W
  三轴、`24/20/20` rotary section，并严格复刻 pinned processor/model 的
  有效 token `13 position IDs/second` 浮点映射；不把近似 80 ms 网格
  四舍五入后冒充官方数值。Full joint-AV parity 由后续 Qwen3
  reference-runtime facade/oracle 计划负责。
- 独立 experimental TM-RoPE 支持显式 80 ms/160 ms 量化；160 ms
  用于 Qwen3.5-inspired 实验，本计划不声明 Qwen3.5 checkpoint 兼容。
- tokenizer 是媒体 token 语义的唯一来源，禁止跨 profile 复用裸数字。
- loader、encoder、assembler 和 position builder 的 fallback 不得改变数值语义。
- legacy wrapper 的现有测试和行为必须继续通过。
- Snippet 中的 `...` 只表示 Protocol/签名节选；任务提交不得保留未实现
  stub，必须由该任务列出的行为测试证明可用。
- 每个行为变更使用 red-green-refactor，并形成独立提交。

## Locked contract decisions

- `sample_index` is always the dense post-quarantine batch row; original
  identity is carried separately.
- source pairing is by contiguous `(sample_index,item_index)`, never transport
  order; final spans retain the complete `MediaSource`.
- only the profile collator quarantines, and it removes a whole row; the strict
  loader never produces a fake tensor.
- the assembler receives `pad_token_id` explicitly and materializes media only
  from validated symbolic token references.
- the assembler, not an expansion policy, defines safe joint-placeholder
  groups from an explicit token-ID whitelist.
- pinned Qwen3 disjoint-media positions (`use_audio_in_video=False`) and
  experimental 80/160 ms timestamp positions are separate builders with
  separate compatibility claims. Full official joint-AV parity remains in the
  dedicated Qwen3 reference-runtime plan, which calls the pinned facade.
- fixed Qwen3 oracle vectors are offline and non-skippable; cached processor
  coverage is an additional integration test.

---

## File responsibility map

- `src/qwen3_omni_pretrain/multimodal/tokenization/schema.py`: profile-specific token string schema 和 tokenizer 解析。
- `src/qwen3_omni_pretrain/multimodal/modalities.py`: 无依赖的媒体模态枚举。
- `src/qwen3_omni_pretrain/multimodal/types.py`: media source、grid、sequence、span、assembled batch 和 position batch。
- `src/qwen3_omni_pretrain/multimodal/io.py`: 只做严格
  image/audio/video decoding，不做 quarantine 或神经网络编码。
- `src/qwen3_omni_pretrain/multimodal/encoders/vision.py`: 小型 patch image/video encoder。
- `src/qwen3_omni_pretrain/multimodal/encoders/audio.py`: 小型顺序保真的 audio window encoder。
- `src/qwen3_omni_pretrain/multimodal/sequence_assembler.py`: sentinel replacement、padding、labels 和 source span。
- `src/qwen3_omni_pretrain/multimodal/positions.py`: legacy 1D 和 configurable TM-RoPE position IDs。
- `src/qwen3_omni_pretrain/multimodal/prefill.py`: 协调 encoder、assembler、position builder，不执行 decoder。
- `src/qwen3_omni_pretrain/data/profile_collator.py`: 新 profile 的多 item 媒体 batch；不改变 `OmniStage2Collator`。

---

### Task 1: Make multimodal token schemas profile-specific

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/modalities.py`
- Create: `src/qwen3_omni_pretrain/multimodal/tokenization/schema.py`
- Modify: `src/qwen3_omni_pretrain/multimodal/tokenization/special_tokens.py:1-71`
- Create: `tests/multimodal/test_token_schema.py`
- Modify: `tests/test_special_tokens.py`

**Interfaces:**
- Consumes: `ArchitectureProfile`.
- Produces: `MediaModality`, `MultimodalTokenSchema`,
  `ResolvedMultimodalTokens`, `schema_for_profile(profile)`, and
  `resolve_token_schema(tokenizer, schema, vocab_size)`.

- [ ] **Step 1: Write failing schema isolation tests**

```python
def test_qwen_and_mimo_audio_strings_cannot_be_interchanged():
    qwen = schema_for_profile(ArchitectureProfile.QWEN3_OMNI_REFERENCE)
    mimo = schema_for_profile(ArchitectureProfile.MIMO_V25_EXPERIMENTAL)
    assert qwen.audio_pad == "<|audio_pad|>"
    assert mimo.audio_start == "<|mimo_audio_start|>"
    assert qwen.audio_start != mimo.audio_start


def test_resolution_is_atomic_when_one_token_is_missing():
    tokenizer = FakeTokenizer(
        {
            "<|image_pad|>": 10,
            "<|video_pad|>": 11,
            "<|audio_pad|>": 12,
        }
    )
    schema = MultimodalTokenSchema.qwen3()
    with pytest.raises(ValueError, match="audio_start"):
        resolve_token_schema(tokenizer, schema, vocab_size=32)
```

- [ ] **Step 2: Run and confirm the schema module is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_token_schema.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement typed schemas and resolved IDs**

```python
class MediaModality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass(frozen=True)
class MultimodalTokenSchema:
    image_pad: str
    video_pad: str
    audio_pad: str
    vision_start: str | None = None
    vision_end: str | None = None
    audio_start: str | None = None
    audio_end: str | None = None

    @classmethod
    def qwen3(cls) -> "MultimodalTokenSchema":
        return cls(
            image_pad="<|image_pad|>",
            video_pad="<|video_pad|>",
            audio_pad="<|audio_pad|>",
            vision_start="<|vision_start|>",
            vision_end="<|vision_end|>",
            audio_start="<|audio_start|>",
            audio_end="<|audio_end|>",
        )


@dataclass(frozen=True)
class ResolvedMultimodalTokens:
    image_pad: int
    video_pad: int
    audio_pad: int
    vision_start: int | None
    vision_end: int | None
    audio_start: int | None
    audio_end: int | None

    def sentinel_for(self, modality: "MediaModality") -> int:
        return {
            MediaModality.IMAGE: self.image_pad,
            MediaModality.VIDEO: self.video_pad,
            MediaModality.AUDIO: self.audio_pad,
        }[modality]
```

Define the MiMo schema from its own tokenizer strings. Define Qwen3.5 as an
explicit schema entry even when it currently shares some Qwen3 strings; do not
return the same object.

- [ ] **Step 4: Resolve all fields before mutating config**

`resolve_token_schema()` must:

1. read `tokenizer.get_vocab()`;
2. collect every missing required field;
3. reject duplicate resolved IDs;
4. verify each ID is in `[0, vocab_size)`;
5. return an immutable result.

Change `reconcile_multimodal_token_ids()` into a legacy adapter that calls this
function with the legacy/Qwen3 schema and only then writes legacy config fields.

- [ ] **Step 5: Run old and new token tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/multimodal/test_token_schema.py tests/test_special_tokens.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_omni_pretrain/multimodal/tokenization \
  src/qwen3_omni_pretrain/multimodal/modalities.py \
  tests/multimodal/test_token_schema.py tests/test_special_tokens.py
git commit -m "feat: isolate multimodal token schemas"
```

---

### Task 2: Define media and assembled-sequence contracts

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/types.py`
- Modify: `src/qwen3_omni_pretrain/multimodal/__init__.py`
- Create: `tests/multimodal/test_media_types.py`

**Interfaces:**
- Consumes: `MediaModality` from `multimodal/modalities.py`.
- Produces: `MediaSource`, `MediaGrid`, `MediaSequence`, `SequenceSpanKind`,
  `SequenceSpan`, `AssembledSequence`, and `PositionBatch`.

- [ ] **Step 1: Write failing shape and timestamp validation tests**

```python
def test_media_sequence_validates_batch_and_token_axes():
    sequence = MediaSequence(
        embeddings=torch.zeros(2, 3, 8),
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
        modality=MediaModality.AUDIO,
        sources=(
            MediaSource(0, 0, "a0"),
            MediaSource(1, 0, "a1"),
        ),
        timestamps=torch.tensor(
            [[0.0, 0.08, 0.16], [0.0, 0.08, 0.16]]
        ),
    )
    sequence.validate()


def test_media_sequence_rejects_non_monotonic_valid_timestamps():
    sequence = make_audio_sequence(
        timestamps=torch.tensor([[0.0, 0.16, 0.08]])
    )
    with pytest.raises(ValueError, match="monotonic"):
        sequence.validate()
```

- [ ] **Step 2: Run and verify the type module is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_media_types.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement exact dataclasses**

```python
@dataclass(frozen=True)
class MediaSource:
    sample_index: int
    item_index: int
    source_id: str


@dataclass(frozen=True)
class MediaGrid:
    temporal: int
    height: int
    width: int

    @property
    def token_count(self) -> int:
        return self.temporal * self.height * self.width


@dataclass(frozen=True)
class MediaSequence:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    modality: MediaModality
    sources: tuple[MediaSource, ...]
    grid: tuple[MediaGrid | None, ...] | None = None
    timestamps: torch.Tensor | None = None
    seconds_per_grid: tuple[float | None, ...] | None = None

    def validate(self) -> None:
        if self.embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [B, M, H]")
        if self.attention_mask.shape != self.embeddings.shape[:2]:
            raise ValueError("attention_mask must have shape [B, M]")
        if len(self.sources) != self.embeddings.shape[0]:
            raise ValueError("one MediaSource is required per batch row")
```

Complete `validate()` with:

- `MediaSource` rejects bool/non-integer/negative indices and empty IDs in
  `__post_init__`; `MediaGrid` rejects bool/non-integer/non-positive axes in
  `__post_init__`;
- floating non-complex embeddings only;
- boolean/integer masks only, with every value exactly `0` or `1`;
- embeddings, mask and optional timestamps on one device;
- grid tuple length equals batch;
- grid token count equals valid tokens for image/video rows;
- seconds-per-grid tuple length equals batch; present values reject booleans,
  NaN/Inf and non-positive numbers, and only VIDEO may carry a non-`None`
  value;
- timestamps shape `[B, M]`;
- valid timestamps finite, non-negative and monotonic;
- `source_id` non-empty and `(sample_index, item_index)` non-negative;
- `(sample_index, item_index)` unique batch-wide and
  `(sample_index, source_id)` unique within one sample. `source_id` is
  sample-local, not required to be batch-global.

- [ ] **Step 4: Define output contracts**

```python
class SequenceSpanKind(str, Enum):
    TEXT = "text"
    MEDIA = "media"
    TIMESTAMP = "timestamp"


@dataclass(frozen=True)
class SequenceSpan:
    sample_index: int
    start: int
    end: int
    kind: SequenceSpanKind
    modality: MediaModality | None
    grid: MediaGrid | None
    timestamps: torch.Tensor | None
    seconds_per_grid: float | None
    source: MediaSource | None
    source_token_indices: tuple[int, ...] | None


@dataclass(frozen=True)
class AssembledSequence:
    expanded_input_ids: torch.LongTensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    spans: tuple[SequenceSpan, ...]


@dataclass(frozen=True)
class PositionBatch:
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    axis_names: tuple[str, ...]
```

Add validation methods for expanded IDs `[B,S]`, embeddings `[B,S,H]`,
mask/label shape, non-overlapping spans, full coverage of valid tokens, and
position shape `[axes,B,S]`, `rope_deltas.shape == [B,1]`, matching numeric
dtypes and finite values. Valid `position_ids` are non-negative;
`rope_deltas` may be negative (the pinned image oracle returns `-2`).
Position tensors may be integer or floating point: the exact Qwen3 oracle emits
`float32` and may produce fractional temporal IDs, while the legacy builder
emits integer IDs. `SequenceSpanKind.MEDIA` requires a complete `MediaSource`
and `source_token_indices` with one original row-major token index per output
token. `SequenceSpanKind.TIMESTAMP` requires a complete source and modality,
and requires grid, timestamps, seconds-per-grid and source-token indices all
to be `None`. During `AssembledSequence.validate()`, every timestamp source
must correspond to a MEDIA source in that assembled sample and its modality
must exactly match that source's canonical MEDIA modality; timestamp spans may
appear before the matching media span, so this check occurs after collecting
all spans. Text spans have no
source/modality/grid/timestamps/seconds-per-grid/indices.
Video media spans preserve explicit `seconds_per_grid` when supplied by their
source; the Qwen3 disjoint builder requires it instead of guessing cadence from
global timestamps. Other span kinds require it to be `None`.
For a media span, indices must be non-negative, have `end-start` entries, be
strictly increasing within that source fragment, and be below the **complete
source** grid/token count. `SequenceSpan.grid` always carries the complete
source grid, never a fragment/slice grid; its token count therefore need not
equal `end-start`. The assembler additionally validates that concatenating all
media-span indices for one source in assembled order yields exactly
`0..valid_token_count-1`; cross-source interleaving is allowed, intra-source
reordering or fragmentation that changes row-major order is not.
`PositionBatch` must not require each axis to be monotonic: image width IDs
naturally reset per row and later timestamped spans may reuse a temporal ID.
Only the one-axis legacy builder applies monotonic sequence validation.
Use `PositionBatch.validate(attention_mask)` so validity/padding is explicit;
it requires mask shape `[B,S]`, unique axis names matching the axis count,
numeric non-complex tensors, a binary mask, all tensors on one device, and zero
positions at masked slots. `AssembledSequence` requires non-negative long IDs,
floating non-complex embeddings, optional long labels, binary masks, all
coupled tensors/span timestamps on one device, and spans in canonical
`(sample_index,start)` order.

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_media_types.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_omni_pretrain/multimodal/types.py \
  src/qwen3_omni_pretrain/multimodal/__init__.py \
  tests/multimodal/test_media_types.py
git commit -m "feat: define media sequence contracts"
```

---

### Task 3: Add strict multi-item image, audio, and video loading

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/io.py`
- Create: `src/qwen3_omni_pretrain/data/profile_collator.py`
- Modify: `src/qwen3_omni_pretrain/data/collators.py:1-116`
- Create: `tests/multimodal/test_media_io.py`
- Create: `tests/multimodal/test_profile_collator.py`

**Interfaces:**
- Consumes: existing `MediaLoadError`, `MediaModality`, token schema.
- Produces: strict-only `DecodedMedia`,
  `StrictMediaLoader.load(item)`, `ProfileStage2Collator`, and a variable-item
  `media` batch field. Only the collator owns the explicit
  `quarantine_bad_samples` policy; the loader never returns fake/absent media.

- [ ] **Step 1: Write failing length-preservation and error tests**

```python
def test_audio_loader_preserves_true_length(tmp_path):
    path = write_wave(tmp_path / "tone.wav", samples=12_345, rate=16_000)
    item = StrictMediaLoader().load(
        MediaRequest(
            sample_id="s0",
            sample_index=0,
            item_index=0,
            source_id="a0",
            modality=MediaModality.AUDIO,
            path=str(path),
        )
    )
    assert item.tensor.shape == (1, 12_345)
    assert item.length == 12_345


def test_referenced_video_without_decoder_is_fatal(monkeypatch):
    monkeypatch.setattr(media_io, "_ffmpeg_available", lambda: False)
    with pytest.raises(MediaLoadError, match="video"):
        StrictMediaLoader().load(video_request("missing.mp4"))
```

- [ ] **Step 2: Run and observe missing loader failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/multimodal/test_media_io.py \
  tests/multimodal/test_profile_collator.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement decoded media without silent padding**

```python
@dataclass(frozen=True)
class MediaRequest:
    sample_id: str
    sample_index: int
    item_index: int
    source_id: str
    modality: MediaModality
    path: str
    original_sample_index: int | None = None
    timeline_offset_seconds: float = 0.0
    timestamps: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DecodedMedia:
    request: MediaRequest
    tensor: torch.Tensor
    length: int
    timestamps: torch.Tensor | None
    seconds_per_grid: float | None
    metadata: Mapping[str, object]
```

Rules:

- `sample_index` always means the dense row in the final retained batch.
  `original_sample_index`/`sample_id` preserve pre-quarantine identity.
  `item_index` is the zero-based canonical media-list order and must be
  contiguous within a sample;
- the collator uses two phases: tokenize/truncate and stage/decode each
  original row first, decide the complete retained-row set, then construct
  final frozen requests (or use `dataclasses.replace`) with dense
  `sample_index`. Final indices and `item_index` are preserved through
  `DecodedMedia` and copied exactly into encoder-produced `MediaSource`;
- `timeline_offset_seconds` is finite/non-negative and defines the item's
  origin on the sample-global AV timeline; encoder token timestamps are
  `timeline_offset_seconds + local_time`;
- image returns RGB float `[3,H,W]` and original dimensions in metadata;
- audio returns mono float `[1,samples]`, configured sample rate, true length,
  and no right padding;
- video returns `[frames,3,H,W]` plus exact per-frame seconds. It sets explicit
  `seconds_per_grid` only when at least two decoded frame times have one
  finite positive uniform cadence within `rtol=1e-6, atol=1e-6`; otherwise the
  field is `None`. Image/audio set it to `None`;
- decoded values form a strict encoder boundary:
  - image tensors are floating `[3,H,W]` with positive `H,W`, `length == 1`,
    no timestamps, and positive integer `original_height`/`original_width`
    metadata;
  - audio tensors are floating `[1,N]` with positive `N`, `length == N`, no
    timestamps, and a positive integer output `sample_rate` in metadata;
  - video tensors are floating `[F,3,H,W]` with positive `F,H,W`,
    `length == F`, timestamps shaped `[F]`, and positive integer decoded
    width/height metadata;
- `ffprobe` discovers dimensions/time base and `ffmpeg` decodes raw RGB frames;
- subprocess arguments are a list with `shell=False`;
- missing binaries, non-zero status, empty output and timestamp/frame mismatch
  become `MediaLoadError`;
- `StrictMediaLoader.load()` always either returns `DecodedMedia` or raises
  `MediaLoadError`. It has no quarantine mode and never reuses legacy
  `skip_bad_media`;
- when `ProfileStage2Collator(quarantine_bad_samples=True)` catches such an
  error it records it and removes the entire sample atomically. If any
  referenced item in a row fails, remove that row from input IDs, masks,
  labels, `_sample_ids`, and every decoded-media collection; never leave its
  sentinel in a retained text row. With
  `quarantine_bad_samples=False` (the default), propagate the first structured
  error. If all rows are quarantined, reuse the existing
  keyword-only structured constructor before model forward:

  ```python
  raise MediaLoadError(
      modality="batch",
      path="<multiple>",
      sample_id="<all-quarantined>",
      cause=AllSamplesQuarantinedError(quarantined_count),
  )
  ```

  `AllSamplesQuarantinedError` is a small local cause type, not a replacement
  public media-error schema. Tests assert the resulting `to_dict()` fields and
  count-bearing cause while proving no raw source path is copied into this
  aggregate error.

- [ ] **Step 4: Implement the new profile collator**

Input example schema:

```json
{
  "id": "sample-1",
  "input_text": "describe <|image_pad|> then <|audio_pad|>",
  "target_text": "answer",
  "media": [
    {"id": "image-0", "modality": "image", "path": "image.png",
     "timeline_offset_seconds": 0.0},
    {"id": "audio-0", "modality": "audio", "path": "audio.wav",
     "timeline_offset_seconds": 0.0}
  ]
}
```

`ProfileStage2Collator.__call__()` returns text tensors plus:

```python
{
    "decoded_media": tuple[DecodedMedia, ...],
    "_sample_ids": list[str],
    "_media_errors": list[dict[str, str]],
}
```

It resolves sentinel strings with the selected schema, requires sentinel/item
order to match, and reserves target supervision before prompt truncation.
Sentinel/item validation runs again against the final truncated token sequence;
if truncation removes or changes a pair, the row fails (or is quarantined) as
a whole. It may not pass an extra decoded item or unmatched sentinel to the
assembler.
Backward-compatible `image_path`/`audio_path` remains only in
`OmniStage2Collator`.

- [ ] **Step 5: Test multiple items, omission, corruption, and video**

Cover:

- two images and two audio items in one sample;
- two batch rows preserve distinct `(sample_index, item_index)` pairs from
  collator through decoded items;
- omitted optional media with no sentinel;
- sentinel with no item and item with no sentinel;
- corrupt referenced media in strict/quarantine modes;
- a mixed good/bad two-row batch with the bad row first keeps the good row with
  all tensors and media reindexed to `sample_index=0`, drops the bad row
  atomically, and records the original bad sample ID/path/modality/error;
- a three-row batch with the middle row bad produces dense indices `0,1`;
- corruption of the second item removes the whole row, final truncation that
  deletes a sentinel cannot leave its item behind, and all-quarantined batches
  raise the aggregate structured error without leaking raw paths;
- video frame timestamps from a 3-frame synthetic clip;
- no tensor truncation before the encoder.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/multimodal/test_media_io.py \
  tests/multimodal/test_profile_collator.py \
  tests/test_stage2_collator.py -q
```

Expected: all tests pass; FFmpeg-dependent tests skip only when either
`ffmpeg` or `ffprobe` is absent.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_omni_pretrain/multimodal/io.py \
  src/qwen3_omni_pretrain/data/profile_collator.py \
  src/qwen3_omni_pretrain/data/collators.py \
  tests/multimodal/test_media_io.py \
  tests/multimodal/test_profile_collator.py
git commit -m "feat: load variable-length multimodal items"
```

---

### Task 4: Encode spatial and temporal sequences instead of summaries

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/encoders/__init__.py`
- Create: `src/qwen3_omni_pretrain/multimodal/encoders/vision.py`
- Create: `src/qwen3_omni_pretrain/multimodal/encoders/audio.py`
- Create: `tests/multimodal/test_sequence_encoders.py`

**Interfaces:**
- Consumes: `DecodedMedia`, `MediaSequence`, `MediaGrid`, `MediaSource`.
- Produces: `PatchVisionEncoder.forward(items) -> MediaSequence`,
  `TemporalVideoEncoder.forward(items, patch_encoder=...) -> MediaSequence`,
  and `AudioWindowEncoder.forward(items) -> MediaSequence`.

Every public `forward()` accepts a non-empty homogeneous
`Sequence[DecodedMedia]`; Task 7 owns modality grouping. Wrong modality, shape,
non-floating dtype, zero length, `length`/shape disagreement, or video
timestamp/frame disagreement fails before projection. Sources remain in input
order and copy `(sample_index,item_index,source_id)` exactly. Public helpers
use these signatures:

```python
class PatchVisionEncoder(nn.Module):
    def forward(
        self,
        items: Sequence[DecodedMedia],
    ) -> MediaSequence: ...

    def from_tensor_batch(
        self,
        pixels: torch.Tensor,
        *,
        sources: tuple[MediaSource, ...],
    ) -> MediaSequence: ...


class TemporalVideoEncoder(nn.Module):
    def __init__(self, *, hidden_size: int) -> None: ...

    def forward(
        self,
        items: Sequence[DecodedMedia],
        *,
        patch_encoder: PatchVisionEncoder,
    ) -> MediaSequence: ...


class AudioWindowEncoder(nn.Module):
    def forward(
        self,
        items: Sequence[DecodedMedia],
    ) -> MediaSequence: ...

    def from_waveforms(
        self,
        waveforms: Sequence[torch.Tensor],
        *,
        sources: tuple[MediaSource, ...],
        timeline_offsets_seconds: Sequence[float] | None = None,
    ) -> MediaSequence: ...
```

All helpers require non-empty inputs and exact source counts.
`from_tensor_batch` accepts only floating `[B,C,H,W]`.
`from_waveforms` accepts only non-empty floating 1-D tensors;
`timeline_offsets_seconds=None` means one zero offset per waveform, while an
explicit sequence must match the waveform count and contain only finite,
non-negative values. Constructors reject booleans/non-integers and
non-positive dimensions; audio additionally requires
`hop_size <= window_size`. Inputs are converted to the owning module
parameter's device/dtype, while timestamps stay FP32 on that device. Every
public result is validated with `MediaSequence.validate()` before return.

- [ ] **Step 1: Write failing sequence-order tests**

```python
def test_patch_encoder_emits_one_token_per_patch():
    encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=8,
        patch_size=2,
    )
    image = torch.arange(3 * 4 * 4, dtype=torch.float32).view(1, 3, 4, 4)
    sequence = encoder.from_tensor_batch(
        image,
        sources=(MediaSource(0, 0, "image-0"),),
    )
    assert sequence.embeddings.shape == (1, 4, 8)
    assert sequence.grid == (MediaGrid(1, 2, 2),)


def test_audio_window_order_changes_when_waveform_is_reversed():
    encoder = AudioWindowEncoder(
        hidden_size=8,
        window_size=4,
        hop_size=4,
        sample_rate=16,
    )
    sources = (MediaSource(0, 0, "audio-0"),)
    forward = encoder.from_waveforms(
        [torch.arange(16).float()],
        sources=sources,
    )
    reverse = encoder.from_waveforms(
        [torch.arange(16).float().flip(0)],
        sources=sources,
    )
    assert not torch.equal(forward.embeddings, reverse.embeddings)
    assert torch.all(forward.timestamps[:, 1:] > forward.timestamps[:, :-1])
```

- [ ] **Step 2: Run and observe missing encoders**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_sequence_encoders.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement patch vision encoding**

Core module:

```python
class PatchVisionEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(pixels)
        return self.norm(features.flatten(2).transpose(1, 2))
```

Pad images only to the next patch multiple and preserve row-major patch order.
Use a ceil grid: every boundary patch containing at least one real pixel is a
valid token. Bottom/right-pad and encode each image independently, flatten
only that source's valid ceil-grid tokens into a contiguous prefix, and only
then right-pad token rows to the batch maximum with zero embeddings and a
boolean prefix mask. Do not spatially pad all sources to a common batch size
before flattening.

`TemporalVideoEncoder` is a separate `nn.Module` with exactly
`temporal_proj = nn.Linear(hidden_size, hidden_size, bias=False)` and
`temporal_norm = nn.LayerNorm(hidden_size)`. Its `forward()` receives the
shared `PatchVisionEncoder` explicitly, never stores it as an attribute, and
requires matching hidden sizes. It applies the patch encoder frame-by-frame,
then exactly
`temporal_norm(temporal_proj(features))`, independently per source. It
flattens temporal-major then row-major order, right-pads only flattened token
rows, and expands each global frame timestamp
(`request.timeline_offset_seconds + local_frame_seconds`) across its spatial
patches and copies decoded `seconds_per_grid` into the corresponding
`MediaSequence` row. Image/video public helpers construct `MediaSource` from
each `DecodedMedia.request` by copying its three shared fields. Consequently
one registered patch encoder and every video-specific parameter appear exactly
once in `state_dict()` and `named_parameters()`. Images use exact grid tuples,
`timestamps=None`, and `seconds_per_grid=None`; videos use exact grid tuples,
FP32 timestamps with zero-filled masked slots, and copy the per-item
`seconds_per_grid` tuple including `None`.

- [ ] **Step 4: Implement audio windows**

```python
class AudioWindowEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        window_size: int,
        hop_size: int,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.proj = nn.Linear(window_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        window_count = max(
            1,
            1
            + (
                max(waveform.numel() - self.window_size, 0)
                + self.hop_size
                - 1
            )
            // self.hop_size,
        )
        required_length = (
            (window_count - 1) * self.hop_size + self.window_size
        )
        waveform = F.pad(
            waveform,
            (0, required_length - waveform.numel()),
        )
        windows = waveform.unfold(0, self.window_size, self.hop_size)
        return self.norm(self.proj(windows))
```

Right-pad only the final partial window. A partial window containing any real
sample is one valid token; starts stop at the earliest window that covers the
tail, using
`max(1, 1 + ceil(max(N-window_size, 0) / hop_size))`. This yields starts
`(0)`, `(0)`, `(0,2)`, `(0,2,4)`, and `(0,2,4,6)` respectively for
`(N,window,hop)` values `(2,4,2)`, `(4,4,2)`, `(5,4,2)`, `(10,6,2)`,
and `(11,6,2)`. Only batch-alignment slots after that count are masked.
Timestamp each token at
`request.timeline_offset_seconds + window_start / sample_rate`. Empty waveform
is an error; omitted media never calls the encoder. Public batch helpers require
one `MediaSource` per waveform and copy those sources unchanged into the
returned `MediaSequence`. `forward(items)` requires the positive integer
`item.metadata["sample_rate"]` to equal the encoder sample rate; it never
infers rate from path or tensor length. Audio uses `grid=None`,
`seconds_per_grid=None`, a boolean prefix mask, and FP32 timestamps with
zero-filled masked slots.

- [ ] **Step 5: Add gradient and ordering tests**

Assert:

- image quadrant swaps change the corresponding patch tokens;
- video frame swaps change temporal order;
- audio tone-order swaps change token order;
- deterministic weights prove exact image row-major, video frame-block, and
  audio-window permutations rather than only unequal random outputs;
- non-divisible image sizes/audio lengths use ceil token counts, boundary
  patches/windows are valid, and only batch-alignment padding is masked;
- height-only and width-only patch remainders both produce valid boundary
  tokens, and changing a real boundary pixel changes its boundary-patch token;
- two differently sized images and two differently sized videos are encoded
  independently and only token rows are right-padded; valid masks contain no
  holes;
- audio window counts match the literal `(N,window,hop)` cases above, and a
  sample-rate mismatch fails explicitly;
- two unequal audio lengths have exact boolean prefix masks and zero timestamps
  in masked slots; changing the final real tail sample changes the final
  partial-window token;
- image, video and audio paths preserve the full `MediaSource`; audio/video
  fixtures with non-zero timeline offsets produce global timestamps;
- multiple items from one sample and items from multiple text rows preserve
  source identity exactly;
- a two-frame `2x2`-patch video with local timestamps `(0.0,0.25)` and offset
  `1.5` produces the exact valid timestamps
  `[1.5,1.5,1.5,1.5,1.75,1.75,1.75,1.75]`; audio starts `(0,4,8)` at 16 Hz
  and offset 2.0 produce `[2.0,2.25,2.5]`;
- uniform video fixtures preserve exact `seconds_per_grid`; non-uniform
  timestamps leave it absent; Task 6 owns the Qwen3-disjoint rejection test;
- outputs and gradients remain finite in FP32/BF16;
- ordinary FP32 decoded tensors work after each module is moved to BF16, with
  finite parameter gradients;
- empty input, wrong modality/channel/shape/dtype, source-count mismatch,
  duplicate sources, length disagreement, and timestamp/frame disagreement
  fail at the encoder boundary;
- missing video timestamps and 2-D/stereo `from_waveforms` inputs fail
  explicitly;
- a holder registering one patch encoder and one video encoder contains patch
  parameters exactly once; video loss reaches both shared patch and
  video-specific parameters without a registered patch alias;
- no encoder collapses valid input to one token unless the input genuinely
  contains one patch/window.

For every variable-length timestamp assertion, select valid slots through the
attention mask; padded slots are required to be zero but are not part of the
monotonicity assertion.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_sequence_encoders.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/multimodal/encoders \
  tests/multimodal/test_sequence_encoders.py
git commit -m "feat: preserve media token sequences"
```

---

### Task 5: Replace media sentinels with encoded sequences

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/time_quantization.py`
- Create: `src/qwen3_omni_pretrain/multimodal/sequence_assembler.py`
- Create: `tests/multimodal/test_sequence_assembler.py`

**Interfaces:**
- Consumes: text embeddings, masks, labels, resolved token schema, optional
  sample-level `MediaExpansionPolicy`, an optional non-owning
  `EmbeddingLookup`, an explicit text `pad_token_id`, a trusted
  `joint_separator_token_ids` whitelist, and `Sequence[MediaSequence]`.
- Produces: `MediaPlaceholder`, `MediaExpansionGroup`, `MediaTokenRef`,
  `ExpansionToken`, `ExpandedMediaRow`, `ExpandedMediaSample`,
  `MediaExpansionPolicy`, `IdentityMediaExpansion`,
  `TimestampInterleaveExpansion`, `AssembledLengthError`, shared
  `quantize_timestamps_half_up(...)`, and
  `SequenceAssembler.assemble(...) -> AssembledSequence`.

- [ ] **Step 1: Write a failing exact replacement test**

```python
def test_assembler_replaces_sentinel_and_masks_media_labels():
    input_ids = torch.tensor([[5, 10, 6, 0]])
    text_embeds = torch.arange(4 * 3).view(1, 4, 3).float()
    media = make_image_sequence(
        embeddings=torch.tensor([[[100.0] * 3, [200.0] * 3]]),
        source=MediaSource(0, 0, "image-0"),
    )
    result = SequenceAssembler().assemble(
        input_ids=input_ids,
        text_embeddings=text_embeds,
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        labels=torch.tensor([[-100, -100, 6, -100]]),
        media_sequences=(media,),
        tokens=resolved_tokens(image_pad=10),
        pad_token_id=0,
        max_assembled_length=16,
    )
    assert result.inputs_embeds.shape == (1, 4, 3)
    assert result.expanded_input_ids.tolist() == [[5, 10, 10, 6]]
    assert result.inputs_embeds[0, 1:3, 0].tolist() == [100.0, 200.0]
    assert result.labels.tolist() == [[-100, -100, -100, 6]]
    assert result.attention_mask.tolist() == [[1, 1, 1, 1]]
```

- [ ] **Step 2: Run and observe the missing assembler failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_sequence_assembler.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement deterministic row assembly**

Public signature:

```python
class SequenceAssembler:
    def assemble(
        self,
        *,
        input_ids: torch.LongTensor,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        media_sequences: Sequence[MediaSequence],
        tokens: ResolvedMultimodalTokens,
        expansion_policy: MediaExpansionPolicy | None = None,
        embedding_lookup: EmbeddingLookup | None = None,
        pad_token_id: int,
        joint_separator_token_ids: frozenset[int] = frozenset(),
        max_assembled_length: int,
    ) -> AssembledSequence:
        ...
```

Shared temporal quantization has the exact public signature:

```python
def quantize_timestamps_half_up(
    timestamps: torch.Tensor,
    seconds_per_bucket: float,
) -> torch.LongTensor:
    ...
```

It requires a floating non-complex tensor and a real, non-boolean, finite,
positive step; every timestamp must be finite and non-negative. It first
downcasts any input dtype to float32 on the same device, constructs the step as
a float32 scalar there, and returns
`floor(value / step + 0.5).long()` with the exact input shape/device. It
rejects malformed values itself rather than relying on a caller's validation.

`SequenceAssembler` is a strict tensor boundary. `input_ids` is non-empty,
non-negative long `[B,T]`; `text_embeddings` is floating non-complex
`[B,T,H]` with `H > 0`; `attention_mask` is binary boolean/integer `[B,T]`;
and `labels` is either `None` or long `[B,T]` containing only `-100` or
non-negative values. Coupled text tensors share one device. Every row has a
non-empty right-padded prefix mask; masked IDs equal `pad_token_id` and masked
labels equal `-100`. A valid token may equal `pad_token_id`; only the mask
defines validity.

Every input `MediaSequence` is validated. Each media row must likewise have a
non-empty prefix mask so `MediaTokenRef.token_index` is simultaneously its
physical embedding column and complete-source row-major index. All media
embeddings exactly match the text hidden size, dtype, and device; the
assembler never silently casts them.

Define the profile extension point:

```python
@dataclass(frozen=True)
class MediaTokenRef:
    source: MediaSource
    token_index: int


@dataclass(frozen=True)
class MediaPlaceholder:
    text_position: int
    sentinel_token_id: int
    modality: MediaModality
    sequence: MediaSequence
    sequence_row: int


@dataclass(frozen=True)
class MediaExpansionGroup:
    placeholders: tuple[MediaPlaceholder, ...]
    first_text_position: int
    last_text_position: int


@dataclass(frozen=True)
class ExpansionToken:
    token_id: int
    kind: SequenceSpanKind
    media_ref: MediaTokenRef | None = None
    source: MediaSource | None = None


@dataclass(frozen=True)
class ExpandedMediaRow:
    tokens: tuple[ExpansionToken, ...]


@dataclass(frozen=True)
class ExpandedMediaSample:
    replacements: Mapping[int, ExpandedMediaRow]


class EmbeddingLookup(Protocol):
    def __call__(self, token_ids: torch.LongTensor) -> torch.Tensor:
        ...


class MediaExpansionPolicy(Protocol):
    def expand_sample(
        self,
        *,
        sample_index: int,
        groups: tuple[MediaExpansionGroup, ...],
    ) -> ExpandedMediaSample:
        ...


class AssembledLengthError(ValueError):
    sample_index: int
    assembled_length: int
    max_assembled_length: int
    retained_text_tokens: int
    inserted_expansion_tokens: int
```

`SequenceSpanKind.MEDIA` expansion tokens require an in-range `media_ref`;
their `token_id` must equal
`tokens.sentinel_for(the referenced MediaSequence.modality)`, and when
`source` is supplied it must equal `media_ref.source`. Arbitrary, negative or
cross-modality sentinel IDs are rejected.
`SequenceSpanKind.TIMESTAMP` forbids `media_ref`, requires a source from the
same approved group, and requires an ordinary non-pad, non-sentinel token ID.
Policies may not emit `TEXT` expansion tokens.

`IdentityMediaExpansion` returns one replacement per placeholder, repeats that
placeholder's sentinel ID once per valid `MediaTokenRef` and preserves the
source unchanged. A policy never supplies media embeddings: the assembler
materializes every media token from the referenced validated source row/index.
This makes exact source consumption independently verifiable instead of
trusting a policy's self-reported consumed-source list.

`TimestampInterleaveExpansion` is the production experimental policy. It stores
only `seconds_per_bucket` and an immutable snapshot of a finite
bucket→ordinary-token-ID mapping. `bucket_token_ids` must be a
`Mapping[int,int]`; an empty mapping is valid so image-only groups can use
identity expansion, while any AV bucket encountered with no entry is an
unmapped-bucket error. The constructor validates the entire mapping and copies
it, so later caller mutation cannot affect policy behavior. For an AV-only
assembler-approved group it
stable-sorts timestamped audio/video `MediaTokenRef` objects by
`(timestamp,item_index,token_index)`. It computes buckets through the shared
helper using float32 tensor arithmetic: convert timestamps to float32 without
promoting them, create a float32 step on the same device, then evaluate
`floor(timestamp / step + 0.5).long()`. Python `round`, float64 promotion and
precomputed Python buckets are forbidden. It emits exactly one
`SequenceSpanKind.TIMESTAMP` token immediately before the first media token in
every bucket, including the first. The marker uses the mapped ordinary token
ID and the source of that bucket's first deterministically sorted media ref.
It assigns the jointly sorted output to the first placeholder with exactly
empty replacements for later placeholders. Image-only groups use identity
expansion per placeholder with no timestamp token; one group mixing IMAGE with
AUDIO/VIDEO is rejected. Images and AV may coexist in a sample when text puts
them in separate groups. A missing/non-finite timestamp or unmapped bucket is
an error. It never groups across a boundary itself.
Its constructor rejects booleans, NaN/Inf or non-positive
`seconds_per_bucket`, non-integer/negative bucket keys, and
non-integer/negative token IDs.

Both policies are pure, parameter-free non-modules and never own or alias the
language-model embedding. `embedding_lookup` is used only by the assembler to
materialize ordinary timestamp tokens. For each timestamp-bearing row, the
assembler makes exactly one lookup call with final-order long IDs on
`input_ids.device`; it requires a floating
`[timestamp_token_count,hidden_size]` result on the exact text-embedding
dtype/device. It does not call the lookup for rows without timestamp tokens.
A policy may not bypass source-span or label validation.

For each batch row:

1. validate all scalar/text fields, then every `MediaSequence`, global source
   uniqueness, hidden size/dtype/device, and non-empty prefix masks;
2. map media sources by `(sample_index, item_index)`, require item indices to be
   exactly `0..N-1`, and pair the row's `k`th sentinel only with
   `(sample_index,k)`; transport/container order is never pairing semantics;
3. build maximal `MediaExpansionGroup` objects over consecutive placeholders
   in original valid-text order. A consecutive pair joins iff every valid
   token strictly between it belongs to `joint_separator_token_ids`; an empty
   interval joins vacuously. Group bounds are inclusive first/last placeholder
   positions. Labels do not grant adjacency: unsupervised natural-language
   tokens still split groups;
4. collect all validated groups for that sample and call
   `expand_sample()` exactly once, including `groups=()` for a no-media row,
   whose only valid replacement mapping is empty;
5. verify the result has exactly the placeholder-position keys (a joint policy
   may assign a zero-length replacement to later placeholders), validate every
   result/container/token type and exact non-negative ID, and require one of
   two shapes per group: either each replacement contains only its own source,
   or the first replacement may contain any group source while all later
   replacements are exactly empty. Refs may never move to a later placeholder
   or split between first and later replacements. The output
   media refs are an exact permutation of every valid input
   `(MediaSource, token_index)` pair. Filtering the output by any one source
   must yield token indices exactly `0..valid_count-1`, preserving that
   source's temporal/row-major order even when sources are interleaved;
6. count `retained_text_tokens` as valid non-sentinel input tokens and
   `inserted_expansion_tokens` as all MEDIA and TIMESTAMP policy output. Before
   lookup/materialization, raise `AssembledLengthError` when their sum exceeds
   the limit; expose all five declared attributes in its message;
7. walk valid text tokens left-to-right, remove sentinels, and copy every
   ordinary text ID/embedding/label exactly;
8. splice the symbolic replacement, gathering media embeddings from canonical
   refs and timestamp embeddings through the one-per-row lookup contract;
9. emit maximal deterministic spans. TEXT carries no media metadata. MEDIA is
   maximal for one source with consecutive indices and carries its canonical
   modality, complete grid, timestamps selected by canonical indices,
   `seconds_per_grid`, and exact indices. TIMESTAMP is maximal for one source,
   carries its canonical source/modality, and has no grid/timestamps/cadence/
   source indices. Empty replacements emit no span. Mask every inserted label
   with `-100`;
10. right-pad to the maximum assembled valid row length with the validated
    caller `pad_token_id`, zero embeddings, boolean false masks and `-100`
    labels. Preserve `labels=None` and call `AssembledSequence.validate()`.

A policy that combines AV items may emit their jointly sorted sequence at the
first placeholder and zero tokens at the remaining placeholders only inside
one pre-approved group. It therefore cannot reorder media across supervised or
unsupervised natural-language content.

Do not encode media and do not compute position IDs. Require `tokens` to be
`ResolvedMultimodalTokens`. Every present resolved token ID, `pad_token_id`,
`max_assembled_length`, and separator ID uses exact integer semantics
(booleans rejected). Present resolved IDs are non-negative and pairwise
distinct; pad is non-negative and distinct from all media sentinels; the hard
limit is positive. `joint_separator_token_ids` is a `frozenset` of
non-negative integers disjoint from pad/media sentinels. Wrapper IDs may be
whitelisted separators. `expansion_policy=None` means a fresh parameter-free
`IdentityMediaExpansion()` for that call.

- [ ] **Step 4: Fail on every ambiguous mapping**

Add exact errors for:

- wrong text ID/label dtype, non-binary or holey text/media masks, a masked
  supervised label, zero-valid row, text/media dtype/device mismatch, and a
  malformed timestamp-lookup result;
- wrong text rank/coupled shape, integer or complex text embeddings, a
  negative input ID, any negative label other than `-100` (including `-1` and
  `-2`), and a masked input ID unequal to `pad_token_id`, all before policy
  invocation;
- invalid/bool/colliding resolved token IDs, a pad/media-sentinel collision,
  bool scalar values, and a mutable/non-frozen separator collection;
- sentinel with no media item;
- extra media item;
- modality mismatch;
- duplicated `(sample_index, item_index)`;
- item-index gaps and duplicate `(sample_index, source_id)`;
- source batch index outside the text batch;
- hidden-size mismatch;
- all-masked media sequence;
- a media ref duplicated, missing, out of range or moved across an approved
  expansion group;
- refs moved to the last placeholder or split between first/later
  replacements, and malformed policy result/container/token types;
- a media expansion token whose ID is not the referenced source modality's
  exact sentinel;
- a source's media refs reordered even when the global ref set remains an
  exact permutation;
- a timestamp expansion without `embedding_lookup`;
- assembled length above the caller-provided hard limit.

- [ ] **Step 5: Test multiple items and mixed batch lengths**

Cover:

- image→text→audio order;
- two items of the same modality;
- image and video in one sample;
- no-media sample sharing a batch with media samples;
- inference with `labels=None`;
- text padding and target labels after expansion; exact output padding is the
  non-zero pad ID, zero embedding, `False` mask and `-100` label, while every
  retained text field is bitwise equal to its valid input value;
- exact complete-source span start/end positions;
- expanded token IDs for every media position;
- a custom sample policy that inserts two ordinary timestamp tokens through the
  supplied lookup and preserves source-span coverage;
- a joint policy fixture with timestamps
  `[audio0=0.00, video0=0.08, audio1=0.16, video1=0.24]` whose assembled media
  spans are exactly audio→video→audio→video;
- half-up bucket boundaries at `0.5` and `1.5` bucket units, a marker before
  the first bucket, one marker for repeated refs in a bucket, and marker source
  equal to that bucket's first sorted ref;
- missing AV timestamps and an unmapped current bucket fail explicitly;
- `bucket_token_ids` rejects non-mappings and bool/negative keys or IDs;
  empty mapping passes for image-only groups and fails when AV is encountered;
  mutating the caller's mapping after policy construction does not change
  output;
- direct shared-helper tests cover non-Tensor/integer/complex timestamps,
  negative/NaN/Inf values, bool/NaN/Inf/non-positive steps, exact shape/device
  preservation and float64 input downcast before the boundary calculation;
- timestamp lookup is called exactly once for each timestamp-bearing row and
  never for other rows; wrong count/rank/hidden-size/dtype/device results fail;
- image-only groups use identity expansion, image and AV work in separate
  groups in one sample, and a single mixed image+AV group fails;
- groups are maximal: adjacent/whitelist-only placeholder intervals join,
  natural text splits even when unsupervised, and a no-media row calls the
  policy once with `groups=()` and requires an empty mapping;
- direct assembler tests shuffle both the `media_sequences` container and rows
  inside a `MediaSequence` while preserving keys and require identical output;
- a non-zero `pad_token_id` test where neither original row has padding but
  media expansion makes one row longer;
- the same joint expansion rejects supervised or unsupervised natural-language
  separators, duplicated/missing/foreign/reordered media refs, item-index gaps,
  or an unknown replacement key;
- a holey media mask `[1,0,1]` fails before policy invocation, while
  `[1,1,0]` gathers exact columns/indices `(0,1)`;
- interleaved video fragments retain original row-major token indices so later
  position builders reconstruct exact H/W coordinates;
- MEDIA spans preserve canonical complete grid, selected timestamps, cadence
  and exact source indices after interleaving; TIMESTAMP spans carry only the
  chosen source/modality and empty replacements create no zero-length span;
- the hard limit passes at equality and fails at `limit+1` with timestamp
  tokens counted as inserted expansion and replaced sentinels excluded from
  retained text, asserting all `AssembledLengthError` fields;
- arbitrary/negative/cross-modality media token IDs are rejected even when the
  symbolic ref and embedding are otherwise valid.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_sequence_assembler.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/multimodal/sequence_assembler.py \
  src/qwen3_omni_pretrain/multimodal/time_quantization.py \
  tests/multimodal/test_sequence_assembler.py
git commit -m "feat: assemble multimodal token sequences"
```

---

### Task 6: Build legacy, Qwen3-disjoint, and experimental TM-RoPE positions

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/positions.py`
- Create: `tests/multimodal/test_position_builders.py`
- Create: `tests/oracle/qwen3_omni_rope_oracle.py`
- Create: `tests/oracle/test_qwen3_tm_rope_oracle.py`
- Modify: `tests/oracle/test_qwen3_omni_processor_oracle.py`

**Interfaces:**
- Consumes: `AssembledSequence`.
- Produces: `PositionBuilder` protocol, `LegacyPositionBuilder`,
  `Qwen3DisjointPositionConfig`, `Qwen3DisjointPositionBuilder`,
  `TMRoPEConfig`, and experimental `TMRoPEPositionBuilder`.

- [ ] **Step 1: Write failing text/image/audio position tests**

```python
def test_experimental_tm_rope_aligns_audio_to_80ms_grid():
    assembled = assembled_audio_example(
        timestamps=torch.tensor([0.00, 0.08, 0.16])
    )
    positions = TMRoPEPositionBuilder(
        TMRoPEConfig(
            temporal_seconds_per_id=0.08,
            rotary_sections=(24, 20, 20),
        )
    ).build(assembled)
    audio_span = next(
        span for span in assembled.spans
        if span.modality is MediaModality.AUDIO
    )
    audio_ids = positions.position_ids[
        :, 0, audio_span.start:audio_span.end
    ]
    assert torch.equal(audio_ids[0], audio_ids[1])
    assert torch.equal(audio_ids[1], audio_ids[2])
    assert torch.diff(audio_ids[0]).tolist() == [1, 1]


def test_qwen3_official_image_uses_pinned_three_axis_vector():
    assembled = assembled_image_example(grid=MediaGrid(1, 2, 2))
    positions = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig(
            position_id_per_seconds=13.0,
            rotary_sections=(24, 20, 20),
        )
    ).build(assembled).position_ids[:, 0]
    media = media_slice(positions, assembled)
    assert media[0].unique().numel() == 1
    assert media[1].tolist() == [0, 0, 1, 1]
    assert media[2].tolist() == [0, 1, 0, 1]
```

- [ ] **Step 2: Run and observe missing builders**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_position_builders.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Define the builder protocol and configs**

```python
class PositionBuilder(Protocol):
    def build(self, assembled: AssembledSequence) -> PositionBatch:
        ...


@dataclass(frozen=True)
class Qwen3DisjointPositionConfig:
    position_id_per_seconds: float = 13.0
    rotary_sections: tuple[int, int, int] = (24, 20, 20)

    def __post_init__(self) -> None:
        if isinstance(self.position_id_per_seconds, bool) or not isinstance(
            self.position_id_per_seconds, Real
        ):
            raise TypeError("position_id_per_seconds must be a real number")
        if (
            not math.isfinite(float(self.position_id_per_seconds))
            or self.position_id_per_seconds <= 0
        ):
            raise ValueError("position_id_per_seconds must be finite and positive")
        if not isinstance(self.rotary_sections, tuple):
            raise TypeError("rotary_sections must be a tuple")
        if len(self.rotary_sections) != 3:
            raise ValueError("rotary_sections must contain three positive integers")
        if any(type(section) is not int for section in self.rotary_sections):
            raise TypeError("rotary_sections must contain integers")
        if any(section <= 0 for section in self.rotary_sections):
            raise ValueError("rotary_sections must contain positive integers")


@dataclass(frozen=True)
class TMRoPEConfig:
    temporal_seconds_per_id: float
    rotary_sections: tuple[int, int, int]
    interleaved: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.temporal_seconds_per_id, bool) or not isinstance(
            self.temporal_seconds_per_id, Real
        ):
            raise TypeError("temporal_seconds_per_id must be a real number")
        if (
            not math.isfinite(float(self.temporal_seconds_per_id))
            or self.temporal_seconds_per_id <= 0
        ):
            raise ValueError(
                "temporal_seconds_per_id must be finite and positive"
            )
        if not isinstance(self.rotary_sections, tuple):
            raise TypeError("rotary_sections must be a tuple")
        if len(self.rotary_sections) != 3:
            raise ValueError(
                "rotary_sections must contain three positive integers"
            )
        if any(type(section) is not int for section in self.rotary_sections):
            raise TypeError("rotary_sections must contain integers")
        if any(section <= 0 for section in self.rotary_sections):
            raise ValueError("rotary_sections must contain positive integers")
        if type(self.interleaved) is not bool:
            raise TypeError("interleaved must be a boolean")
```

Both configs reject booleans, NaN/Inf and non-positive temporal values, require
exactly three positive integer rotary sections, and expose the sections as a
downstream rotary-layout contract. Position generation does not itself split
head dimensions; tests assert the pinned `(24,20,20)` contract explicitly.
`TMRoPEConfig` additionally requires `type(interleaved) is bool`; version one
preserves either value only as downstream rotary-layout metadata because
position generation does not consume it. Wrong types raise `TypeError` and
invalid numeric values raise `ValueError` consistently. Define exact module
constants for axis names `("sequence",)` and
`("temporal","height","width")`.

Every `build()` first calls `assembled.validate()`, then requires a non-empty
batch and sequence plus one or more valid tokens in every non-empty
right-padded prefix row. It allocates outputs on
`assembled.attention_mask.device`, zeroes every masked returned position, and
calls `PositionBatch.validate(assembled.attention_mask)` before return.

`LegacyPositionBuilder` returns exactly `torch.long [1,B,S]` with axis
`("sequence",)`. For valid row length `L`, valid IDs are `0..L-1`, padded IDs
are zero, and `rope_deltas` is an exact long zero tensor `[B,1]`. It checks
monotonicity only over valid positions. This matches the current legacy
model's default arange at every supported valid prefix position; legacy does
not consume Qwen-style deltas.
Both three-axis builders return `float32 [3,B,S]`, axes
`("temporal","height","width")`, and `float32 rope_deltas [B,1]`.

- [ ] **Step 4: Implement exact official and separate experimental semantics**

`Qwen3DisjointPositionBuilder` reproduces the pinned
`use_audio_in_video=False` branches, not the joint-AV facade and not the
experimental timestamp grid. It groups by complete `MediaSource`, requires
exactly one MEDIA span per source (even adjacent split fragments fail), and
requires that span's indices to be exactly `0..valid_count-1`. Qwen3 profile
wiring must use `IdentityMediaExpansion`; the builder cannot infer policy
provenance from an already assembled tensor and therefore makes no broader
categorical joint-policy claim.

- use `position_id_per_seconds=13.0` and preserve the official float32 eager
  operation order (no 80 ms rounding): cast original frame index to float32,
  multiply by a float32 `seconds_per_grid`, then separately multiply by a
  float32 rate and add the float32 continuation. Never precompute/fold the two
  scalar factors or round-trip continuation/max through Python scalars;
- text and timestamp spans advance all axes together from one plus the maximum
  value across **all** axes emitted so far;
- image T/H/W follow the official merged-grid layout and require
  `grid.temporal == 1`. Project `MediaGrid` H/W are post-spatial-merge output
  token axes; the oracle maps them to official pre-merge axes by the pinned
  spatial merge factor (`project 1x2x2 <-> official 1x4x4`);
- video requires explicit finite positive `seconds_per_grid` whenever
  `grid.temporal > 1`; temporal IDs use the separate float32 operations above
  and then the current continuation tensor;
- image/video H/W and frame indices are reconstructed from each span's
  `source_token_indices`, never from the fragment's new output offset;
- audio media IDs follow the pinned sequential feature-grid behavior;
- after every block, continuation is based on the maximum across T/H/W;
- padded positions are normalized to zero. Select the pure-text fallback only
  when the entire batch contains no MEDIA span. If any row has media, every
  row uses the multimodal continuation/delta branch. For such a batch,
  `rope_deltas` is
  `max(position_ids over all valid axes) + 1 - valid_sequence_length`;
- for a pure-text batch (the facade branch with all media metadata `None`),
  reproduce the official delta's masked filler semantics:
  compute the maximum over valid 1D positions plus an implicit value `1` when
  padding exists, then subtract valid length. Thus one valid token plus padding
  has delta `1.0`, even though the common output normalizes masked positions to
  zero.

The common builder deliberately normalizes masked padding positions to zero.
The pinned text-only facade fills them with one; masked values are outside the
numerical compatibility claim. Oracle comparisons are exact on valid
positions, dtype/shape, attention-mask behavior and rope delta. Add a
non-skippable single-valid-token-plus-padding vector and a
two-valid-token-plus-padding vector to prevent this branch from regressing.
Also add a mixed image/one-token-text batch proving the text row's delta is
`0.0`, while the identical row in an all-text batch has delta `1.0`.

The mandatory non-skippable fractional oracle uses project
`MediaGrid(8,1,1)` (official `[8,2,2]`), `seconds_per_grid=0.08`, one start
token and one end token. Its exact float32 temporal row is
`[0.0, 1.0, 2.0399999618530273, 3.0799999237060547,
4.119999885559082, 5.159999847412109, 6.199999809265137,
7.239999771118164, 8.280000686645508, 9.280000686645508]` and its delta is
`0.2800006866455078`; this must distinguish sequential multiplication from a
folded step.

`TMRoPEPositionBuilder` is explicitly experimental. For each sample, start
`text_cursor=0`, `timeline_anchor=None`, and `max_position_seen=-1`:

- before each span use
  `continuation=max(text_cursor,max_position_seen+1)`;
- text and canonical `SequenceSpanKind.TIMESTAMP` spans use
  `continuation + arange(length)` on all axes, then set
  `text_cursor=continuation+length`;
- image requires no timestamps and `grid.temporal == 1`; T is continuation,
  while H/W are continuation plus row/column reconstructed from original
  complete-source indices. Consecutive images therefore cannot reuse a base;
- on the first timestamped audio/video span, set the single sample-global
  `timeline_anchor=continuation`;
- timestamped audio uses
  `timeline_anchor + quantize_timestamps_half_up(timestamp,
  temporal_seconds_per_id)` on all T/H/W axes;
- timestamped video uses the same value for T and
  `timeline_anchor + row/column` for H/W. Aggregate all fragments for a source,
  derive frame from `source_index // (grid.height * grid.width)`, require every
  patch in one frame to have exactly one identical global timestamp, quantize
  one value per frame and scatter back to each fragment;
- update `max_position_seen` from all three axes after every span, but do not
  add a new per-modality timeline origin. Two audio/video tokens at the same
  global timestamp therefore receive the same T ID even if their spans are
  separate;
- spatial axes may reset and a later media span may reuse an earlier temporal
  ID; causal order remains the assembled token order, not numeric position
  monotonicity;
- padded output positions are zero and ignored by the attention mask;
- `rope_deltas` is exactly
  `max_position_seen + 1 - valid_sequence_length` in float32 `[B,1]`.

The experimental builder supports exactly untimestamped static IMAGE,
timestamped AUDIO with no grid, timestamped VIDEO with a complete grid, and
canonical text-like TIMESTAMP spans. Video `seconds_per_grid` may remain as
metadata but explicit timestamps are authoritative. Every IMAGE source must
occupy exactly one complete MEDIA span; split/fragmented images are rejected,
while validated AUDIO/VIDEO source fragmentation remains supported. Reject untimestamped
AUDIO/VIDEO, timestamped IMAGE, non-finite/negative/non-monotonic source
timestamps, inconsistent timestamps within a video frame, and a complete grid
whose token count disagrees with the source's total referenced token count.
Never compare one fragment length to the full grid count.

Experimental quantization imports the Task 5 shared helper and stays in
float32: convert timestamps to float32 without promotion, construct a float32
step on the same device, then compute `floor(value / step + 0.5).long()`.
Python round and float64 promotion are forbidden. Add cross-component tests at
0.5 and 1.5 bucket boundaries and float32 epsilon neighbors that assert both
the Task 5 timestamp marker ID and Task 6 media T coordinate. Add a fixture
where audio and video at global `0.32s` share T while spatial IDs differ,
timestamp spans that advance continuation, consecutive image and image→first
AV literal vectors, and an A/B/A fragmented video fixture with exact original
coordinates. Qwen disjoint tests reject split sources and missing/invalid
multi-frame cadence; experimental accepts validated fragmentation.

Add strict common and legacy coverage:

- every builder rejects invalid `AssembledSequence`, empty batch/sequence,
  all-masked or holey rows, non-binary masks and device mismatch before output;
- CPU dtype/device/axis names and masked zeros are exact, CUDA is covered when
  available, and every result passes `PositionBatch.validate(mask)`;
- a two-row unequal-length legacy batch (including non-zero pad ID), one-token
  padded row, and a media-containing row all receive long sequential valid
  IDs and long zero deltas;
- config tests reject booleans/lists/zero/NaN/Inf as specified and verify both
  boolean `interleaved` values are preserved without changing positions;
- Qwen covers no-padding text, both padded-text quirks, batch-global mixed
  branch, pinned image/audio/integer-video/fractional-video vectors,
  continuation across multiple disjoint blocks, original-index H/W, a
  temporal-one video without cadence, and adjacent/A-B-A source fragmentation
  failures;
- experimental covers distinct 80 ms/160 ms configs, exact float32 boundary
  quantization, fixed anchor across later text/marker spans, consecutive
  images, image-to-first-AV anchor, split-image rejection, fragmented A/B/A
  AV sources, inconsistent
  frame-patch timestamps, every unsupported metadata combination, exact delta,
  padding independence, and permitted spatial/temporal reuse.

- [ ] **Step 5: Compare Qwen3 disjoint builder with official `get_rope_index()`**

Extract the pinned configuration-only facade and fixed vectors from
`test_qwen3_omni_processor_oracle.py` into the reusable
`tests/oracle/qwen3_omni_rope_oracle.py`. Both oracle test modules import that
helper. The helper contains only immutable constants/fresh vector factories,
the lightweight facade, pinned Transformers version, expected implementation
SHA-256 and hash function. It never imports pytest, calls `from_pretrained`,
constructs model layers, or skips. The non-skippable oracle test first requires
`transformers==5.2.0`, verifies the implementation hash and official facade
against every fixed vector, then independently converts each vector to
`AssembledSequence` and compares the project builder. Missing/wrong
Transformers, a hash mismatch, or a fixed-vector mismatch fails.

The checked-in set includes no-padding text, one-valid-plus-padding text,
two-valid-plus-padding text, the mixed image/text batch-global branch, image,
audio, integer-cadence video and the fractional 0.08-second eight-frame video.
Oracle conversion explicitly maps official pre-merge H/W grids to project
post-merge grids. Compare valid Qwen3 IDs, exact dtype/device-independent
values, shapes and rope deltas. These vectors cover
`use_audio_in_video=False`; profile wiring owns `IdentityMediaExpansion`, and
Task 6 makes no full joint-AV parity claim. The later Qwen3 reference-runtime
plan keeps the official facade for its audio-video parity fixture.

Keep the processor-derived integration case separate and
`@pytest.mark.reference`; only that integration case may explicitly skip when
the pinned processor artifact is not cached.

- [ ] **Step 6: Run prototype and oracle tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/multimodal/test_position_builders.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/test_qwen3_tm_rope_oracle.py \
  tests/oracle/test_qwen3_omni_processor_oracle.py -q
```

Expected: both suites pass. The fixed-vector oracle never skips; only a
separate processor integration test may skip when its artifact is absent.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/multimodal/positions.py \
  tests/multimodal/test_position_builders.py \
  tests/oracle/qwen3_omni_rope_oracle.py \
  tests/oracle/test_qwen3_omni_processor_oracle.py \
  tests/oracle/test_qwen3_tm_rope_oracle.py
git commit -m "feat: build time-aligned multimodal positions"
```

---

### Task 7: Expose a reusable multimodal prefill pipeline

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/prefill.py`
- Create: `tests/multimodal/test_multimodal_prefill.py`
- Modify: `src/qwen3_omni_pretrain/multimodal/__init__.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: strict retained text tensors, the final `DecodedMedia` tuple from
  the collator, one caller-owned text-embedding callable, registered media
  encoders, an explicit expansion policy, `SequenceAssembler`, and one
  explicit `PositionBuilder`.
- Produces: frozen `MultimodalPrefillOutput(assembled, positions,
  media_sequences)` through both conventional `forward(...)` and the named
  `encode_and_assemble(...)` API.
- Owns only the three media encoders. It never owns a text embedding, language
  model, decoder, quarantine policy, cache, or profile factory.

- [ ] **Step 1: Write a failing end-to-end prefill test**

```python
def test_prefill_keeps_image_and_audio_sequences_in_prompt_order():
    pipeline = tiny_prefill_pipeline(
        position_builder=tiny_three_axis_builder(),
    )
    text_embedding = RecordingEmbedding(32, 8)
    output = pipeline.encode_and_assemble(
        input_ids=torch.tensor([[5, 10, 6, 12, 7]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        labels=torch.tensor([[-100, -100, -100, -100, 7]]),
        decoded_media=(
            tiny_image_item(source_id="image-0"),
            tiny_audio_item(source_id="audio-0"),
        ),
        text_embedding=text_embedding,
    )
    assert output.assembled.inputs_embeds.shape[1] > 5
    assert [span.modality for span in output.assembled.spans if span.modality] == [
        MediaModality.IMAGE,
        MediaModality.AUDIO,
    ]
    assert output.positions.position_ids.shape[:2] == (3, 1)
    assert text_embedding.rank2_calls == 1
```

- [ ] **Step 2: Run and observe missing pipeline failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_multimodal_prefill.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement orchestration without decoder logic**

```python
@dataclass(frozen=True)
class MultimodalPrefillOutput:
    assembled: AssembledSequence
    positions: PositionBatch
    media_sequences: tuple[MediaSequence, ...]


class MultimodalPrefillPipeline(nn.Module):
    def __init__(
        self,
        *,
        tokens: ResolvedMultimodalTokens,
        image_encoder: PatchVisionEncoder,
        video_encoder: TemporalVideoEncoder,
        audio_encoder: AudioWindowEncoder,
        assembler: SequenceAssembler,
        expansion_policy: MediaExpansionPolicy,
        position_builder: PositionBuilder,
        pad_token_id: int,
        joint_separator_token_ids: frozenset[int],
        max_assembled_length: int,
    ) -> None:
        super().__init__()
        # Perform the exact checks specified below before registration.
        self.image_encoder = image_encoder
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.tokens = tokens
        self.assembler = assembler
        self.expansion_policy = expansion_policy
        self.position_builder = position_builder
        self.pad_token_id = pad_token_id
        self.joint_separator_token_ids = joint_separator_token_ids
        self.max_assembled_length = max_assembled_length

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor | None,
        decoded_media: Sequence[DecodedMedia],
        text_embedding: EmbeddingLookup,
    ) -> MultimodalPrefillOutput:
        ...

    def encode_and_assemble(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor | None,
        decoded_media: Sequence[DecodedMedia],
        text_embedding: EmbeddingLookup,
    ) -> MultimodalPrefillOutput:
        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoded_media=decoded_media,
            text_embedding=text_embedding,
        )
```

`forward()` is the canonical implementation. The named method delegates via
`self(...)`, so normal `nn.Module` hooks still run; the reverse delegation is
forbidden. The constructor performs all of these exact trusted-boundary checks:

- `type(tokens) is ResolvedMultimodalTokens`; every present token ID has exact
  `int` type, is non-negative, and all present IDs are pairwise distinct;
- `pad_token_id` has exact `int` type, is non-negative, and differs from the
  image/video/audio sentinel IDs;
- `max_assembled_length` has exact `int` type and is positive;
- `joint_separator_token_ids` is already a `frozenset`; every member has exact
  non-negative `int` type and differs from pad and all media sentinel IDs.
  Wrapper IDs may be explicitly whitelisted. The pipeline never normalizes a
  list, set, iterator, bool, or other mutable/ambiguous input;
- the encoders are respectively `PatchVisionEncoder`,
  `TemporalVideoEncoder`, and `AudioWindowEncoder`, and their exact
  `hidden_size` values agree;
- `assembler` is a `SequenceAssembler`; the non-`None` policy exposes callable
  `expand_sample`, and the builder exposes callable `build`;
- assembler, policy, and builder are parameter-free non-`nn.Module` objects.
  Use structural callable validation rather than `isinstance()` against a
  non-runtime-checkable protocol.

The pipeline is cast-free at its orchestration boundary. It never moves or
casts `input_ids`, masks, labels, base text embeddings, or encoder-produced
sequences. Only the encoders may move/cast decoded float tensors. The caller
must put the external text embedding and all registered encoders on one device
and floating dtype. A BF16 pipeline paired with an FP32 text embedding fails;
there is no implicit CPU/GPU or FP32/BF16 fallback. Video receives the exact
registered `self.image_encoder` object, but never registers it as a child.

`forward()` follows this non-negotiable execution order:

1. Validate the Task 5 text preflight before any embedding, encoder, policy,
   assembler, or builder call: tensors have exact ranks and coupled `[B,T]`
   shapes with non-empty dimensions; IDs/labels are long; IDs are
   non-negative; labels contain only `-100` or non-negative values; the mask is
   binary with a non-empty right-padded prefix per row; all coupled tensors
   share a device; masked IDs equal the exact pad ID and masked labels equal
   `-100`. `labels=None` remains valid.
2. Require `decoded_media` to be a non-string `Sequence`, snapshot it to a
   tuple without mutation, and require every entry to be `DecodedMedia`.
   Before encoder work, globally validate final request keys against batch
   size: every `sample_index` is in range; `(sample_index,item_index)` and
   `(sample_index,source_id)` are unique across all modalities; and each
   sample's item indices are exactly `0..N-1`. Never repair/reindex requests.
3. Sort the snapshot by `(sample_index,item_index)`, then partition in the
   fixed order IMAGE, VIDEO, AUDIO. This is the canonical public
   `media_sequences` order; it is intentionally independent of the prompt
   order later represented by assembled spans.
4. Require `text_embedding` to be callable and invoke
   `text_embedding(input_ids)` exactly once with the original rank-2 tensor.
   Require a floating non-complex tensor of exact shape `[B,T,H]`, on
   `input_ids.device`, with `H` equal to the encoders' common hidden size.
   Never assign/cache the callable on `self`. On every call, collect all
   floating parameters from all three registered encoders, require a non-empty
   collection with one exact common `(device,dtype)` pair, and require the base
   text embedding to match it. This applies when `decoded_media=()` or a
   modality is omitted and is repeated per call because a caller may move one
   child module independently after construction.
5. Invoke exactly one top-level encoder `forward` for each non-empty canonical
   partition and no encoder for an empty partition. For video call
   `video_encoder(items, patch_encoder=self.image_encoder)`; its internal patch
   helper use is not a second image-encoder `forward`.
6. Require each result to be `MediaSequence`, call `validate()`, require its
   modality to equal the partition modality, and require its sources to equal
   the partition requests converted to `MediaSource` in canonical order.
   Require embeddings to match the base text hidden size, dtype, and device.
7. Call the assembler exactly once with every explicit argument:

   ```python
   assembled = self.assembler.assemble(
       input_ids=input_ids,
       text_embeddings=text_embeddings,
       attention_mask=attention_mask,
       labels=labels,
       media_sequences=media_sequences,
       tokens=self.tokens,
       expansion_policy=self.expansion_policy,
       embedding_lookup=text_embedding,
       pad_token_id=self.pad_token_id,
       joint_separator_token_ids=self.joint_separator_token_ids,
       max_assembled_length=self.max_assembled_length,
   )
   ```

   The same non-owning callable is used for Task 5 timestamp markers. Total
   lookup count is one base rank-2 call plus one rank-1 call for each assembled
   row whose policy expansion contains at least one
   `SequenceSpanKind.TIMESTAMP` token. Timestamped MEDIA spans alone never
   trigger a lookup. A hard-length failure occurs after the base call/required
   encoding, but before all rank-1 timestamp lookups.
8. Require an `AssembledSequence` and call `assembled.validate()`.
9. Call `position_builder.build(assembled)` exactly once. It is never called
   if assembly or assembled validation fails.
10. Require a `PositionBatch` and call
    `positions.validate(assembled.attention_mask)`.
11. Return the frozen output without adding diagnostic fields or mutating an
    input. Malformed custom returns fail explicitly, never via a later
    incidental attribute error.

`decoded_media=()` is valid for non-empty text. It makes no encoder call and
returns `media_sequences=()`, while the base embedding, assembler, and builder
still run. For a subset, only present encoders run and no empty/fake sequence
is synthesized. Mixed text-only/media rows and `labels=None` are valid; an
empty text batch/sequence is not.

The collator remains the only quarantine owner. This pipeline accepts retained
text tensors and final dense `DecodedMedia` requests, but never decodes,
quarantines, removes rows, catches `MediaLoadError`, synthesizes missing media,
or copies `_sample_ids`/`_media_errors` into its output. "Direct" collator use
means explicitly passing its four model fields plus `text_embedding`, not
`encode_and_assemble(**collated)`, and the pipeline does not accept/ignore
arbitrary keyword fields.

Every caller supplies a matched policy/builder pair. The generic constructor
performs structural validation only and never claims semantic compatibility:

- Qwen3 disjoint: `IdentityMediaExpansion` with
  `Qwen3DisjointPositionBuilder(Qwen3DisjointPositionConfig(
  position_id_per_seconds=13.0, rotary_sections=(24,20,20)))`;
- Qwen3.5-inspired experiment: `TimestampInterleaveExpansion` and
  `TMRoPEPositionBuilder(TMRoPEConfig(temporal_seconds_per_id=0.16,
  rotary_sections=(24,20,20)))` with the same literal 160 ms quantum;
- MiMo: only its later explicitly declared experimental pair.

No profile-name conditionals, attribute introspection, implicit factory, or
checkpoint-compatibility claim is permitted here. Full Qwen3 joint-AV runtime
parity remains owned by the later reference-runtime adapter/oracle plan.

- [ ] **Step 4: Add strict constructor, ownership, and call-order tests**

Test:

- wrong encoder concrete types, unequal hidden sizes, malformed assembler,
  missing/malformed policy/builder, and any orchestration component that is an
  `nn.Module` fail at construction;
- bool/string/negative pad/limit values, malformed/colliding resolved tokens,
  mutable separators, bool/negative separator members, and pad/sentinel
  collisions fail; explicitly whitelisted wrapper IDs remain valid;
- only image, video-specific, and audio encoder parameters appear in
  `state_dict()`/`named_parameters()`, each exactly once. No patch subtree is
  nested below video and no external embedding appears before or after calls;
- `.to(dtype=...)` moves every registered floating encoder parameter, and
  video receives the exact registered image encoder;
- the base rank-2 lookup occurs exactly once; only rows whose policy expansion
  emits at least one `TIMESTAMP` token add the exact Task 5 rank-1 call.
  Timestamped MEDIA under identity expansion adds none. Callable non-modules
  work. Wrong lookup type, rank, shape, hidden size, dtype, or device fails
  explicitly;
- malformed text IDs/masks/labels fail before lookup, encoder, policy,
  assembler, or builder calls; `labels=None` and non-zero pad IDs work;
- malformed fake encoder/assembler/builder results fail at their declared
  boundary. Assembler and builder each run exactly once when valid, and module
  hooks fire through both `pipeline(...)` and `encode_and_assemble(...)`;
- at the hard limit the base lookup and media encoding occurred, no rank-1
  timestamp lookup or builder call occurred, and every
  `AssembledLengthError` field is preserved.

- [ ] **Step 5: Add dtype, ordering, omission, gradient, and quarantine tests**

Test:

- FP32 decoded tensors with pipeline plus external embedding moved to BF16
  produce BF16 assembled embeddings and finite gradients; a BF16 pipeline plus
  FP32 external embedding fails rather than casting, including with
  `decoded_media=()`. Moving/casting one omitted encoder away from the other
  two fails before every encoder, policy, assembler, and builder call, while a
  matched text-only BF16 path succeeds. Add matched/mismatched CUDA tests when
  CUDA exists, and preserve FP32 timestamps on output device;
- a transport shuffle leaves the complete result unchanged: canonical fixed
  modality and per-modality source order, masks, embeddings, grids,
  timestamps, assembled tensors/spans, and positions;
- cross-modality duplicate item keys/source IDs, out-of-range samples, and
  item-index gaps fail before encoder calls and are never repaired;
- a payload/request swap changes source-associated embeddings while spans keep
  the declared source keys;
- every modality subset, `decoded_media=()`, a mixed text-only/media batch,
  and `labels=None` call only present encoders, produce no fake sequence/span,
  and preserve ignored media labels;
- parameterize the text-only path over `IdentityMediaExpansion` paired with
  both Qwen3-disjoint and experimental TM-RoPE builders: no encoder call,
  `media_sequences=()`, exactly one rank-2 base lookup and no rank-1 marker
  lookup, only TEXT spans, and validated builder-specific three-axis
  positions/deltas;
- gradients reach image parameters through image and video-only paths,
  video-specific parameters through video, audio parameters through audio, and
  the external text embedding. Omitted modality parameters receive no
  fabricated gradient;
- Qwen3 disjoint uses literal expected `13 IDs/second` positions; the 160 ms
  experiment uses a literal matching policy/builder pair and produces actual
  audio0→video0→audio1→video1 assembled media order with timestamp markers;
- a bad first collator row and retained good row prove dense final
  `sample_index == 0`, preserved `original_sample_index`, external diagnostics,
  no mutation, and full-result shuffle invariance. Encoder/assembler failures
  propagate instead of becoming quarantine, and all-quarantined input fails in
  the collator before the pipeline is called.

- [ ] **Step 6: Export the public API and document the profile boundary**

Export `MultimodalPrefillPipeline` and `MultimodalPrefillOutput` from
`multimodal.__init__` and add a public import test. Update README:

- legacy still uses one-token prepend and is unchanged;
- new experimental profiles must use `MultimodalPrefillPipeline`;
- show explicit field reuse from the collator instead of `**collated`;
- show that callers co-locate the externally owned text embedding and the
  pipeline encoders, and that the pipeline performs no implicit casts;
- the common Qwen3 disjoint builder uses pinned fixed vectors/implementation as
  its non-skippable numerical oracle; full joint-AV Qwen3 reference execution
  remains on the official facade in the later runtime plan, with the cached
  processor as an additional integration oracle;
- policy/builder compatibility is caller-owned; passing common tests means
  sequence semantics are implemented, not Qwen3, Qwen3.5, or MiMo checkpoint
  compatibility.

- [ ] **Step 7: Run focused and full suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q
```

Expected: all focused and full-suite tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/qwen3_omni_pretrain/multimodal/prefill.py \
  src/qwen3_omni_pretrain/multimodal/__init__.py \
  tests/multimodal/test_multimodal_prefill.py README.md
git commit -m "feat: add sequence-preserving multimodal prefill"
```

---

## Plan completion gate

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q

PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/test_qwen3_tm_rope_oracle.py \
  tests/oracle/test_qwen3_omni_processor_oracle.py -q

git diff --check
git status --short
```

The plan is complete only when:

- image, audio and video retain more than one token when input structure
  contains more than one patch/window;
- multiple media items replace sentinels in exact conversation order;
- no omitted/corrupt media is silently replaced by a learned or zero summary;
- media labels are `-100` and text supervision survives expansion;
- Qwen3 disjoint-media `13 IDs/second` float TM-RoPE positions agree exactly
  with non-skippable pinned vectors; joint AV is explicitly deferred to the
  Qwen3 reference-runtime facade tests and experimental 80 ms remains separate;
- Qwen3.5 160 ms positions use the same contract without an official
  compatibility claim;
- legacy behavior and the full prototype suite remain green;
- all changes are intentional and committed.
