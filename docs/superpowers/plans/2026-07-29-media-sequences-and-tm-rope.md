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
- Qwen3 TM-RoPE 使用 T/H/W 三轴、`24/20/20` rotary section、80 ms temporal grid。
- Qwen3.5-inspired TM-RoPE 使用 160 ms temporal grid；本计划只实现可配置 builder，不声明 Qwen3.5 checkpoint 兼容。
- tokenizer 是媒体 token 语义的唯一来源，禁止跨 profile 复用裸数字。
- loader、encoder、assembler 和 position builder 的 fallback 不得改变数值语义。
- legacy wrapper 的现有测试和行为必须继续通过。
- Snippet 中的 `...` 只表示 Protocol/签名节选；任务提交不得保留未实现
  stub，必须由该任务列出的行为测试证明可用。
- 每个行为变更使用 red-green-refactor，并形成独立提交。

---

## File responsibility map

- `src/qwen3_omni_pretrain/multimodal/tokenization/schema.py`: profile-specific token string schema 和 tokenizer 解析。
- `src/qwen3_omni_pretrain/multimodal/modalities.py`: 无依赖的媒体模态枚举。
- `src/qwen3_omni_pretrain/multimodal/types.py`: media source、grid、sequence、span、assembled batch 和 position batch。
- `src/qwen3_omni_pretrain/multimodal/io.py`: 严格 image/audio/video decoding 与 quarantine，不做神经网络编码。
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
- Produces: `MediaSource`, `MediaGrid`, `MediaSequence`, `SequenceSpan`,
  `AssembledSequence`, and `PositionBatch`.

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

    def validate(self) -> None:
        if self.embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [B, M, H]")
        if self.attention_mask.shape != self.embeddings.shape[:2]:
            raise ValueError("attention_mask must have shape [B, M]")
        if len(self.sources) != self.embeddings.shape[0]:
            raise ValueError("one MediaSource is required per batch row")
```

Complete `validate()` with:

- boolean/integer masks only;
- grid tuple length equals batch;
- grid token count equals valid tokens for image/video rows;
- timestamps shape `[B, M]`;
- valid timestamps finite, non-negative and monotonic;
- `source_id` non-empty and `(sample_index, item_index)` non-negative.

- [ ] **Step 4: Define output contracts**

```python
@dataclass(frozen=True)
class SequenceSpan:
    sample_index: int
    start: int
    end: int
    kind: str
    modality: MediaModality | None
    grid: MediaGrid | None
    timestamps: torch.Tensor | None
    source_id: str | None


@dataclass(frozen=True)
class AssembledSequence:
    expanded_input_ids: torch.LongTensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    spans: tuple[SequenceSpan, ...]


@dataclass(frozen=True)
class PositionBatch:
    position_ids: torch.LongTensor
    rope_deltas: torch.LongTensor
    axis_names: tuple[str, ...]
```

Add validation methods for expanded IDs `[B,S]`, embeddings `[B,S,H]`,
mask/label shape, non-overlapping spans, full coverage of valid tokens, and
position shape `[axes,B,S]`, integer dtype and non-negative valid entries.
`PositionBatch` must not require each axis to be monotonic: image width IDs
naturally reset per row and later timestamped spans may reuse a temporal ID.
Only the one-axis legacy builder applies monotonic sequence validation.

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
- Produces: `DecodedMedia`, `StrictMediaLoader.load(item)`, `ProfileStage2Collator`, and a variable-item `media` batch field.

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
    timeline_offset_seconds: float = 0.0
    timestamps: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DecodedMedia:
    request: MediaRequest
    tensor: torch.Tensor
    length: int
    timestamps: torch.Tensor | None
    metadata: Mapping[str, object]
```

Rules:

- collator assigns `sample_index` from batch row and `item_index` from the
  canonical media-list order; both are preserved through `DecodedMedia` and
  copied exactly into encoder-produced `MediaSource`;
- `timeline_offset_seconds` is finite/non-negative and defines the item's
  origin on the sample-global AV timeline; encoder token timestamps are
  `timeline_offset_seconds + local_time`;
- image returns RGB float `[3,H,W]` and original dimensions in metadata;
- audio returns mono float `[1,samples]`, configured sample rate, true length,
  and no right padding;
- video returns `[frames,3,H,W]` plus exact per-frame seconds;
- `ffprobe` discovers dimensions/time base and `ffmpeg` decodes raw RGB frames;
- subprocess arguments are a list with `shell=False`;
- missing binaries, non-zero status, empty output and timestamp/frame mismatch
  become `MediaLoadError`;
- quarantine returns no fake tensor; it returns a structured error record that
  the collator uses to remove the entire sample atomically. If any referenced
  item in a row fails, remove that row from input IDs, masks, labels,
  `_sample_ids`, and every decoded-media collection; never leave its sentinel
  in a retained text row. If all rows are quarantined, reuse the existing
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
- a mixed good/bad two-row batch keeps the good row with all tensors and media
  reindexed to `sample_index=0`, drops the bad row atomically, and records the
  original bad sample ID/path/modality/error;
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

Expected: all tests pass; FFmpeg-dependent test skips only when the binary is
absent.

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

Pad images only to the next patch multiple, create a valid-patch mask from
original sizes, and preserve row-major patch order. `TemporalVideoEncoder` is a
separate `nn.Module` that owns only video-specific temporal projection/norm
parameters; its `forward()` receives the shared `PatchVisionEncoder` explicitly
and must not store the same patch encoder as a child-module alias. It applies
that encoder frame-by-frame, flattens temporal-major then row-major order, and
expands each frame timestamp across its spatial patches. Consequently one
registered patch encoder and every video-specific parameter appear exactly once
in `state_dict()` and `named_parameters()`.

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
        windows = waveform.unfold(0, self.window_size, self.hop_size)
        return self.norm(self.proj(windows))
```

Right-pad only the final partial window while masking it according to true
length. Timestamp each token at its window start in seconds. Empty waveform is
an error; omitted media never calls the encoder. Public batch helpers require
one `MediaSource` per waveform and copy those sources unchanged into the
returned `MediaSequence`.

- [ ] **Step 5: Add gradient and ordering tests**

Assert:

- image quadrant swaps change the corresponding patch tokens;
- video frame swaps change temporal order;
- audio tone-order swaps change token order;
- padded areas do not become valid tokens;
- outputs and gradients remain finite in FP32/BF16;
- no encoder collapses valid input to one token unless the input genuinely
  contains one patch/window.

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
- Create: `src/qwen3_omni_pretrain/multimodal/sequence_assembler.py`
- Create: `tests/multimodal/test_sequence_assembler.py`

**Interfaces:**
- Consumes: text embeddings, masks, labels, resolved token schema, optional
  sample-level `MediaExpansionPolicy`, an optional non-owning
  `EmbeddingLookup`, and `Sequence[MediaSequence]`.
- Produces: `MediaPlaceholder`, `ExpandedMediaRow`,
  `ExpandedMediaSample`, `MediaExpansionPolicy`, `IdentityMediaExpansion`,
  and `SequenceAssembler.assemble(...) -> AssembledSequence`.

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
        max_assembled_length: int,
    ) -> AssembledSequence:
        ...
```

Define the profile extension point:

```python
@dataclass(frozen=True)
class ExpandedMediaRow:
    token_ids: torch.LongTensor
    embeddings: torch.Tensor
    spans: tuple[SequenceSpan, ...]


@dataclass(frozen=True)
class MediaPlaceholder:
    text_position: int
    sentinel_token_id: int
    modality: MediaModality
    sequence: MediaSequence
    sequence_row: int


@dataclass(frozen=True)
class ExpandedMediaSample:
    replacements: Mapping[int, ExpandedMediaRow]
    consumed_sources: tuple[MediaSource, ...]


class EmbeddingLookup(Protocol):
    def __call__(self, token_ids: torch.LongTensor) -> torch.Tensor:
        ...


class MediaExpansionPolicy(Protocol):
    def expand_sample(
        self,
        *,
        sample_index: int,
        placeholders: tuple[MediaPlaceholder, ...],
        embedding_lookup: EmbeddingLookup | None,
    ) -> ExpandedMediaSample:
        ...
```

`IdentityMediaExpansion` returns one replacement per placeholder, repeats that
placeholder's sentinel ID once per valid media embedding and preserves the
source unchanged. Later profiles may insert ordinary timestamp text tokens or
jointly time-sort adjacent AV placeholders. The policy is a pure, non-module
object: it stores token IDs/config only and receives `embedding_lookup` for the
duration of the call, so it never owns or aliases the language-model embedding
module. It may not bypass source-span or label validation.

For each batch row:

1. ignore padded text positions;
2. map media sources by `(sample_index, item_index)`;
3. collect all validated placeholders for that sample and call
   `expand_sample()` exactly once;
4. verify the result has exactly the placeholder-position keys (a joint policy
   may assign a zero-length replacement to later placeholders), and that
   `consumed_sources` is an exact permutation of the input sources;
5. walk text tokens left-to-right and copy ordinary text embeddings;
6. splice the precomputed replacement for each sentinel;
7. emit text/media/timestamp `SequenceSpan` objects with exact source
   provenance and mask every inserted label with `-100`;
8. right-pad assembled rows to the batch maximum.

A policy that combines AV items may emit their jointly sorted sequence at the
first placeholder and zero tokens at the remaining placeholders only when the
placeholder group is adjacent apart from unsupervised wrapper/separator tokens.
It must fail if supervised text lies between them. This prevents reordering
media across natural-language causal content.

Do not encode media and do not compute position IDs.
Validate `max_assembled_length > 0` before row assembly and fail before batch
padding as soon as a row's expanded token count exceeds it. The error records
sample index and separate text/media counts.

- [ ] **Step 4: Fail on every ambiguous mapping**

Add exact errors for:

- sentinel with no media item;
- extra media item;
- modality mismatch;
- duplicated `(sample_index, item_index)`;
- source batch index outside the text batch;
- hidden-size mismatch;
- all-masked media sequence;
- assembled length above the caller-provided hard limit.

- [ ] **Step 5: Test multiple items and mixed batch lengths**

Cover:

- image→text→audio order;
- two items of the same modality;
- image and video in one sample;
- no-media sample sharing a batch with media samples;
- inference with `labels=None`;
- text padding and target labels after expansion;
- exact source span start/end positions.
- expanded token IDs for every media position;
- a custom sample policy that inserts two ordinary timestamp-token embeddings
  through the supplied lookup and preserves source-span coverage;
- a joint policy fixture with timestamps
  `[audio0=0.00, video0=0.08, audio1=0.16, video1=0.24]` whose assembled media
  spans are exactly audio→video→audio→video;
- the same joint expansion rejects an intervening supervised text token,
  duplicated/missing consumed source, or an unknown replacement key.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal/test_sequence_assembler.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/multimodal/sequence_assembler.py \
  tests/multimodal/test_sequence_assembler.py
git commit -m "feat: assemble multimodal token sequences"
```

---

### Task 6: Build legacy and TM-RoPE positions from spans

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/positions.py`
- Create: `tests/multimodal/test_position_builders.py`
- Create: `tests/oracle/test_qwen3_tm_rope_oracle.py`

**Interfaces:**
- Consumes: `AssembledSequence`.
- Produces: `PositionBuilder` protocol, `LegacyPositionBuilder`, `TMRoPEConfig`, and `TMRoPEPositionBuilder`.

- [ ] **Step 1: Write failing text/image/audio position tests**

```python
def test_qwen3_tm_rope_aligns_audio_to_80ms_grid():
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


def test_image_uses_constant_time_and_distinct_grid_axes():
    assembled = assembled_image_example(grid=MediaGrid(1, 2, 2))
    positions = qwen3_builder().build(assembled).position_ids[:, 0]
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
class TMRoPEConfig:
    temporal_seconds_per_id: float
    rotary_sections: tuple[int, int, int]
    interleaved: bool = True

    def __post_init__(self) -> None:
        if self.temporal_seconds_per_id <= 0:
            raise ValueError("temporal_seconds_per_id must be positive")
        if any(section <= 0 for section in self.rotary_sections):
            raise ValueError("rotary sections must be positive")
```

`LegacyPositionBuilder` returns `[1,B,S]`, axis `("sequence",)`.
`TMRoPEPositionBuilder` returns `[3,B,S]`, axes
`("temporal","height","width")`.

- [ ] **Step 4: Implement a sample-global media timeline and text continuity**

For each sample, start `text_cursor=0`, `timeline_anchor=None`, and
`max_temporal_seen=-1`:

- text span: all axes use
  `max(text_cursor, max_temporal_seen + 1) + arange(length)`, then advance
  `text_cursor`;
- on the first timestamped audio/video span, set `timeline_anchor` to the
  current `text_cursor`;
- timestamped audio: round each sample-global timestamp by the configured
  temporal grid and add `timeline_anchor`; use the same IDs on T/H/W;
- image without timestamps: temporal is the current `text_cursor`; H/W are
  row/column grid IDs plus that cursor;
- timestamped video: temporal is
  `timeline_anchor + round(global_frame_seconds / grid)` expanded across
  spatial patches; H/W repeat per frame;
- update `max_temporal_seen` from every media span, but do not add a new
  per-modality timeline origin. Two audio/video tokens at the same global
  timestamp therefore receive the same T ID even if their spans are separate;
- spatial axes may reset and a later media span may reuse an earlier temporal
  ID; causal order remains the assembled token order, not numeric position
  monotonicity;
- padded output positions are zero and ignored by the attention mask;
- `rope_deltas` is `max_position + 1 - valid_sequence_length` per row.

Reject non-finite/negative timestamps, timestamps that are non-monotonic
within one source after rounding, and grids whose token count disagrees with
the span. Add a fixture where audio and video at global `0.32s` share their
temporal ID while their spatial IDs remain modality-specific.

- [ ] **Step 5: Compare Qwen3 builder with official `get_rope_index()`**

In the reference environment, use the deterministic processor fixture from the
oracle plan. The processor supplies token/media/grid inputs only; invoke the
pinned official model class's `get_rope_index()` through the same
configuration-only facade defined by the oracle plan. Convert the inputs into
`AssembledSequence` metadata and compare valid Qwen3 position IDs and rope
deltas exactly. Keep this test `@pytest.mark.reference`; it must not construct
model layers or load weights.

- [ ] **Step 6: Run prototype and oracle tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider \
  tests/multimodal/test_position_builders.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/test_qwen3_tm_rope_oracle.py -q
```

Expected: both suites pass, or the oracle test explicitly skips when the
processor artifact is not cached.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/multimodal/positions.py \
  tests/multimodal/test_position_builders.py \
  tests/oracle/test_qwen3_tm_rope_oracle.py
git commit -m "feat: build time-aligned multimodal positions"
```

---

### Task 7: Expose a reusable multimodal prefill pipeline

**Files:**
- Create: `src/qwen3_omni_pretrain/multimodal/prefill.py`
- Create: `tests/multimodal/test_multimodal_prefill.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: token schema, decoded items, media encoders, `SequenceAssembler`, and `PositionBuilder`.
- Produces: `MultimodalPrefillPipeline.encode_and_assemble(...) -> MultimodalPrefillOutput`.

- [ ] **Step 1: Write a failing end-to-end prefill test**

```python
def test_prefill_keeps_image_and_audio_sequences_in_prompt_order():
    pipeline = tiny_prefill_pipeline()
    output = pipeline.encode_and_assemble(
        input_ids=torch.tensor([[5, 10, 6, 12, 7]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        labels=torch.tensor([[-100, -100, -100, -100, 7]]),
        decoded_media=(
            tiny_image_item(source_id="image-0"),
            tiny_audio_item(source_id="audio-0"),
        ),
        text_embedding=nn.Embedding(32, 8),
    )
    assert output.assembled.inputs_embeds.shape[1] > 5
    assert [span.modality for span in output.assembled.spans if span.modality] == [
        MediaModality.IMAGE,
        MediaModality.AUDIO,
    ]
    assert output.positions.position_ids.shape[:2] == (3, 1)
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
        max_assembled_length: int,
    ) -> None:
        super().__init__()
        if any(
            isinstance(component, nn.Module)
            for component in (assembler, expansion_policy, position_builder)
        ):
            raise TypeError(
                "assembler, expansion policy and position builder "
                "must be parameter-free non-modules"
            )
        self.image_encoder = image_encoder
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        if max_assembled_length <= 0:
            raise ValueError("max_assembled_length must be positive")
        self.tokens = tokens
        self.assembler = assembler
        self.expansion_policy = expansion_policy
        self.position_builder = position_builder
        self.max_assembled_length = max_assembled_length
```

`encode_and_assemble()` groups decoded items by modality, calls each encoder
once (passing the registered image patch encoder into the video encoder), calls
`assemble(expansion_policy=self.expansion_policy,
embedding_lookup=text_embedding, max_assembled_length=self.max_assembled_length)`,
builds
positions, validates the three outputs, and returns.
It does not call a language model and does not own request cache.
Every factory supplies the policy explicitly: Qwen3/MiMo use
`IdentityMediaExpansion`; Qwen3.5 injects its timestamp policy. The pipeline
must not instantiate an implicit policy based on profile-name conditionals.
Because the pipeline is an `nn.Module`, all three encoder modules are registered
for `.to()`, checkpointing and optimization. `text_embedding` remains an
argument owned by the language model and is not assigned to the pipeline.

- [ ] **Step 4: Add ablation and ordering tests**

Test:

- shuffling the transport container while preserving each
  `(sample_index,item_index)` key leaves assembled output unchanged;
- a deliberate ablation that swaps payload-to-source association changes
  embeddings while spans continue to identify the declared sources;
- omitted media leaves no media span;
- duplicate source IDs fail;
- assembled hard limit reports text/media token counts;
- gradients flow to selected encoder/projector and text embedding;
- `state_dict()` contains image, video-specific and audio parameters exactly
  once, and `.to(dtype=...)` moves all registered encoder parameters;
- media labels remain ignored;
- Qwen3 and Qwen3.5 temporal configs produce different temporal IDs for the
  same 160 ms timestamps.
- a sample-level timestamp policy produces true cross-modal
  audio0→video0→audio1→video1 token order, not merely equal temporal position
  IDs on modality-contiguous blocks.

- [ ] **Step 5: Document the profile boundary**

Update README:

- legacy still uses one-token prepend and is unchanged;
- new experimental profiles must use `MultimodalPrefillPipeline`;
- Qwen3 reference uses the official processor/model as its numerical oracle;
- passing these common tests means sequence semantics are implemented, not
  official checkpoint compatibility.

- [ ] **Step 6: Run focused and full suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider tests/multimodal -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q
```

Expected: all focused and full-suite tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/qwen3_omni_pretrain/multimodal/prefill.py \
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
  tests/oracle/test_qwen3_tm_rope_oracle.py -q

git diff --check
git status --short
```

The plan is complete only when:

- image, audio and video retain more than one token when input structure
  contains more than one patch/window;
- multiple media items replace sentinels in exact conversation order;
- no omitted/corrupt media is silently replaced by a learned or zero summary;
- media labels are `-100` and text supervision survives expansion;
- Qwen3 80 ms TM-RoPE positions agree with the processor oracle;
- Qwen3.5 160 ms positions use the same contract without an official
  compatibility claim;
- legacy behavior and the full prototype suite remain green;
- all changes are intentional and committed.
