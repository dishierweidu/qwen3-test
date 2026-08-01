# Qwen3.5-Omni-Inspired Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将公开 Qwen3.5 Hybrid-MoE backbone 与论文披露的 6.25 Hz AuT、显式 timestamp、TM-RoPE、ARIA 以及前代 codec proxy 组合成可测试的 `qwen35_omni_inspired` 原型。

**Implementation status (2026-08-01):** Complete and regression-tested. Per
the final delivery instruction, the task-level commit commands below are
historical execution notes; all changes are delivered in one consolidated
commit.

**Architecture:** 文本 backbone 直接采用固定 Transformers 5.2.0 的公开 Qwen3.5 MoE 实现，公共 cache adapter 将其 full-attention KV、GDN convolution 和 rank-4 recurrent matrix state 转换为 `DecoderState`；由于公开 native cache 使用 batch-shared cache position，cached batch 只接受等长且无 padding 的行，异构长度在 adapter 外分桶或逐行执行。媒体和时间使用公共 `MediaSequence`/assembler/position contracts，AuT Transformer 明确是 offline-only。Talker 是独立 paper-inspired Hybrid-MoE 组件，ARIA 是无需预知未来长度、按固定整数速率目标与双 EOS 推进的在线状态机，MTP/Code2Wav 参数明确来自 Qwen3-Omni predecessor proxy。

**Tech Stack:** Python 3.10, PyTorch 2.10.0, TorchAudio 2.10.0, Transformers 5.2.0 `qwen3_5_moe`, qwen-omni-utils 0.0.9 for predecessor processing, pytest, Hypothesis-free deterministic property tests.

## Global Constraints

- 本计划依赖前四个计划全部完成。
- profile 固定为 `qwen35_omni_inspired`，compatibility 固定为 `paper-inspired`，`exact_official_checkpoint_compatible=false`。
- 类名、checkpoint 名、CLI 参数不得使用 `Qwen3.5-Omni-Plus` 或 `Qwen3.5-Omni-Flash`。
- 公开 backbone source 固定为 Qwen3.5-35B-A3B revision `59d61f3ce65a6d9863b86d2e96597125219dc754` 和 Transformers 5.2.0。
- 公开 backbone 的 248,320 vocab、3:1 GDN/full pattern、256 experts/top-8 是 backbone oracle，不是 Omni Plus/Flash 精确配置。
- GDN 必须保留 Q/K/V、beta、decay、z、depthwise causal convolution 和 recurrent matrix state；当前 legacy DeltaNet 不复用。
- 音频 frontend 固定为 16 kHz、128 Mel、25 ms window、10 ms hop、四个 stride-2 Conv2D block、6.25 Hz 输出。
- AuT hidden/layer/head 数、Talker hidden/layer/expert 数和 codec 参数中未公开的字段必须列入 manifest assumptions。
- 时间网格固定为 160 ms；显式 timestamp string 的格式属于可配置实验假设。
- ARIA 的目标 speech:text 速率使用正整数分子/分母与交叉乘法；状态中不得保存或接收未来总 token 数，text/speech 各自以 EOS 结束，padding/tie-break 语义必须在 config 和 tests 中固定。
- predecessor codec 固定为 Qwen3-Omni revision `26291f793822fb6be9555850f06dfe95f2d7e695` 并标记 `predecessor-codec-proxy`。
- AuT Transformer 只提供 offline encoding；仅 Mel/Conv frontend 可验证有 overlap 的分块等价，manifest assumption 必须写明 offline-only，`ArchitectureSummary.capabilities["streaming_audio_encoder"]` 必须为 `false`。
- native Qwen3.5 cached path 只接受所有行 `seen_tokens` 相等且 `key_valid_mask` 全真的 batch；异构长度必须在调用前 length-bucket 或逐行执行。
- optional fused kernel 只能替换已通过数值 parity 的公开/PyTorch 路径。
- 本计划不声称复现未公开训练数据、权重、论文质量或 API latency。
- Snippet 中的 `...` 只表示 Protocol/签名节选；任务提交不得保留未实现
  stub，必须由该任务列出的行为测试证明可用。
- 每个行为变更使用 red-green-refactor，并形成独立提交。

---

## File responsibility map

- `profiles/qwen35_omni_inspired/configuration.py`: profile config、assumptions 和公开 backbone 绑定。
- `profiles/qwen35_omni_inspired/cache_adapter.py`: `Qwen3_5MoeDynamicCache` 与公共 state 转换。
- `profiles/qwen35_omni_inspired/audio_encoder.py`: 6.25 Hz frontend/AuT prototype。
- `profiles/qwen35_omni_inspired/timestamp_alignment.py`: timestamp text expansion 和 160 ms positions。
- `profiles/qwen35_omni_inspired/thinker.py`: public backbone + common multimodal prefill。
- `profiles/qwen35_omni_inspired/aria.py`: 无未来 totals、固定整数速率目标与双 EOS 的在线 ARIA state machine。
- `profiles/qwen35_omni_inspired/talker.py`: paper-inspired Talker 和 predecessor codec proxy。
- `profiles/qwen35_omni_inspired/factory.py`: lazy profile construction and capability summary。

---

### Task 1: Define the paper-inspired profile and isolated dependency contract

**Files:**
- Create: `constraints/qwen35-backbone-py310.txt`
- Create: `requirements-qwen35-backbone.txt`
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/__init__.py`
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/configuration.py`
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py`
- Modify: `src/qwen3_omni_pretrain/profiles/registry.py`
- Create: `tests/qwen35/test_configuration.py`
- Modify: `tests/test_dependency_profiles.py`

**Interfaces:**
- Consumes: profile manifest and registry.
- Produces: `Qwen35AuTConfig`, `AriaConfig`, `CodecProxyConfig`, `Qwen35InspiredConfig`, `Qwen35InspiredFactory`.

- [ ] **Step 1: Pin the public-backbone environment**

`constraints/qwen35-backbone-py310.txt`:

```text
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
transformers==5.2.0
qwen-omni-utils==0.0.9
```

`requirements-qwen35-backbone.txt`:

```text
-r requirements.txt
-c constraints/qwen35-backbone-py310.txt
qwen-omni-utils==0.0.9
```

Add exact dependency-profile tests and `.venv-qwen35-backbone/` to
`.gitignore`. Bootstrap the isolated environment reproducibly from the
repository root (never install this profile's pins into `.venv-prototype`):

```bash
python3.10 -m venv .venv-qwen35-backbone
.venv-qwen35-backbone/bin/python -m pip install --upgrade "pip==25.1.1"
.venv-qwen35-backbone/bin/python -m pip install \
  -r requirements-qwen35-backbone.txt
.venv-qwen35-backbone/bin/python -m pip freeze \
  > .venv-qwen35-backbone/installed-freeze.txt
.venv-qwen35-backbone/bin/python - <<'PY'
from importlib.metadata import version

import torch
import transformers

assert torch.__version__.split("+")[0] == "2.10.0"
assert transformers.__version__ == "5.2.0"
assert version("qwen-omni-utils") == "0.0.9"
PY
```

`installed-freeze.txt` is diagnostic output inside the ignored venv and is
not committed. `tests/test_dependency_profiles.py` parses both constraint and
requirements files and requires exactly the versions above, including
`qwen-omni-utils==0.0.9`.

- [ ] **Step 2: Write failing provenance tests**

```python
def test_qwen35_profile_cannot_claim_official_omni_compatibility():
    config = tiny_qwen35_config()
    assert config.model_type == "qwen35_omni_inspired"
    assert config.profile_manifest.compatibility_level.value == "paper-inspired"
    assert not config.profile_manifest.exact_official_checkpoint_compatible
    raw = tiny_qwen35_config_dict()
    raw["profile_manifest"] = checkpoint_compatible_manifest().to_dict()
    with pytest.raises(ValueError, match="checkpoint"):
        Qwen35InspiredConfig(**raw)


def test_predecessor_codec_proxy_is_mandatory():
    raw = tiny_qwen35_config_dict()
    raw["codec_proxy"] = {}
    with pytest.raises(ValueError, match="predecessor"):
        Qwen35InspiredConfig(**raw)
```

- [ ] **Step 3: Run and observe the missing profile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_configuration.py -q
```

Expected: FAIL at import.

- [ ] **Step 4: Implement fixed frontend and explicit assumption configs**

```python
@dataclass(frozen=True)
class Qwen35AuTConfig:
    sample_rate: int = 16_000
    num_mel_bins: int = 128
    window_ms: float = 25.0
    hop_ms: float = 10.0
    temporal_downsample: int = 16
    output_frame_hz: float = 6.25
    hidden_size: int = 512
    encoder_layers: int = 8
    attention_heads: int = 8
    intermediate_size: int = 2048


@dataclass(frozen=True)
class AriaConfig:
    speech_tokens_per_text_num: int = 12
    speech_tokens_per_text_den: int = 5
    text_first: bool = True
    tie_break: str = "text"
    count_eos: bool = False
    count_padding: bool = False


@dataclass(frozen=True)
class CodecProxyConfig:
    source_model: str
    source_revision: str
    provenance_label: str
```

The AuT model dimensions above are defaults for the prototype and must appear
in `assumptions`; only frontend/rate fields are paper-explicit.
`speech_tokens_per_text_num/den` are positive prototype assumptions and encode
the online target speech:text rate without future sequence totals.

- [ ] **Step 5: Implement the profile config**

```python
class Qwen35InspiredConfig(PretrainedConfig):
    model_type = "qwen35_omni_inspired"

    def __init__(
        self,
        *,
        backbone_config: Mapping[str, object],
        audio_config: Mapping[str, object],
        aria_config: Mapping[str, object],
        codec_proxy: Mapping[str, object],
        source_revision: str,
        timestamp_format: str = "[{seconds:.2f}s]",
        **kwargs: object,
    ) -> None:
        ...
```

Validation requires:

- exact public source revision;
- backbone `model_type` is Qwen3.5 MoE;
- layer types follow configured 3:1 pattern for production config;
- timestamp format contains `{seconds`;
- codec source/revision/provenance exactly identify predecessor proxy;
- manifest lists every non-paper architecture default as an assumption;
- manifest assumptions 包含 AuT offline-only，factory 生成的完整
  `ArchitectureSummary.capabilities` 包含
  `streaming_audio_encoder=false`;
- ARIA rate numerator/denominator are plain positive integers, and
  `tie_break` is exactly `text` or `speech`.

- [ ] **Step 6: Register lazily and run tests**

Register the factory by module path. The prototype environment must be able to
list the profile without importing Transformers 5.2 Qwen3.5 classes. At this
task boundary its `build()` is already usable: it returns a
`ProfileBuildResult` whose artifact is the validated immutable config contract
below, not an error-raising partial model. Task 5 replaces the artifact with
the complete trainable Thinker.

```python
@dataclass(frozen=True)
class Qwen35ConfigArtifact:
    config: Qwen35InspiredConfig
    public_backbone_revision: str
```

The result's `architecture_summary` is a strict config-contract summary with
zero allocated parameters, declared layer types and
`unsupported_capabilities=("model_runtime",)`. It is produced by a dedicated
`summarize_qwen35_config_contract()` function, not by passing this non-module
artifact to `summarize_model()`. A test calls `build()` and round-trips the
manifest/summary, so the initial registry entry is usable at this commit.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider \
  tests/qwen35/test_configuration.py tests/test_dependency_profiles.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add constraints/qwen35-backbone-py310.txt \
  requirements-qwen35-backbone.txt .gitignore \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/__init__.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/configuration.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py \
  src/qwen3_omni_pretrain/profiles/registry.py \
  tests/qwen35/test_configuration.py tests/test_dependency_profiles.py
git commit -m "feat: define the Qwen3.5 inspired profile"
```

---

### Task 2: Adapt the public Qwen3.5 GDN and cache without semantic rewrites

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/cache_adapter.py`
- Create: `tests/oracle/test_qwen35_public_config.py`
- Create: `tests/oracle/test_qwen35_gdn_oracle.py`
- Create: `tests/qwen35/test_cache_adapter.py`

**Interfaces:**
- Consumes: `Qwen3_5MoeDynamicCache`, public backbone config, and common `DecoderState`.
- Produces:
  `Qwen35CacheAdapter.to_native(state, *, request_id) -> Qwen3_5MoeDynamicCache`
  and
  `Qwen35CacheAdapter.from_native(native, *, request_id, seen_tokens,
  key_valid_mask, position_ids) -> DecoderState`.

- [ ] **Step 1: Write failing state round-trip tests**

```python
def test_gdn_state_round_trip_preserves_conv_and_matrix_tensors():
    config = tiny_public_qwen35_config()
    native = Qwen3_5MoeDynamicCache(config)
    native.conv_states[0] = torch.randn(
        expected_gdn_conv_shape(config, batch_size=1)
    )
    native.recurrent_states[0] = torch.randn(
        expected_gdn_recurrent_shape(config, batch_size=1)
    )
    assert native.recurrent_states[0].ndim == 4  # [B,H,K,V]
    common = Qwen35CacheAdapter(config).from_native(
        native,
        request_id="r1",
        seen_tokens=torch.tensor([5], dtype=torch.long),
        key_valid_mask=torch.ones(1, 5, dtype=torch.bool),
        position_ids=torch.arange(5).view(1, 5),
    )
    restored = Qwen35CacheAdapter(config).to_native(common, request_id="r1")
    torch.testing.assert_close(
        restored.conv_states[0],
        native.conv_states[0],
    )
    torch.testing.assert_close(
        restored.recurrent_states[0],
        native.recurrent_states[0],
    )


def test_cached_heterogeneous_or_padded_batch_fails_before_native_call():
    config = tiny_public_qwen35_config()
    state = common_qwen35_state(
        seen_tokens=torch.tensor([4, 2]),
        key_valid_mask=torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        ),
    )
    with pytest.raises(
        ValueError,
        match="equal-length.*without padding",
    ):
        Qwen35CacheAdapter(config).to_native(state, request_id="r1")


def test_wrong_rank_recurrent_state_is_rejected():
    config = tiny_public_qwen35_config()
    native = Qwen3_5MoeDynamicCache(config)
    native.recurrent_states[0] = torch.zeros(1, 4, 4)
    with pytest.raises(ValueError, match="rank-4"):
        Qwen35CacheAdapter(config).from_native(
            native,
            request_id="r1",
            seen_tokens=torch.tensor([1]),
            key_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            position_ids=torch.zeros(1, 1, dtype=torch.long),
        )
```

- [ ] **Step 2: Run and observe missing adapter failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_cache_adapter.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement exact native/common conversion**

For every layer:

- full attention native `key_cache/value_cache` becomes `AttentionKV`;
- `from_native()` requires explicit boolean `[B,S] key_valid_mask` and
  `[B,S]`/`[A,B,S] position_ids` from the just-completed model call rather
  than inventing either from tensor shape, and stores both in every
  `AttentionKV`;
- GDN `conv_states` becomes `gdn_convolution_state[layer]`;
- GDN `recurrent_states` becomes
  `gdn_recurrent_matrix_state[layer]`;
- absent native entries remain absent, not zero-filled;
- tensors are cloned when entering/leaving native mutable cache;
- `request_id` ownership is checked;
- rank-1 `[B] seen_tokens`, mask batch/sequence shape, dtype and per-row valid
  counts are checked;
- GDN convolution shapes and rank-4 recurrent shapes `[B,H,K,V]` are derived
  from the public config/module attributes and validated without flattening;
- layer type and expected tensor ranks are validated against config.

The adapter must not reshape or truncate recurrent matrices. Before any
`use_cache=True` native call, `to_native()` requires every row's
`seen_tokens` to be equal and every element of `key_valid_mask` to be true.
This is a deliberate compatibility boundary: the public native GDN cache uses
a batch-shared cache position and does not expose a semantics-preserving
per-row masked recurrent update. Heterogeneous cached requests must be
length-bucketed or executed row-by-row by the caller. The adapter raises
before invoking the native model; it never claims that right-padding leaves
individual recurrent states unchanged.

- [ ] **Step 4: Verify the public GDN implementation itself**

Build a tiny public config and test the official
`Qwen3_5MoeGatedDeltaNet` directly:

- Q/K/V, beta, decay and z projections exist;
- depthwise conv and rank-4 recurrent matrix shapes match config;
- full-sequence and token-by-token outputs match for an equal-length,
  padding-free batch;
- state size is constant with processed length;
- forward and parameter gradients are finite.

This is the numerical implementation used by the profile; do not compare it to
legacy `GatedDeltaNetAttention`. Do not add a direct-public-model test that
asserts per-row recurrent state preservation under right padding: that is not
an official cache guarantee. The heterogeneous-padding behavior belongs to
the adapter fail-fast test above.

- [ ] **Step 5: Pin the public config oracle**

Read Qwen3.5-35B-A3B config at the fixed revision and assert:

- vocab 248,320;
- context 262,144;
- 3 linear-attention layers followed by one full-attention layer;
- 256 routed experts, top-8;
- Q/K norm, attention output gate and SwiGLU fields.

No 35B weights are required.

- [ ] **Step 6: Run oracle and adapter tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/test_qwen35_public_config.py \
  tests/oracle/test_qwen35_gdn_oracle.py \
  tests/qwen35/test_cache_adapter.py -q
```

Expected: all tests pass with FP32 `max_abs <= 1e-5`.

- [ ] **Step 7: Commit**

```bash
git add \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/cache_adapter.py \
  tests/oracle/test_qwen35_public_config.py \
  tests/oracle/test_qwen35_gdn_oracle.py \
  tests/qwen35/test_cache_adapter.py
git commit -m "feat: adapt the public Qwen3.5 GDN cache"
```

---

### Task 3: Implement the offline-only 6.25 Hz AuT input path

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/audio_encoder.py`
- Create: `tests/qwen35/test_audio_encoder.py`

**Interfaces:**
- Consumes: raw 16 kHz waveform batches and `Qwen35AuTConfig`.
- Produces: offline-only `Qwen35AuTEncoder.forward() -> MediaSequence` and
  `Qwen35MelFrontend.push_chunk()` for frontend parity only.

- [ ] **Step 1: Write failing rate and padding tests**

```python
def test_two_seconds_produces_about_twelve_or_thirteen_tokens():
    encoder = tiny_qwen35_audio_encoder().eval()
    waveform = torch.zeros(1, 32_000)
    output = encoder(
        waveform,
        lengths=torch.tensor([32_000]),
        sources=(MediaSource(0, 0, "audio-0"),),
    )
    assert int(output.attention_mask.sum()) in {12, 13}
    valid = output.timestamps[0, output.attention_mask[0].bool()]
    torch.testing.assert_close(
        torch.diff(valid),
        torch.full_like(valid[1:], 0.16),
        atol=1e-6,
        rtol=0,
    )


def test_padded_audio_does_not_create_valid_tokens():
    encoder = tiny_qwen35_audio_encoder().eval()
    output = encoder(
        torch.zeros(2, 32_000),
        lengths=torch.tensor([16_000, 32_000]),
        sources=audio_sources(2),
    )
    assert output.attention_mask[0].sum() < output.attention_mask[1].sum()
```

- [ ] **Step 2: Run and observe missing encoder failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_audio_encoder.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement the fixed frontend**

```python
class Qwen35MelFrontend(nn.Module):
    def __init__(self, config: Qwen35AuTConfig) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=128,
            center=False,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.mel(waveform)
        return torch.log(features.clamp_min(1e-10))
```

Add four Conv2D blocks with temporal stride 2, then flatten frequency/channel
into a projection to configured hidden size. Propagate exact valid lengths
through each convolution formula; do not infer them from zero values.
`Qwen35MelFrontend.push_chunk()` retains the exact waveform overlap required
by `center=False`, emits only newly stable Mel/Conv frames, and owns no
Transformer state.

- [ ] **Step 4: Implement the prototype AuT encoder**

Use non-causal `nn.TransformerEncoder` with the explicit experimental
dimensions from `Qwen35AuTConfig`. It always consumes the complete valid
utterance and is intentionally offline-only; this task does not expose
incremental AuT state or claim streaming audio encoding. Return:

- projected embeddings `[B,M,H_backbone]`;
- boolean valid mask;
- `MediaModality.AUDIO`;
- sources;
- timestamps `arange(M) / 6.25`.

The projector to backbone hidden size is a separate named module.

Expose the common media interface through a registered adapter:

```python
class Qwen35AudioSequenceAdapter(nn.Module):
    def __init__(self, encoder: Qwen35AuTEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        items: tuple[DecodedMedia, ...],
    ) -> MediaSequence:
        """Validate audio items, pad waveforms, and preserve typed sources."""
        ...
```

`Qwen35AuTEncoder.forward(waveform, lengths, sources)` remains the numerical
kernel used by focused audio tests. `Qwen35AudioSequenceAdapter.forward(items)`
is the only object installed as
`MultimodalPrefillPipeline.audio_encoder`; it validates every item is audio,
derives waveform lengths from decoded tensors, preserves
`sample_index`/`item_index` provenance as `MediaSource`, pads once, and returns
the kernel's `MediaSequence`. No pipeline call uses profile-specific waveform
kwargs.

- [ ] **Step 5: Add offline/chunk boundary tests**

Test:

- duration-to-token rounding at 160 ms boundaries;
- 1 s, 2 s and mixed-length batches;
- valid output unchanged by right padding;
- monotonic timestamps;
- finite forward/backward;
- chunked Mel/Conv frontend with retained waveform overlap matches offline
  valid frames;
- no `Qwen35AuTEncoder.push_chunk` or streaming-state API is exported;
- the common pipeline calls `Qwen35AudioSequenceAdapter.forward(items)` and
  produces the same valid embeddings/mask/timestamps as the raw batched
  kernel, including two items from different sample rows;
- profile manifest/capability summary reports
  `streaming_audio_encoder=false`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_audio_encoder.py -q
```

Expected: all tests pass.

```bash
git add \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/audio_encoder.py \
  tests/qwen35/test_audio_encoder.py
git commit -m "feat: encode Qwen3.5 inspired audio at 6.25 Hz"
```

---

### Task 4: Insert explicit timestamps and build 160 ms TM-RoPE

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/timestamp_alignment.py`
- Create: `tests/qwen35/test_timestamp_alignment.py`

**Interfaces:**
- Consumes: the common sample-aware `MediaExpansionPolicy`,
  `MediaSequence.source.sample_index/item_index`, profile tokenizer/embedding,
  and `TMRoPEPositionBuilder`.
- Produces: `Qwen35TimestampExpansionPolicy` and
  `build_qwen35_position_builder()`.

- [ ] **Step 1: Write a failing timestamp insertion test**

```python
def test_each_temporal_audio_unit_gets_a_timestamp_prefix():
    embedding = nn.Embedding(64, 8)
    policy = Qwen35TimestampExpansionPolicy(
        tokenizer=FakeTimestampTokenizer(),
        timestamp_format="[{seconds:.2f}s]",
    )
    sequence = audio_sequence(
        timestamps=torch.tensor([[0.00, 0.16, 0.32]])
    )
    expanded_sample = policy.expand_sample(
        sample_index=0,
        placeholders=(
            MediaPlaceholder(
                text_position=1,
                sentinel_token_id=12,
                modality=MediaModality.AUDIO,
                sequence=sequence,
                sequence_row=0,
            ),
        ),
        embedding_lookup=embedding,
    )
    expanded = expanded_sample.replacements[1]
    assert expanded.token_ids.tolist() == [
        40, 12,
        41, 12,
        42, 12,
    ]
    assert len(expanded.spans) == 6


def test_audio_and_video_are_interleaved_per_sample_timeline():
    prompt = two_row_av_prompt()
    embedding = nn.Embedding(128, 8)
    assembled = common_sequence_assembler().assemble(
        input_ids=prompt,
        text_embeddings=embedding(prompt),
        attention_mask=torch.ones_like(prompt, dtype=torch.bool),
        labels=None,
        media_sequences=sample_scoped_audio_and_video_sequences(),
        tokens=tiny_resolved_multimodal_tokens(),
        expansion_policy=tiny_qwen35_timestamp_policy(),
        embedding_lookup=embedding,
        max_assembled_length=128,
    )
    assert timeline_modalities(assembled, sample_index=0) == [
        "audio", "video", "audio", "video"
    ]
    assert timeline_modalities(assembled, sample_index=1) == [
        "video", "audio"
    ]
```

- [ ] **Step 2: Run and observe missing policy failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_timestamp_alignment.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement explicit timestamp expansion**

For each temporal unit across all placeholders in one sample:

1. format seconds with the configured template;
2. tokenize without automatic special tokens;
3. embed all resulting ordinary text tokens;
4. append that unit's media patch/audio embeddings;
5. emit separate timestamp-text and media spans;
6. preserve source ID and timestamp on the media span.

Image gets one timestamp only when profile config explicitly requests it.
Video gets one timestamp before each frame's spatial patch group. Pure-audio
random timestamp insertion is a training augmentation in the stage plan, not
an inference default.

`expand_sample()` receives every already-validated placeholder for exactly one
sample. For each adjacent audio/video placeholder group it expands temporal
units, then stable-sorts them by `(global_timestamp, item_index,
unit_local_index, modality_tie_break)`. It emits the joint sequence at the
first placeholder and explicit zero-length replacements at the remaining
placeholders, while returning every consumed `MediaSource` exactly once and
retaining a source span for every unit. It rejects supervised text between
joint placeholders through the common assembler contract. Image-only and
non-joint groups retain placeholder order. Sorting is sample-local, never
batch-global, so the transport container order is irrelevant and audio/video
tokens are genuinely interleaved rather than merely assigned matching
TM-RoPE IDs.

`embedding_lookup: Callable[[torch.LongTensor], torch.Tensor]` is supplied to
`expand_sample()` and is non-owning: the policy constructor stores no module or
callable closure over the backbone embedding. The policy is not an `nn.Module`
and cannot register a second module alias. Gradients still flow through the
call-scoped lookup to the single backbone-owned embedding parameters.

- [ ] **Step 4: Configure the 160 ms builder**

```python
def build_qwen35_position_builder() -> TMRoPEPositionBuilder:
    return TMRoPEPositionBuilder(
        TMRoPEConfig(
            temporal_seconds_per_id=0.16,
            rotary_sections=(24, 20, 20),
            interleaved=True,
        )
    )
```

Timestamp text tokens receive ordinary continuous 1D positions. Media
temporal IDs map to 160 ms. Cross-modal cursor continuity remains enforced by
the common builder.

- [ ] **Step 5: Test audio/video alignment**

Cover:

- audio at 0.32 s and a video frame at 0.32 s share temporal offset;
- dynamic-FPS video frame timestamps;
- timestamp format round-trip;
- multi-item media remains ordered independently in every batch sample;
- AV items are interleaved on each sample's common timeline even when the
  transport container is shuffled;
- an exact fixture at 0.00/0.08/0.16/0.24 seconds yields
  audio0→video0→audio1→video1 spans and a modality-contiguous implementation
  fails the test;
- changed timestamp format changes token IDs and
  `profile_manifest.canonical_sha256()`;
- no duplicate/conflicting position IDs across segment boundaries.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_timestamp_alignment.py -q
```

Expected: all tests pass.

```bash
git add \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/timestamp_alignment.py \
  tests/qwen35/test_timestamp_alignment.py
git commit -m "feat: align Qwen3.5 inspired timestamps"
```

---

### Task 5: Combine the public backbone with multimodal prefill

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/thinker.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py`
- Create: `tests/qwen35/test_thinker.py`
- Create: `tests/qwen35/test_profile_end_to_end.py`

**Interfaces:**
- Consumes: public `Qwen3_5MoeForCausalLM`, multimodal prefill pipeline,
  3-axis position IDs, and cache adapter.
- Produces: `Qwen35InspiredThinker.forward()` and stable
  `named_parameter_groups()`, plus typed `CacheCapableModel.prefill()` and
  `.decode()` serving methods.

- [ ] **Step 1: Write failing text-only backbone parity**

```python
def test_text_only_adapter_matches_public_backbone():
    public = tiny_public_qwen35_model().eval()
    thinker = Qwen35InspiredThinker.from_public_model(
        public,
        prefill_pipeline=tiny_qwen35_prefill_pipeline(public),
    ).eval()
    ids = torch.tensor([[3, 4, 5, 6]])
    expected = public(input_ids=ids, use_cache=False).logits
    actual = thinker(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        decoded_media=(),
        labels=None,
        request_id="r1",
        use_cache=False,
    )["logits"]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_factory_returns_profile_build_result():
    result = tiny_qwen35_factory().build(tiny_profile_build_request())
    assert isinstance(result, ProfileBuildResult)
    assert isinstance(result.artifact, Qwen35InspiredThinker)
    assert result.manifest.compatibility_level.value == "paper-inspired"
    assert result.architecture_summary.compatibility_level == "paper-inspired"
    assert (
        result.architecture_summary.capabilities["streaming_audio_encoder"]
        is False
    )
    assert (
        result.architecture_summary.capabilities["paper_inspired_aria"]
        is False
    )
```

- [ ] **Step 2: Run and observe missing Thinker failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_thinker.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement the adapter without changing public parameters**

```python
class Qwen35InspiredThinker(nn.Module):
    def __init__(
        self,
        *,
        backbone: Qwen3_5MoeForCausalLM,
        prefill_pipeline: MultimodalPrefillPipeline,
        cache_adapter: Qwen35CacheAdapter,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.prefill_pipeline = prefill_pipeline
        self.cache_adapter = cache_adapter

    def named_parameter_groups(
        self,
    ) -> Mapping[str, tuple[nn.Parameter, ...]]:
        groups = {
            "thinker": unique_parameters_by_identity(
                self.backbone.parameters()
            ),
            "vision_encoder": unique_parameters_by_identity(chain(
                non_projector_parameters(
                    self.prefill_pipeline.image_encoder
                ),
                non_projector_parameters(
                    self.prefill_pipeline.video_encoder
                ),
            )),
            "audio_encoder": unique_parameters_by_identity(
                non_projector_parameters(
                    self.prefill_pipeline.audio_encoder
                )
            ),
            "projector": unique_parameters_by_identity(chain(
                projector_parameters(self.prefill_pipeline.image_encoder),
                projector_parameters(self.prefill_pipeline.video_encoder),
                projector_parameters(self.prefill_pipeline.audio_encoder),
            )),
        }
        return assert_disjoint_complete_groups(self, groups)
```

`MultimodalPrefillPipeline` is the registered `nn.Module` supplied by the
common media plan, so its image, video and audio parameters are visible from
`Qwen35InspiredThinker.named_parameters()`. `non_projector_parameters()` and
`projector_parameters()` classify by registered submodule ownership;
`unique_parameters_by_identity()` preserves first-seen order and removes
shared image-patch parameters reached again through the video encoder.
Consequently `vision_encoder` contains every video-specific parameter as well
as the shared image core exactly once, and `projector` contains image, video
and audio projectors. The timestamp expansion policy is a non-owning callable
over `backbone.get_input_embeddings()` and must not register a second module
alias. The four tuples must be pairwise disjoint by parameter identity and
together cover every Thinker parameter exactly once. Use only the generic
group names owned by the stage-training plan; dotted profile-specific aliases
are forbidden.

Forward:

- text-only delegates unchanged;
- multimodal prefill supplies `inputs_embeds`, expanded mask and `[3,B,S]`
  positions;
- cached decode converts common state to/from native cache;
- cached decode passes the exact `key_valid_mask` to
  `Qwen35CacheAdapter.from_native()` together with the exact one-/three-axis
  positions, and fails before the public backbone call unless every cached row
  is equal-length and unpadded;
- outputs remain the existing mapping keys plus `decoder_state`;
- no `nan_to_num` is applied.

Expose the exact common keyword-only
`prefill(inputs=ModelPrefillInputs, request_id=..., use_cache=...)` and
`decode(token_ids=..., current_attention_mask=..., decoder_state=...,
request_id=...)` signatures as thin typed adapters returning
`CausalLMOutput`. Prefill performs media expansion
once; decode accepts no raw media and applies the equal-length/unpadded native
cache restriction. The generic generation/serving layers use these methods,
not arbitrary forward kwargs.

- [ ] **Step 4: Test 3:1 layers and recurrent cache**

Assert:

- public 3:1 GDN/full layer pattern;
- public Q/K norm, output gate, SwiGLU and MoE modules remain present;
- text-only parity;
- multimodal forward shape;
- cached/uncached text and multimodal next-token parity;
- GDN state size stays constant;
- media encoder runs once;
- uncached padding does not alter valid outputs;
- cached heterogeneous lengths/right padding raise the documented
  `equal-length without padding` error before the native model is called;
- parameter groups are pairwise disjoint and cover `named_parameters()`
  exactly once, including a synthetic video-only temporal parameter; shared
  image/video patch parameters occur in one group once.

- [ ] **Step 5: Register the complete profile factory**

The factory requires the Qwen35 environment, builds the public backbone from
the exact config or a test config, constructs media/timestamp components,
validates the manifest, and returns exactly:

```python
ProfileBuildResult(
    artifact=thinker,
    manifest=config.profile_manifest,
    architecture_summary=summarize_qwen35_runtime(
        thinker,
        manifest=config.profile_manifest,
    ),
)
```

`summarize_qwen35_runtime()` first calls the public
`summarize_model(thinker, manifest)` two-argument interface, then returns a new
strict `ArchitectureSummary` whose capability mapping is defensively copied
and extended from **inspected attached modules**. At this Task 5 boundary it
must record:

```python
{
    "streaming_audio_encoder": False,
    "paper_inspired_aria": False,
    "speech_talker": False,
    "predecessor_codec_proxy": False,
}
```

It must not pass an unsupported `capability_overrides` keyword into the common
summarizer or claim components scheduled for later tasks. Task 7 regenerates
the summary after attaching and inspecting ARIA/Talker/codec.

The result remains paper-inspired, and the factory never calls an official
Qwen3.5-Omni model class.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider \
  tests/qwen35/test_thinker.py \
  tests/qwen35/test_profile_end_to_end.py -q
```

Expected: all tests pass.

```bash
git add \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/thinker.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py \
  tests/qwen35/test_thinker.py \
  tests/qwen35/test_profile_end_to_end.py
git commit -m "feat: build the Qwen3.5 inspired Thinker"
```

---

### Task 6: Implement online ARIA without future sequence totals

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/aria.py`
- Create: `tests/qwen35/test_aria.py`

**Interfaces:**
- Consumes: `AriaConfig`.
- Produces: `AriaState`, `AriaScheduler.allowed_modalities()`,
  `AriaScheduler.commit()`, `AriaScheduler.snapshot()`, and
  `AriaScheduler.restore()`.

- [ ] **Step 1: Write failing online-rate and dual-EOS tests**

```python
def test_scheduler_uses_rate_target_without_future_totals():
    config = AriaConfig(
        speech_tokens_per_text_num=12,
        speech_tokens_per_text_den=5,
        text_first=True,
        tie_break="text",
    )
    scheduler = AriaScheduler(config)
    state = AriaState()
    assert all("total_" not in field.name for field in fields(state))

    # The fixture owns finite queues, but neither their lengths nor final
    # counts are passed into AriaState or AriaScheduler.
    queues = {
        "text": deque([11, 12, TEXT_EOS]),
        "speech": deque([21, 22, 23, 24, 25, SPEECH_EOS]),
    }
    while not state.finished:
        allowed = scheduler.allowed_modalities(
            state,
            text_available=bool(queues["text"]),
            speech_available=bool(queues["speech"]),
        )
        assert allowed  # both finite fixtures eventually expose an EOS
        modality = allowed[0]
        token = queues[modality].popleft()
        kind = (
            AriaTokenKind.EOS
            if token in {TEXT_EOS, SPEECH_EOS}
            else AriaTokenKind.CONTENT
        )
        state = scheduler.commit(state, modality, token_kind=kind)
        if not state.text_eos_seen and not state.speech_eos_seen:
            assert (
                state.emitted_speech_tokens
                * config.speech_tokens_per_text_den
                <= state.emitted_text_tokens
                * config.speech_tokens_per_text_num
            )
    assert state.text_eos_seen and state.speech_eos_seen


def test_one_eos_does_not_finish_or_block_the_other_stream():
    scheduler = AriaScheduler(tiny_aria_config())
    state = scheduler.commit(
        AriaState(),
        "text",
        token_kind=AriaTokenKind.EOS,
    )
    assert not state.finished
    assert scheduler.allowed_modalities(
        state,
        text_available=False,
        speech_available=True,
    ) == ("speech",)
    state = scheduler.commit(
        state,
        "speech",
        token_kind=AriaTokenKind.EOS,
    )
    assert state.finished


def test_snapshot_contains_no_oracle_future_lengths():
    scheduler = AriaScheduler(tiny_aria_config())
    snapshot = scheduler.snapshot(AriaState())
    assert set(snapshot) == {
        "emitted_text_tokens",
        "emitted_speech_tokens",
        "text_eos_seen",
        "speech_eos_seen",
        "step",
    }
    assert scheduler.restore(snapshot) == AriaState()
```

`tiny_aria_config()` uses the fixed positive integer rate numerator and
denominator from the profile config.

- [ ] **Step 2: Run and observe missing scheduler failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_aria.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement immutable online state and exact integer rules**

```python
@dataclass(frozen=True)
class AriaState:
    emitted_text_tokens: int = 0
    emitted_speech_tokens: int = 0
    text_eos_seen: bool = False
    speech_eos_seen: bool = False
    step: int = 0

    @property
    def finished(self) -> bool:
        return self.text_eos_seen and self.speech_eos_seen


class AriaTokenKind(str, Enum):
    CONTENT = "content"
    EOS = "eos"
    PADDING = "padding"


class AriaScheduler:
    def allowed_modalities(
        self,
        state: AriaState,
        *,
        text_available: bool,
        speech_available: bool,
    ) -> tuple[Literal["text", "speech"], ...]:
        self._validate_state(state)
        candidates: list[Literal["text", "speech"]] = []
        if text_available and not state.text_eos_seen:
            candidates.append("text")
        if speech_available and not state.speech_eos_seen:
            speech_within_rate = (
                (state.emitted_speech_tokens + 1)
                * self.config.speech_tokens_per_text_den
                <= state.emitted_text_tokens
                * self.config.speech_tokens_per_text_num
            )
            if state.text_eos_seen or speech_within_rate:
                candidates.append("speech")
        if state.speech_eos_seen:
            return tuple(x for x in candidates if x == "text")
        if state.text_eos_seen:
            return tuple(x for x in candidates if x == "speech")
        return self._sort_by_integer_rate_error(state, candidates)
```

`_sort_by_integer_rate_error()` computes only these integer candidate scores:

```python
text_error = abs(
    state.emitted_speech_tokens
    * config.speech_tokens_per_text_den
    - (state.emitted_text_tokens + 1)
    * config.speech_tokens_per_text_num
)
speech_error = abs(
    (state.emitted_speech_tokens + 1)
    * config.speech_tokens_per_text_den
    - state.emitted_text_tokens
    * config.speech_tokens_per_text_num
)
```

Lower error is first; exact equality uses `tie_break`. At the initial
`(0,0)` state, `text_first=true` forces text first. Version one validates
`text_first is True`, both rate terms are plain positive integers (booleans
are rejected), and `tie_break` is `text` or `speech`. `allowed_modalities()`
may return an empty tuple when the only currently available stream would
break the active two-stream rate envelope; the caller waits for availability
instead of emitting an invalid token.

`commit()` applies the following exact rules:

- reject a modality whose EOS was already seen;
- `CONTENT` increments the selected emitted counter;
- `EOS` marks only the selected modality finished and increments its counter
  only when `count_eos=true`;
- `PADDING` increments only when `count_padding=true`; otherwise reject it so
  an uncounted no-op cannot spin the scheduler;
- before either EOS is committed, a counted speech token must satisfy
  `(speech + 1) * den <= text * num`; once one stream has ended, the remaining
  stream drains to its own EOS without a rate constraint;
- every accepted event increments `step`, and no event is accepted after both
  EOS flags are true;
- `finished` means both EOS flags, including the zero-content case; there are
  no total-token, expected-length, or look-ahead fields anywhere in state,
  snapshot, restore, Talker call sites, or checkpoint payloads.

`snapshot()` emits exactly the five primitive fields asserted above.
`restore()` requires exactly those keys, rejects booleans in integer fields,
negative counters/step, and `finished`/counter contradictions. It recreates
an immutable `AriaState`; it does not accept legacy total-count payloads.

- [ ] **Step 4: Add exhaustive bounded-stream and availability tests**

Enumerate fixture content lengths from 0 through 8 for each stream, append an
EOS to both queues, and exercise deterministic availability traces (both
ready, alternating stalls, text delayed, speech delayed). The fixture may
know queue lengths to bound the test, but those values are never passed to
the scheduler. Verify:

- both EOS events are required and termination occurs when availability
  eventually resumes;
- while both streams are active, every counted speech prefix satisfies the
  integer upper envelope;
- after either EOS, the other stream drains without deadlock;
- the same state plus availability produces the same ordered modalities;
- no float operation or future total appears in the implementation/state;
- interrupted `snapshot()`/`restore()` yields the same remaining decisions;
- `count_eos` and `count_padding` combinations match counter changes,
  including rejected uncounted padding;
- invalid numerator/denominator, `text_first=false`, post-EOS commit,
  impossible state and unknown snapshot key fail with stable messages.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35/test_aria.py -q
```

Expected: all tests pass.

```bash
git add src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/aria.py \
  tests/qwen35/test_aria.py
git commit -m "feat: schedule online Qwen3.5 inspired ARIA output"
```

---

### Task 7: Add the paper-inspired Talker and predecessor codec proxy

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/talker.py`
- Create: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/runtime.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/thinker.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py`
- Create: `tests/qwen35/test_talker.py`
- Create: `tests/qwen35/test_aria_talker.py`
- Modify: `tests/qwen35/test_thinker.py`
- Modify: `tests/qwen35/test_profile_end_to_end.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Thinker hidden/text stream, public Qwen3.5 backbone blocks, ARIA,
  Qwen3 predecessor Talker/MTP/Code2Wav configuration, and the pinned
  `ReferenceCodecStreamer` contract.
- Produces: `Qwen35InspiredTalker`, `Qwen35InspiredRuntime`,
  `AriaStepOutput | AriaWaitOutput`, and
  `PredecessorCodecProxy.decode_code_chunk()`.

- [ ] **Step 1: Write failing Talker and provenance tests**

```python
def test_talker_predicts_main_codebook_and_marks_proxy():
    talker = tiny_qwen35_inspired_talker()
    output = talker(
        thinker_hidden=torch.randn(1, 4, 16),
        text_input_ids=torch.tensor([[3, 4, 5, 6]]),
        main_codes=torch.tensor([[1, 2, 3]]),
        labels=torch.tensor([[2, 3, 4]]),
    )
    assert output.main_code_logits.shape[-1] == 3072
    assert output.manifest.assumptions.count(
        "predecessor-codec-proxy"
    ) == 1


def test_profile_never_labels_proxy_values_as_qwen35_official():
    result = tiny_qwen35_speech_build_result()
    assert result.architecture_summary.compatibility_level == "paper-inspired"
    assert result.manifest.compatibility_level.value == "paper-inspired"
    assert not result.manifest.exact_official_checkpoint_compatible
    assert result.manifest.sources["codec"].name == (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    assert result.manifest.sources["codec"].revision == (
        "26291f793822fb6be9555850f06dfe95f2d7e695"
    )


def test_proxy_chunk_decode_matches_offline_decode():
    proxy = tiny_predecessor_codec_proxy()
    codes = valid_proxy_codes(frames=10)
    state = CodecOverlapState.empty(request_id="r1")
    chunks = []
    for part, final in (
        (codes[:, :, :4], False),
        (codes[:, :, 4:7], False),
        (codes[:, :, 7:], True),
    ):
        output = proxy.decode_code_chunk(
            codes=part,
            state=state,
            request_id="r1",
            final=final,
        )
        chunks.append(output.waveform)
        state = output.state
    torch.testing.assert_close(
        torch.cat(chunks, dim=-1),
        proxy.decode_codes(codes),
    )
    assert output.final
```

- [ ] **Step 2: Run and observe missing Talker failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider \
  tests/qwen35/test_talker.py tests/qwen35/test_aria_talker.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement a configurable Hybrid-MoE Talker**

Use a small public Qwen3.5 MoE text model configured through explicit
paper-inspired assumptions. Condition it on:

- Thinker multimodal hidden states;
- historical text IDs;
- current text stream;
- Talker-specific system prompt embedding;
- optional predecessor codec prompt.

Add one main-codebook head. Do not reuse official Plus/Flash class names.
Create `Qwen35InspiredRuntime(nn.Module)` as the complete artifact. It owns the
Task-5 Thinker, ARIA scheduler/state adapter, Talker, MTP proxy and codec proxy;
its typed `prefill()`/`decode()` delegates text/cache work to the Thinker and
returns the common immutable state plus speech state. Its
`named_parameter_groups()` delegates the Thinker's four stable groups and
adds stable `talker`, `mtp`, and `codec` tuples by object identity. Re-run the
unique-ownership test after these components are attached. The wrapper and its
new state-dict prefix are introduced in this explicit profile schema revision;
the factory refuses Task-5 partial checkpoints unless a documented migration
rewrites and validates the entire key inventory before tensor loading.

- [ ] **Step 4: Wrap predecessor MTP and Code2Wav**

`PredecessorCodecProxy` loads the pinned Qwen3 predecessor components and
owns the pinned `ReferenceCodecStreamer` from the Qwen3 reference-runtime
plan. It exposes:

```python
def predict_residual_codes(
    self,
    *,
    main_codes: torch.LongTensor,
    conditioning: torch.Tensor,
) -> torch.LongTensor:
    ...

def decode_codes(
    self,
    codes: torch.LongTensor,
) -> torch.Tensor:
    ...

def decode_code_chunk(
    self,
    *,
    codes: torch.LongTensor,
    state: CodecOverlapState,
    request_id: str,
    final: bool,
) -> WaveformChunk:
    return self.codec_streamer.push(
        codes=codes,
        state=state,
        request_id=request_id,
        final=final,
    )
```

`CodecOverlapState` and `WaveformChunk` are imported from
`profiles.qwen3_omni_reference.codec_streamer`; no parallel state schema is
invented. Offline `decode_codes()` calls the same pinned decoder's complete
`chunked_decode()` path. Validate 16 codebooks, integer dtype, correct vocab
ranges, request ownership, final-state reuse and chunk order. Concatenated
chunks must satisfy the reference streamer's boundary/interior tolerances
against offline output. This is overlap-recompute component streaming, not an
incremental AuT or end-to-end latency claim. No proxy component may change the
profile compatibility label.

- [ ] **Step 5: Integrate ARIA token selection**

```python
@dataclass(frozen=True)
class AriaStepOutput:
    modality: Literal["text", "speech"]
    token_id: torch.LongTensor
    state: AriaState


@dataclass(frozen=True)
class AriaWaitOutput:
    state: AriaState
    reason: Literal["no_allowed_available_modality"]


def select_interleaved_token(
    *,
    text_logits: torch.Tensor | None,
    speech_logits: torch.Tensor | None,
    text_eos_token_id: int,
    speech_eos_token_id: int,
    state: AriaState,
    scheduler: AriaScheduler,
) -> AriaStepOutput | AriaWaitOutput:
    ...
```

Pass `text_logits is not None` / `speech_logits is not None` as current
availability, select only the first scheduler-allowed modality, apply
deterministic greedy selection, classify the selected ID as that modality's
EOS or content, then commit state. An empty allowed tuple returns an explicit
`AriaWaitOutput` and consumes neither logits nor state. There is no future
length argument. Keep non-greedy sampling policy outside the scheduler.

- [ ] **Step 6: Test shapes, causality, scheduling, and boundaries**

Cover:

- main/residual codebook shapes and ranges;
- codebook CE/MTP masks;
- no future-code leakage;
- every active two-stream generated prefix satisfies the configured integer
  rate envelope without passing total lengths;
- text EOS alone and speech EOS alone do not finish the request; both do;
- interrupted/resumed state matches uninterrupted output;
- codec proxy chunk/offline parity and cross-request/final-state rejection are
  inherited from `ReferenceCodecStreamer`;
- profile remains paper-inspired before and after loading proxy weights.

After attaching Talker/MTP/codec, `Qwen35InspiredFactory.build()` still returns
a `ProfileBuildResult` (never a bare model/config). Its artifact is the
typed `Qwen35InspiredRuntime`, its manifest is the same validated paper-inspired
manifest, and its full `ArchitectureSummary` is regenerated through
`summarize_qwen35_runtime()`. The helper inspects attached modules and now
sets `paper_inspired_aria`, `speech_talker` and
`predecessor_codec_proxy` true while leaving
`streaming_audio_encoder=false`; tests inspect exact compatibility/source fields on
`result.manifest`, not nonexistent `ArchitectureSummary.exact_*` or
`ArchitectureSummary.sources` attributes.

Update every Task-5 factory assertion in `test_thinker.py` and
`test_profile_end_to_end.py` at this same commit: the final
artifact is now `Qwen35InspiredRuntime`, `artifact.thinker` is the original
`Qwen35InspiredThinker`, and prefill/decode delegation preserves the Task-5
text/cache parity. No full-suite test may retain the superseded
`isinstance(result.artifact, Qwen35InspiredThinker)` expectation.

- [ ] **Step 7: Document and run all Qwen3.5 tests**

README capability row:

```text
Qwen3.5 public backbone: oracle-aligned
Omni media/timestamps/ARIA: paper-inspired
Speech codec: Qwen3 predecessor proxy
Official Qwen3.5-Omni checkpoint compatibility: false
```

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35 \
  tests/oracle/test_qwen35_public_config.py \
  tests/oracle/test_qwen35_gdn_oracle.py -q
```

Expected: zero failures.

- [ ] **Step 8: Commit**

```bash
git add \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/talker.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/runtime.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/thinker.py \
  src/qwen3_omni_pretrain/profiles/qwen35_omni_inspired/factory.py \
  tests/qwen35/test_talker.py \
  tests/qwen35/test_aria_talker.py \
  tests/qwen35/test_thinker.py \
  tests/qwen35/test_profile_end_to_end.py \
  README.md
git commit -m "feat: add Qwen3.5 inspired speech path"
```

---

## Plan completion gate

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m pytest -p no:cacheprovider tests/qwen35 \
  tests/oracle/test_qwen35_public_config.py \
  tests/oracle/test_qwen35_gdn_oracle.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q

PYTHONDONTWRITEBYTECODE=1 .venv-qwen35-backbone/bin/python \
  -m compileall -q src tests

git diff --check
git status --short
```

The plan is complete only when:

- the public backbone config, GDN/full pattern and recurrent cache are preserved
  without legacy DeltaNet substitution;
- common/native cache conversion passes full and recurrent parity;
- two-second audio produces approximately 12–13 valid tokens at 6.25 Hz;
- timestamp text and 160 ms TM-RoPE alignment are explicit and tested;
- text-only adapter matches the public backbone numerically;
- ARIA terminates and satisfies its prefix invariant for exhaustive small cases;
- Talker/codec values are labeled assumptions/proxy at every serialization and
  reporting boundary;
- no class or checkpoint claim uses Plus/Flash identity;
- Qwen3.5 and full prototype suites pass;
- all changes are intentional and committed.
