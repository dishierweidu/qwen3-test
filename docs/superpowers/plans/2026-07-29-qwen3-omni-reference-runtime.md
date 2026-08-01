# Qwen3-Omni Reference Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用固定 revision 的官方 Transformers 组件提供 checkpoint-compatible Qwen3-Omni Thinker、Talker、MTP 和 Code2Wav 运行路径，并提供不混淆离线与实时语义的生成接口。

**Architecture:** `qwen3_omni_reference` factory 直接构建官方 Transformers 5.2.0 类，避免复制或改写权重结构。processor、严格 checkpoint loader、text/speech output 和离线 codec streamer 由薄适配器封装。Transformers 路径明确标为 offline；真正端到端增量输入/输出通过独立 vLLM-Omni WebSocket client 实现。

**Tech Stack:** Python 3.10, PyTorch 2.10.0, Transformers 5.2.0, qwen-omni-utils 0.0.9, huggingface_hub 1.25.1 safetensors metadata, FFmpeg, websockets 16.1.1 for optional realtime client, pytest.

## Global Constraints

- 本计划依赖 profile/oracle、media/TM-RoPE 和 decoder-state 计划完成。
- 官方模型固定为 `Qwen/Qwen3-Omni-30B-A3B-Instruct` revision `26291f793822fb6be9555850f06dfe95f2d7e695`。
- 官方实现固定为 Transformers 5.2.0；版本不一致必须在模型分配前失败。
- checkpoint-compatible 只在 config、state key/shape、严格加载、processor、FP32 numerics、cache 和 generation gates 全部通过后成立。
- reference loader 禁止 `strict=False`、语义字段 coercion、未解释的 missing/unexpected keys。
- 普通测试不得下载大权重；完整权重测试要求本地 artifact、显式 marker 和足够硬件。
- Transformers `generate()` 是离线完整生成，不得标为逐帧端到端 streaming。
- 官方 `chunked_decode()` 是 overlap/recompute；只有通过拼接 parity 的 wrapper 才能称为增量 codec 输出。
- vLLM-Omni 是独立服务依赖，不是 Qwen checkpoint 的训练代码或 Transformers backend。
- reference runtime 不修改官方 state-dict key，也不把模型包装进会增加 key prefix 的 `nn.Module`。
- waveform 输出固定记录采样率、dtype、shape、speaker 和 source revision。
- 论文的 234 ms 等 latency 数字不是本地 pass/fail gate。
- Snippet 中的 `...` 只表示 Protocol/签名节选；任务提交不得保留未实现
  stub，必须由该任务列出的行为测试证明可用。
- 每个行为变更使用 red-green-refactor，并形成独立提交。

---

## File responsibility map

- `profiles/qwen3_omni_reference/checkpoint.py`: safetensors inventory 和严格官方 loader。
- `profiles/qwen3_omni_reference/processor.py`: canonical conversation 到官方 BatchFeature。
- `profiles/qwen3_omni_reference/outputs.py`: text、speech、codec chunk 的稳定返回类型。
- `profiles/qwen3_omni_reference/runtime.py`: offline Thinker/full-Omni 调用，不继承 `nn.Module`。
- `profiles/qwen3_omni_reference/codec_streamer.py`: Code2Wav overlap state 和安全音频增量。
- `profiles/qwen3_omni_reference/realtime_client.py`: vLLM-Omni `/v1/realtime` client。
- `requirements-vllm-omni-client.txt`: `websockets==16.1.1`，不加入 prototype 核心环境。
- `tests/oracle/qwen3_omni`: 官方 config/processor/state/numerical tests。
- `tests/qwen3_reference`: 适配器和 mock-service tests。

---

### Task 1: Compare the official model graph with checkpoint metadata

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/checkpoint.py`
- Create: `tests/oracle/qwen3_omni/test_state_inventory.py`
- Create: `tests/fixtures/qwen3_omni/state_inventory_contract.json`

**Interfaces:**
- Consumes: pinned oracle constants and `ProfileManifest`.
- Produces: `TensorInventoryEntry`, `StateInventory`, `CompatibilityReport`, `capture_remote_inventory()`, `capture_local_inventory()`, `compare_state_inventory()`, and `load_reference_model_strict()`.

- [ ] **Step 1: Write failing inventory comparison tests**

```python
def test_inventory_comparison_reports_exact_shape_mismatch():
    expected = StateInventory(
        tensors={
            "thinker.model.embed_tokens.weight": TensorInventoryEntry(
                shape=(32, 8),
                dtype="BF16",
            )
        },
        source_repo=QWEN3_OMNI_MODEL_ID,
        source_revision=QWEN3_OMNI_REVISION,
        transformers_version="5.2.0",
    )
    actual = {
        "thinker.model.embed_tokens.weight": torch.empty(
            31, 8, device="meta", dtype=torch.bfloat16
        )
    }
    report = compare_state_inventory(actual, expected)
    assert report.shape_mismatches == {
        "thinker.model.embed_tokens.weight": ((31, 8), (32, 8))
    }
    assert not report.compatible
```

- [ ] **Step 2: Run and observe missing checkpoint module**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/qwen3_omni/test_state_inventory.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Define inventory and report types**

```python
@dataclass(frozen=True)
class TensorInventoryEntry:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class StateInventory:
    tensors: Mapping[str, TensorInventoryEntry]
    source_repo: str
    source_revision: str
    transformers_version: str


@dataclass(frozen=True)
class CompatibilityReport:
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: Mapping[
        str,
        tuple[tuple[int, ...], tuple[int, ...]],
    ]
    dtype_mismatches: Mapping[str, tuple[str, str]]

    @property
    def compatible(self) -> bool:
        return not (
            self.missing_keys
            or self.unexpected_keys
            or self.shape_mismatches
            or self.dtype_mismatches
        )
```

- [ ] **Step 4: Capture metadata without downloading tensors**

Use:

```python
metadata = huggingface_hub.get_safetensors_metadata(
    QWEN3_OMNI_MODEL_ID,
    revision=QWEN3_OMNI_REVISION,
)
```

Convert every parameter name, shape and dtype into `StateInventory`. Capture
the official model's meta-device `state_dict()` and compare. The fixture stores
the source/revision/tool versions, SHA-256 of its normalized JSON, aggregate
parameter count, component prefixes and the full key/shape inventory.

This is an explicit `@pytest.mark.network` capture operation; ordinary tests
read only the checked-in fixture.

- [ ] **Step 5: Implement strict weight loading**

```python
def load_reference_model_strict(
    source: str,
    *,
    torch_dtype: torch.dtype,
    device_map: str | Mapping[str, object],
    local_files_only: bool = True,
):
    model, loading = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        source,
        revision=QWEN3_OMNI_REVISION,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=local_files_only,
        output_loading_info=True,
    )
    failures = {
        "missing_keys": loading["missing_keys"],
        "unexpected_keys": loading["unexpected_keys"],
        "mismatched_keys": loading["mismatched_keys"],
        "error_msgs": loading["error_msgs"],
    }
    if any(failures.values()):
        raise RuntimeError(f"strict Qwen3-Omni load failed: {failures}")
    return model
```

Do not subclass or wrap the returned model as a child module; return it directly
through the profile factory so state keys remain official.

- [ ] **Step 6: Run fixture tests and optional strict-load test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/qwen3_omni/test_state_inventory.py -q

PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider -m large_model \
  --run-large-model-tests \
  tests/oracle/qwen3_omni/test_state_inventory.py -q
```

Expected: fixture test passes; large test either passes with local weights or
skips before model construction with an explicit local-artifact reason.

- [ ] **Step 7: Commit**

```bash
git add \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/checkpoint.py \
  tests/oracle/qwen3_omni/test_state_inventory.py \
  tests/fixtures/qwen3_omni/state_inventory_contract.json
git commit -m "test: verify Qwen3 Omni checkpoint inventory"
```

---

### Task 2: Adapt canonical conversations to the official processor

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/processor.py`
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/outputs.py`
- Create: `tests/qwen3_reference/test_processor_adapter.py`
- Modify: `tests/oracle/test_qwen3_omni_processor_oracle.py`

**Interfaces:**
- Consumes: official processor, qwen-omni-utils, canonical `messages`.
- Produces: `ReferenceProcessorInput`, `ReferenceProcessorOutput`, and `Qwen3ReferenceProcessor.prepare()`.

- [ ] **Step 1: Write a failing no-coercion processor test**

```python
def test_processor_adapter_passes_official_media_metadata(monkeypatch):
    processor = CapturingOfficialProcessor()
    adapter = Qwen3ReferenceProcessor(
        processor=processor,
        process_mm_info=fake_process_mm_info,
    )
    output = adapter.prepare(
        ReferenceProcessorInput(
            messages=(
                {
                    "role": "user",
                    "content": (
                        {"type": "image", "image": "image.png"},
                        {"type": "text", "text": "describe"},
                    ),
                },
            ),
            use_audio_in_video=True,
        )
    )
    assert processor.kwargs["use_audio_in_video"] is True
    assert output.batch["image_grid_thw"].shape[-1] == 3
    assert output.manifest_digest
```

- [ ] **Step 2: Run and observe missing adapter failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/qwen3_reference/test_processor_adapter.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Define immutable processor inputs and outputs**

```python
@dataclass(frozen=True)
class ReferenceProcessorInput:
    messages: tuple[Mapping[str, object], ...]
    use_audio_in_video: bool
    add_generation_prompt: bool = True


@dataclass(frozen=True)
class ReferenceProcessorOutput:
    batch: Mapping[str, torch.Tensor]
    rendered_text: str
    manifest_digest: str
    processor_revision: str
```

`Qwen3ReferenceProcessor.prepare()`:

1. validates roles and content types;
2. calls official `apply_chat_template`;
3. calls pinned `process_mm_info`;
4. calls official processor with unmodified image/video/audio outputs;
5. validates required grid/feature fields;
6. records the manifest digest and revision.

It never maps the reference inputs into legacy `pixel_values/audio_values`.

- [ ] **Step 4: Add deterministic processor parity**

For text-only, image, audio, video and audio-video fixtures compare:

- input IDs;
- media sentinel expansion counts;
- feature masks and lengths;
- `image_grid_thw`/`video_grid_thw`;
- `video_second_per_grid`;
- `get_rope_index()` position IDs and rope delta.

Inputs include content SHA-256 and decoder/FFmpeg versions. Processor parity is
exact for integer metadata and tolerance-based only for floating features.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/qwen3_reference/test_processor_adapter.py \
  tests/oracle/test_qwen3_omni_processor_oracle.py -q
```

Expected: all cached-fixture tests pass.

```bash
git add \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/processor.py \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/outputs.py \
  tests/qwen3_reference/test_processor_adapter.py \
  tests/oracle/test_qwen3_omni_processor_oracle.py
git commit -m "feat: adapt official Qwen3 Omni processing"
```

---

### Task 3: Provide a strict Thinker text-generation runtime

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/runtime.py`
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/numeric_fixture.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/factory.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/__init__.py`
- Modify: `src/qwen3_omni_pretrain/profiles/registry.py`
- Create: `tests/qwen3_reference/test_text_runtime.py`
- Create: `tests/oracle/qwen3_omni/test_thinker_numerics.py`
- Create: `scripts/capture_qwen3_text_oracle.py`

**Interfaces:**
- Consumes: strict model loader and `ReferenceProcessorOutput`.
- Produces: `ReferenceTextResult` and `Qwen3OmniReferenceRuntime.generate_text()`.

- [ ] **Step 1: Write a failing runtime delegation test**

```python
def test_text_runtime_requests_no_audio_and_returns_only_new_tokens():
    model = FakeOfficialOmniModel(generated=torch.tensor([[5, 6, 7, 8]]))
    runtime = Qwen3OmniReferenceRuntime(
        model=model,
        processor=fake_reference_processor(prompt_ids=[5, 6]),
    )
    result = runtime.generate_text(messages=text_messages(), max_new_tokens=2)
    assert model.generate_calls[0]["return_audio"] is False
    assert model.disable_talker_calls == 0
    assert result.prompt_token_ids.tolist() == [[5, 6]]
    assert result.generated_token_ids.tolist() == [[7, 8]]
    assert result.audio is None
```

- [ ] **Step 2: Run and observe the missing runtime**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider tests/qwen3_reference/test_text_runtime.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Implement a non-module runtime facade**

```python
@dataclass(frozen=True)
class ReferenceTextResult:
    prompt_token_ids: torch.LongTensor
    generated_token_ids: torch.LongTensor
    text: tuple[str, ...]
    audio: None
    manifest: ProfileManifest


class Qwen3OmniReferenceRuntime:
    def __init__(self, *, model, processor: Qwen3ReferenceProcessor) -> None:
        self.model = model
        self.processor = processor

    def generate_text(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        max_new_tokens: int,
        use_audio_in_video: bool = False,
    ) -> ReferenceTextResult:
        ...
```

The facade is not an `nn.Module`; it cannot alter state-dict prefixes. It
prepares official inputs, calls official generation with `return_audio=False`,
slices prompt tokens, decodes only new tokens, and attaches the manifest.
`generate_text()` must not call destructive `disable_talker()` because the
same full runtime may later generate speech. When
`ProfileBuildRequest.requested_capabilities == ("text",)`, the factory may
expose a text-only low-memory build mode that disables Talker once during
construction; that artifact advertises text-only capability and cannot later
be upgraded to speech in place. An empty capability tuple requests the full
runtime.

- [ ] **Step 4: Add optional numerical oracle tests**

Capture a fixed FP32/BF16 fixture containing:

- input and processor hashes;
- first/last valid token top-20 IDs and logits;
- one cached next-token step;
- greedy generated IDs;
- hardware and dtype.

The large-model test checks documented tolerances and official cached/uncached
parity. It must not run on ordinary CI. `numeric_fixture.py` defines a strict
schema parser for the capture output. The ordinary test validates a fixture
only when an explicit `QWEN3_TEXT_NUMERIC_FIXTURE` path is supplied; otherwise
it skips with `external numeric evidence not supplied`. It never creates a
passing fixture, sets an evidence gate, or changes the manifest.

`scripts/capture_qwen3_text_oracle.py` requires an absolute local checkpoint
path, `--run-large-model-tests`, the pinned revision, and an explicit output
path. It refuses a dirty source revision, writes normalized JSON through an
atomic temporary file, and records checkpoint/config/processor/input hashes,
exact command, package versions, dtype and hardware. Capture with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  scripts/capture_qwen3_text_oracle.py \
  --checkpoint /absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --run-large-model-tests \
  --output tests/fixtures/qwen3_omni/text_numerics.json
```

The fixture contains no full logits or user media—only hashes, selected top-k
values/IDs, generated IDs and documented tolerances. It is a local evidence
artifact, not a source-controlled deliverable of this task. Ordinary tests may
validate its schema/hash but do not mark `numeric_oracle=true`.

- [ ] **Step 5: Register the completed reference factory**

The factory:

- validates reference environment and local artifacts;
- loads processor and model with exact revision;
- builds `Qwen3OmniReferenceRuntime` inside `ProfileBuildResult.artifact`;
- records strict-load evidence but remains `structure-aligned` until the
  numerical and cache evidence in Task 7 is also present;
- reports text-only capability only for an explicitly requested text-only
  construction; the default full runtime keeps Talker available.

Add a registry-level test that resolves `QWEN3_OMNI_REFERENCE`, calls
`factory.build(request)` with fake local official components, and asserts the
artifact is `Qwen3OmniReferenceRuntime` rather than the Task-1 oracle loader or
a bare model. This task explicitly modifies `factory.py`; registry metadata
continues to point at the same class.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/qwen3_reference/test_text_runtime.py -q
```

Expected: adapter tests pass.

```bash
git add src/qwen3_omni_pretrain/profiles/qwen3_omni_reference \
  src/qwen3_omni_pretrain/profiles/registry.py \
  tests/qwen3_reference/test_text_runtime.py \
  tests/oracle/qwen3_omni/test_thinker_numerics.py \
  scripts/capture_qwen3_text_oracle.py
git commit -m "feat: run the official Qwen3 Omni Thinker"
```

---

### Task 4: Expose Talker, MTP, and offline Code2Wav generation

**Files:**
- Modify: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/runtime.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/outputs.py`
- Create: `tests/qwen3_reference/test_speech_runtime.py`
- Create: `tests/oracle/qwen3_omni/test_speech_numerics.py`
- Create: `scripts/capture_qwen3_speech_oracle.py`

**Interfaces:**
- Consumes: official full model and processor.
- Produces: `ReferenceSpeechCodes`, `ReferenceSpeechResult`,
  `Qwen3OmniReferenceRuntime.generate_speech_codes()`, and
  `generate_speech()`.

- [ ] **Step 1: Write a failing speech-output normalization test**

```python
def test_speech_runtime_normalizes_official_tuple_output():
    model = FakeOfficialOmniModel(
        generated=(
            torch.tensor([[5, 6, 7]]),
            torch.zeros(24_000),
        )
    )
    runtime = fake_runtime(model)
    result = runtime.generate_speech(
        messages=text_messages(),
        speaker="Ethan",
        thinker_max_new_tokens=2,
        talker_max_new_tokens=16,
    )
    assert result.generated_token_ids.tolist() == [[7]]
    assert result.waveform.shape == (24_000,)
    assert result.sample_rate == 24_000
    assert result.streaming is False
```

- [ ] **Step 2: Run and observe missing method failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider tests/qwen3_reference/test_speech_runtime.py -q
```

Expected: FAIL because `generate_speech()` does not exist.

- [ ] **Step 3: Define exact speech result**

```python
@dataclass(frozen=True)
class ReferenceSpeechResult:
    prompt_token_ids: torch.LongTensor
    generated_token_ids: torch.LongTensor
    text: tuple[str, ...]
    waveform: torch.Tensor
    sample_rate: int
    speaker: str
    streaming: bool
    codes: "ReferenceSpeechCodes | None"
    manifest: ProfileManifest


@dataclass(frozen=True)
class ReferenceSpeechCodes:
    main_codes: torch.LongTensor
    residual_codes: torch.LongTensor
    all_codes: torch.LongTensor
```

`generate_speech()` validates one of the checkpoint-supported speakers, calls
official `generate(return_audio=True)`, validates batch limitations and
waveform finiteness, and returns `streaming=False`.
`generate_speech_codes()` uses the pinned official Talker and 5-layer
code-predictor components, returns main `[B,1,T]`, residual `[B,15,T]`, and
concatenated `[B,16,T]`, and validates that concatenation/ranges exactly match.
When the top-level official API cannot expose codes from the same call,
`generate_speech()` sets `codes=None`; callers explicitly request the
component path instead of inferring codes from waveform.

- [ ] **Step 4: Verify component structure before numerical fixtures**

Assert against official config/module graph:

- Talker 20 layers, hidden 1024, 16Q/2KV, 128 experts/top-6;
- routed/shared FFN dimensions 384/768;
- two 2048→2048→1024 projectors and hidden layer index 24;
- 16 code groups, main/residual vocab sizes;
- 5-layer dense MTP;
- Code2Wav input `[B,16,T]`, 24 kHz, 1920 samples/frame;
- 8-layer sliding-attention frontend plus causal convolution decoder.

- [ ] **Step 5: Add opt-in numerical fixtures**

Capture:

- Thinker generated token IDs;
- Talker main codebook IDs;
- all 15 residual codebook IDs for the first frames;
- Code2Wav waveform shape, finite stats, first/last sample windows and hash;
- exact source revision, speaker, dtype and hardware.

Compare token/code IDs exactly and waveform with documented tolerance plus hash
for the identical deterministic environment.

As in Task 3, this fixture is optional local evidence. The ordinary oracle test
loads it only from `QWEN3_SPEECH_NUMERIC_FIXTURE`, otherwise skips explicitly;
absence never blocks the runtime commit and never promotes compatibility.

Capture them with the same local-artifact/revision safeguards as the text
fixture:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  scripts/capture_qwen3_speech_oracle.py \
  --checkpoint /absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --speaker Ethan \
  --run-large-model-tests \
  --output tests/fixtures/qwen3_omni/speech_numerics.json
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/qwen3_reference/test_speech_runtime.py -q
```

Expected: adapter and structure tests pass.

```bash
git add src/qwen3_omni_pretrain/profiles/qwen3_omni_reference \
  tests/qwen3_reference/test_speech_runtime.py \
  tests/oracle/qwen3_omni/test_speech_numerics.py \
  scripts/capture_qwen3_speech_oracle.py
git commit -m "feat: expose Qwen3 Omni speech generation"
```

---

### Task 5: Add a parity-tested incremental Code2Wav wrapper

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/codec_streamer.py`
- Create: `tests/qwen3_reference/test_codec_streamer.py`

**Interfaces:**
- Consumes: official Code2Wav `chunked_decode()` and code tensor `[B,16,T]`.
- Produces: `CodecOverlapState`, `WaveformChunk`, and `ReferenceCodecStreamer.push()`.

- [ ] **Step 1: Write a failing chunk-concatenation test**

```python
def test_codec_stream_concatenation_matches_offline_decoder():
    decoder = DeterministicFakeCode2Wav(samples_per_frame=4)
    codes = torch.arange(16 * 10).view(1, 16, 10)
    streamer = ReferenceCodecStreamer(
        decoder=decoder,
        left_context_frames=2,
        samples_per_frame=4,
    )
    state = CodecOverlapState.empty(request_id="r1")
    outputs = []
    for chunk, final in ((codes[:, :, :4], False), (codes[:, :, 4:7], False),
                         (codes[:, :, 7:], True)):
        result = streamer.push(
            codes=chunk,
            state=state,
            request_id="r1",
            final=final,
        )
        outputs.append(result.waveform)
        state = result.state
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1),
        decoder.chunked_decode(codes),
    )
```

- [ ] **Step 2: Run and observe missing streamer failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider tests/qwen3_reference/test_codec_streamer.py -q
```

Expected: FAIL at import.

- [ ] **Step 3: Define state and output**

```python
@dataclass(frozen=True)
class CodecOverlapState:
    request_id: str
    left_codes: torch.LongTensor | None
    emitted_frames: int

    @classmethod
    def empty(cls, request_id: str) -> "CodecOverlapState":
        return cls(request_id=request_id, left_codes=None, emitted_frames=0)


@dataclass(frozen=True)
class WaveformChunk:
    waveform: torch.Tensor
    state: CodecOverlapState
    sample_rate: int
    final: bool
```

`push()` concatenates retained left codes and new codes, calls official
`chunked_decode()`, removes audio corresponding to recomputed left context,
emits only new safe samples, and returns the last configured context frames.
It rejects cross-request state, wrong codebook count and non-integer codes.
Add an integration test that obtains `ReferenceSpeechCodes.all_codes` from
`generate_speech_codes()` and feeds successive code slices to `push()`. This
proves the public producer/consumer boundary, but remains a post-generation
overlap/recompute path: Talker code generation itself is not streamed and the
test must not report full-model TTFC.

- [ ] **Step 4: Validate official overlap behavior**

Use deterministic random codes with lengths around official chunk/context
boundaries. Compare concatenated increments to one offline
`chunked_decode()` call. Record boundary tolerance separately from interior
tolerance. This test may use a tiny official Code2Wav config for CPU; checkpoint
waveform parity remains opt-in.

- [ ] **Step 5: Keep the capability label precise**

Expose:

```text
codec_incremental_output=true
end_to_end_streaming=false
implementation=overlap-recompute
```

Do not report TTFC for the full model from this component-only wrapper.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider tests/qwen3_reference/test_codec_streamer.py -q
```

Expected: all tests pass.

```bash
git add \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/codec_streamer.py \
  tests/qwen3_reference/test_codec_streamer.py
git commit -m "feat: stream Qwen3 codec output incrementally"
```

---

### Task 6: Add a vLLM-Omni realtime service client

**Files:**
- Create: `requirements-vllm-omni-client.txt`
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/realtime_client.py`
- Create: `tests/qwen3_reference/test_realtime_client.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: vLLM-Omni OpenAI-style `/v1/realtime` WebSocket API.
- Produces: `RealtimeRequest`, `RealtimeAudioDelta`, `RealtimeTextDelta`, and `VllmOmniRealtimeClient.stream()`.

- [ ] **Step 1: Pin the optional client dependency**

`requirements-vllm-omni-client.txt`:

```text
websockets==16.1.1
```

Do not add it to `requirements.txt`. Add a dependency-profile test asserting
the exact pin.

- [ ] **Step 2: Write a failing mocked event-order test**

```python
def test_realtime_client_sends_commit_before_response_create():
    async def scenario():
        socket = FakeWebSocket(
            incoming=[
                {
                    "type": "response.audio.delta",
                    "delta": base64_pcm(b"\x00\x01"),
                },
                {"type": "response.done"},
            ]
        )
        client = VllmOmniRealtimeClient(
            url="ws://localhost:8091/v1/realtime",
            connector=fake_connector(socket),
        )
        events = [
            event
            async for event in client.stream(
                RealtimeRequest(
                    request_id="r1",
                    model=QWEN3_OMNI_MODEL_ID,
                    pcm16_chunks=(b"\x01\x02", b"\x03\x04"),
                    input_sample_rate=16_000,
                )
            )
        ]
        sent_types = [json.loads(message)["type"] for message in socket.sent]
        assert sent_types[-2:] == [
            "input_audio_buffer.commit",
            "response.create",
        ]
        assert isinstance(events[0], RealtimeAudioDelta)

    asyncio.run(scenario())
```

- [ ] **Step 3: Run and observe missing client failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider tests/qwen3_reference/test_realtime_client.py -q
```

Expected: FAIL at import.

- [ ] **Step 4: Implement strict event parsing**

The async client:

1. connects to the exact URL;
2. sends `session.update` with model, modalities and audio format;
3. sends each PCM16 mono 16 kHz chunk as base64
   `input_audio_buffer.append`;
4. sends `commit`, then `response.create`;
5. yields `response.audio.delta` as PCM chunks and transcription/text deltas;
6. stops only on the matching response completion event;
7. propagates structured server errors;
8. closes on cancellation without reusing the session.

Unknown event types are recorded as diagnostics, not treated as audio.
`websockets` is imported lazily only inside the default connector. Supplying a
fake connector keeps the unit test runnable in the reference environment even
when the optional client requirements have not been installed.

- [ ] **Step 5: Add an opt-in live smoke test**

Environment variables:

```text
QWEN3_OMNI_REALTIME_URL
QWEN3_OMNI_REALTIME_MODEL
```

The live test sends a deterministic 200 ms silent PCM chunk, verifies at least
one terminal response event, records TTFC/RTF metadata, and skips when variables
are absent. It never starts or installs the server.

- [ ] **Step 6: Document offline vs realtime**

README must state:

- Transformers runtime: exact checkpoint, offline full text/audio return;
- codec streamer: local incremental overlap/recompute only;
- vLLM-Omni client: external service, true incremental response;
- no local latency promise without documented server hardware/config.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider tests/qwen3_reference/test_realtime_client.py -q
```

Expected: mock tests pass; live test skips unless explicitly configured.

```bash
git add requirements-vllm-omni-client.txt \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/realtime_client.py \
  tests/qwen3_reference/test_realtime_client.py README.md
git commit -m "feat: connect to Qwen3 Omni realtime serving"
```

---

### Task 7: Publish non-promoting reference capability gates

**Files:**
- Create: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/compatibility.py`
- Modify: `src/qwen3_omni_pretrain/architecture/summary.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/runtime.py`
- Modify: `README.md`
- Create: `tests/qwen3_reference/test_capability_gates.py`

**Interfaces:**
- Consumes: all reference runtime components.
- Produces: `ReferenceCompatibilityEvidence`,
  `VerifiedReferenceEvidence`, `load_verified_reference_evidence()`, and
  `promote_reference_build_result()`.

- [ ] **Step 1: Write failing non-promotion tests**

```python
def test_unverified_runtime_stays_structure_aligned():
    result = build_fake_reference_result()
    assert result.manifest.compatibility_level.value == "structure-aligned"
    assert not result.manifest.exact_official_checkpoint_compatible
    assert result.architecture_summary.capabilities["offline_text"] is True
    assert result.architecture_summary.capabilities["end_to_end_streaming"] is False


def test_boolean_only_or_synthetic_evidence_cannot_promote(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"strict_checkpoint_loaded": True}))
    with pytest.raises(ValueError, match="schema|measurement|hash"):
        load_verified_reference_evidence(
            path,
            expected_checkpoint_sha256="a" * 64,
            expected_text_fixture_sha256="b" * 64,
            expected_speech_fixture_sha256="c" * 64,
        )
```

Ordinary tests only prove that missing, incomplete, modified, wrong-revision,
wrong-package and wrong-artifact evidence is rejected. They must never build a
`VerifiedReferenceEvidence` test double and must never exercise the successful
promotion branch.

- [ ] **Step 2: Define complete, hash-bound evidence**

```python
@dataclass(frozen=True)
class ReferenceCompatibilityEvidence:
    schema_version: int
    source_repo: str
    source_revision: str
    checkpoint_sha256: str
    config_contract_sha256: str
    state_inventory_sha256: str
    processor_fixture_sha256: str
    text_fixture_sha256: str
    speech_fixture_sha256: str
    implementation_commit: str
    package_versions: Mapping[str, str]
    gate_measurements: Mapping[str, Mapping[str, object]]
    config_oracle: bool
    processor_oracle: bool
    state_inventory: bool
    strict_checkpoint_loaded: bool
    numeric_oracle: bool
    cached_decode_parity: bool
    offline_text: bool
    offline_speech: bool
    codec_incremental_output: bool
```

The parser requires every field, rejects unknown fields and validates canonical
JSON SHA-256 plus raw measurements for every true gate. Checkpoint compatibility
requires all of `config_oracle`, `processor_oracle`, `state_inventory`,
`strict_checkpoint_loaded`, `numeric_oracle`, `cached_decode_parity`,
`offline_text`, and `offline_speech`. Codec overlap parity is reported
separately and is not a checkpoint-state gate. Realtime requires a successful
external service handshake and never changes checkpoint compatibility.

`VerifiedReferenceEvidence` has no public constructor. It is returned only by
`load_verified_reference_evidence()` after the evidence source/revision,
checkpoint, fixture, package and implementation hashes match the artifacts
selected for the current build.

- [ ] **Step 3: Promote manifest, summary and runtime atomically**

`promote_reference_build_result(result, verified)` must:

1. reject a text-only build or any evidence lacking the eight mandatory gates;
2. create a new manifest with `compatibility_level=checkpoint-compatible` and
   `exact_official_checkpoint_compatible=true`;
3. clone the non-module runtime facade with that manifest so every returned
   `ReferenceTextResult`/`ReferenceSpeechResult` carries the promoted manifest;
4. create a new summary with the same compatibility level and evidence-derived
   capabilities;
5. return one new `ProfileBuildResult` and assert that artifact, manifest and
   summary profile/compatibility values agree.

No field is mutated in place. Merely passing booleans to `reference_summary()`
is not an API and cannot upgrade a result.

- [ ] **Step 4: Run all non-large reference and prototype tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/qwen3_omni \
  tests/qwen3_reference -m "not large_model and not network" -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q
```

Expected: zero failures. The reference manifest is still
`structure-aligned` because no external weight evidence is part of this test.

- [ ] **Step 5: Commit**

```bash
git add src/qwen3_omni_pretrain/architecture/summary.py \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/compatibility.py \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/runtime.py \
  tests/qwen3_reference/test_capability_gates.py README.md
git commit -m "feat: enforce Qwen3 reference capability gates"
```

---

### Task 8: Run the opt-in checkpoint-compatibility promotion

**Files:**
- Create: `scripts/verify_qwen3_reference.py`
- Modify: `src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/factory.py`
- Modify: `src/qwen3_omni_pretrain/profiles/registry.py`
- Create: `tests/oracle/qwen3_omni/test_reference_promotion.py`

**Interfaces:**
- Consumes: an absolute local checkpoint, the two locally captured numerical
  fixtures, pinned processor/config fixtures and all Task 1–7 gates.
- Produces: a canonical local evidence JSON and an optionally promoted
  `ProfileBuildResult`.

- [ ] **Step 1: Keep the promotion path opt-in**

Modify the canonical `ProfileBuildRequest` dataclass in
`profiles/registry.py` (where the common profile plan defines it) to add
`verified_evidence_path: str | None = None`; do not create a shadow request
type in the reference profile. Add a strict request-construction/round-trip
test in `test_reference_promotion.py` so the field is proven present before the
factory consumes it.
Without it, the factory always returns `structure-aligned`, including after a
successful strict load. With it, the factory resolves an absolute path,
recomputes current checkpoint/fixture hashes, calls
`load_verified_reference_evidence()`, then promotes the whole build result.
Invalid evidence fails before generation.

- [ ] **Step 2: Implement the verifier as an execution gate**

`scripts/verify_qwen3_reference.py` does not accept precomputed booleans. In one
run it:

1. verifies the pinned environment and absolute local checkpoint;
2. compares config and full state inventory, then performs strict loading;
3. reruns processor fixtures;
4. reruns text logits/IDs, cached/uncached decode and offline text generation;
5. reruns Talker/MTP code IDs and offline Code2Wav speech generation;
6. optionally runs codec overlap parity;
7. records raw measurements, hashes and package/implementation versions only
   after all mandatory checks pass;
8. writes canonical evidence atomically to an explicit output path.

Any exception removes the temporary output and leaves no promotable artifact.

- [ ] **Step 3: Capture and verify on a machine that has the checkpoint**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  scripts/capture_qwen3_text_oracle.py \
  --checkpoint /absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --run-large-model-tests \
  --output /absolute/path/to/evidence/text_numerics.json

PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  scripts/capture_qwen3_speech_oracle.py \
  --checkpoint /absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --speaker Ethan --run-large-model-tests \
  --output /absolute/path/to/evidence/speech_numerics.json

PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  scripts/verify_qwen3_reference.py \
  --checkpoint /absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
  --text-fixture /absolute/path/to/evidence/text_numerics.json \
  --speech-fixture /absolute/path/to/evidence/speech_numerics.json \
  --run-large-model-tests \
  --output /absolute/path/to/evidence/reference_compatibility.json
```

These outputs are machine-local evidence and are not added to git.

- [ ] **Step 4: Run the promoted-build integration test**

```bash
QWEN3_REFERENCE_CHECKPOINT=/absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
QWEN3_REFERENCE_EVIDENCE=/absolute/path/to/evidence/reference_compatibility.json \
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider -m large_model \
  --run-large-model-tests \
  tests/oracle/qwen3_omni/test_reference_promotion.py -q
```

The test asserts the returned artifact manifest, top-level manifest and summary
are all checkpoint-compatible, then runs one offline text and speech request.
It skips before allocation when either environment variable is absent.

- [ ] **Step 5: Commit code, never generated evidence**

```bash
git add scripts/verify_qwen3_reference.py \
  src/qwen3_omni_pretrain/profiles/qwen3_omni_reference/factory.py \
  src/qwen3_omni_pretrain/profiles/registry.py \
  tests/oracle/qwen3_omni/test_reference_promotion.py
git commit -m "feat: verify Qwen3 checkpoint compatibility"
```

---

## Plan completion gate

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m pytest -p no:cacheprovider \
  tests/oracle/qwen3_omni tests/qwen3_reference \
  -m "not large_model and not network" -q

PYTHONDONTWRITEBYTECODE=1 .venv-prototype/bin/python \
  -m pytest -p no:cacheprovider -q

PYTHONDONTWRITEBYTECODE=1 .venv-qwen-reference/bin/python \
  -m compileall -q src tests

git diff --check
git status --short
```

The source implementation is complete when:

- pinned official config and safetensors inventory match the official model
  graph;
- text and offline speech adapters preserve official inputs/outputs;
- Talker/MTP/Code2Wav structure is checked and optional external numerics have a
  strict capture/verification path;
- codec incremental output passes offline concatenation parity;
- Transformers is never described as end-to-end streaming;
- realtime capability is tied to an external vLLM-Omni handshake;
- both environments pass their non-large suites;
- all changes are intentional and committed.

The stronger `checkpoint-compatible` label is achieved only after Task 8 is
actually run on the pinned local checkpoint: strict loading must report zero
missing/unexpected/mismatched keys; processor, numeric, cached decode, offline
text and offline speech gates must all pass; and the promoted-build integration
test must confirm manifest/summary/runtime agreement. Until then the shipped
default remains `structure-aligned`, which is a valid completion state for
machines without the external 30B artifact.
