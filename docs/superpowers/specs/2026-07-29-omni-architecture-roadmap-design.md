# Omni Architecture Roadmap Design

Date: 2026-07-29
Status: Approved design, pending written-spec review
Local baseline: `391d0186cb225f41f8e47aa600c36da0bc6b21e4`

## 1. Purpose

This design converts three architecture studies into one implementable roadmap:

- [Qwen3-Omni architecture gap](../../research/2026-07-29-qwen3-omni-architecture-gap.md)
- [Qwen3.5-Omni reproduction boundary](../../research/2026-07-29-qwen3.5-omni-reproduction.md)
- [MiMo-V2.5 architecture lessons](../../research/2026-07-29-mimo-v2.5-architecture-lessons.md)

The repository already has a stable P0 prototype with tested Stage-2
configuration, media error handling, tokenizer-derived multimodal IDs, MoE
routing, numerical fail-fast, checkpoint resume, and isolated dependency
profiles. The next work replaces architecture placeholders without losing those
correctness guarantees.

The core design decision is:

> Build an architecture-faithful, externally verifiable Qwen3-Omni reference
> path first. Keep Qwen3.5 and MiMo mechanisms in isolated, explicitly named
> experimental profiles.

This prevents three different architectures from being mixed behind ambiguous
boolean flags and prevents paper-inspired code from being described as an
official reproduction.

## 2. Goals

1. Preserve the current prototype as a regression baseline.
2. Establish an official Qwen3-Omni golden oracle.
3. Replace one-token media summaries with variable-length media sequences.
4. Implement placeholder replacement, video, timestamp alignment, and TM-RoPE.
5. Add correct prefill/decode state before optimizing long-context inference.
6. Add a Qwen3-Omni-compatible Thinker, Talker, MTP, and Code2Wav path.
7. Implement Qwen3.5-style GDN and ARIA as a paper-inspired profile.
8. Implement MiMo-style SWA/GA, routed-only MoE, and MTP as a separate profile.
9. Make all architecture claims machine-verifiable through manifests and tests.
10. Keep unsupported or unpublished behavior explicit and fail-fast.

## 3. Non-goals

This roadmap does not claim to:

- reproduce the training results of Qwen3-Omni from undisclosed data and
  recipes;
- reproduce Qwen3.5-Omni Plus or Flash checkpoints, which are not open;
- infer undisclosed Qwen3.5-Omni dimensions or codec parameters;
- make MiMo-V2.5 a speech-output model;
- jump directly to 256/384 experts, 256K/1M context, FP4 QAT, DFlash, GCache,
  or production RL;
- preserve accidental checkpoint compatibility with the current custom
  `model_type="qwen3_omni_moe"` collision.

## 4. Considered approaches

### 4.1 Chosen: faithful baseline plus isolated experiments

Use official Qwen3-Omni as the verifiable baseline. Put Qwen3.5 and MiMo ideas in
separate profiles with separate schemas and manifests.

Benefits:

- clear provenance;
- official checkpoint and numerical tests are possible for Qwen3-Omni;
- experimental mechanisms can be ablated independently;
- legacy behavior remains available for regression;
- published and unpublished claims cannot be silently mixed.

Cost:

- requires explicit common interfaces;
- carries more than one profile during migration.

### 4.2 Rejected as the sole solution: wrap only Transformers

Wrapping official Transformers would produce the quickest Qwen3-Omni inference
compatibility, but it would bypass most custom training and architecture
research code. It also cannot solve the lack of Qwen3.5-Omni weights.

Official modules may still serve as the oracle or as an implementation backend
inside the reference profile.

### 4.3 Rejected: three independent model trees

Completely separate Qwen3-Omni, Qwen3.5-Omni, and MiMo trees would maximize
isolation but duplicate media, configuration, training, cache, and checkpoint
infrastructure. The maintenance cost is not justified before common contracts
are stable.

## 5. Deliverables

The documentation phase produces:

1. three evidence-backed research reports;
2. this unified architecture design;
3. one detailed implementation plan after written-spec review.

The implementation phase produces, in dependency order:

1. identity and profile contracts;
2. official Qwen3-Omni oracle tests;
3. sequence-preserving media and multimodal assembly;
4. position and cache infrastructure;
5. official-structure Qwen3-Omni modules;
6. Talker/codec streaming;
7. Qwen3.5-inspired modules;
8. MiMo-style experiments;
9. stage-aware training and evaluation.

## 6. Architecture profiles

### 6.1 `legacy_prototype`

Purpose:

- preserve current behavior and checkpoint loading;
- keep the current 111-test baseline;
- provide before/after quality and performance comparisons.

This profile remains explicitly experimental. It is not checkpoint-compatible
with any official Qwen model.

### 6.2 `qwen3_omni_reference`

Purpose:

- represent the official open Qwen3-Omni architecture;
- load pinned official configuration and checkpoints;
- provide golden processor, state-dict, logits, codebook, and waveform oracles.

Compatibility may only be claimed after the gates in section 13 pass.

### 6.3 `qwen35_omni_inspired`

Purpose:

- combine the public Qwen3.5 backbone with the open predecessor Omni pipeline;
- implement paper-disclosed AuT, timestamps, TM-RoPE, ARIA, and training stages;
- support controlled API black-box comparison.

This profile must never use Plus or Flash in its model class or checkpoint name.
Its manifest must state that exact Qwen3.5-Omni checkpoint compatibility is
false.

### 6.4 `mimo_v25_experimental`

Purpose:

- test Hybrid SWA/GA;
- test routed-only sparse MoE;
- test speculative MTP and media-serving optimizations;
- compare these mechanisms with full attention and DeltaNet.

This profile produces text only. Talker and Code2Wav remain part of the
Qwen3-Omni lineage.

## 7. Profile manifest and provenance

Every saved config and checkpoint must contain a profile manifest:

```yaml
architecture_profile: qwen35_omni_inspired
compatibility_level: paper-inspired
sources:
  backbone:
    name: Qwen3.5-35B-A3B
    revision: 59d61f3ce65a6d9863b86d2e96597125219dc754
  omni_pipeline:
    name: Qwen3-Omni-30B-A3B-Instruct
    revision: 26291f793822fb6be9555850f06dfe95f2d7e695
assumptions:
  codec: qwen3-omni predecessor proxy
exact_official_checkpoint_compatible: false
```

The manifest is validated on load and round-tripped through checkpoint saves.
Unknown profiles, incompatible tokenizers, or contradictory compatibility
claims fail before model construction.

## 8. Common contracts

The profiles share contracts, not concrete architecture assumptions.

### 8.1 `MediaSequence`

Logical fields:

```text
embeddings        [batch, media_tokens, hidden]
attention_mask    [batch, media_tokens]
modality          image | video | audio
grid              optional temporal/height/width grid metadata
timestamps        optional time per media unit
source_spans      mapping to source placeholder or conversation item
```

Rules:

- media token count is variable;
- missing media creates no fake summary token;
- corrupt referenced media follows the existing strict/skip policy;
- grid and timestamp units are profile-defined and validated;
- media tensors never silently truncate without structured diagnostics.

### 8.2 `SequenceAssembler`

Responsibilities:

- find media placeholders in tokenized conversations;
- replace each placeholder with the corresponding media sequence;
- support multiple media items;
- interleave audio and video by timestamp where the profile requires it;
- produce labels that ignore media positions;
- preserve text padding and conversation boundaries;
- emit source-to-output span mappings for debugging.

The assembler does not encode media and does not compute rotary embeddings.

### 8.3 `PositionBuilder`

Responsibilities:

- build 1D positions for the legacy profile;
- build Qwen3-Omni TM-RoPE T/H/W positions;
- support explicit timestamp tokens for the Qwen3.5-inspired profile;
- support profile-specific rotary dimensions and bases;
- guarantee monotonic, non-conflicting cross-modal positions.

It accepts assembled sequence metadata and returns position IDs without
mutating media or tokenizer state.

### 8.4 `DecoderState`

Logical state partitions:

```text
full_attention_kv
swa_kv
gdn_convolution_state
gdn_recurrent_matrix_state
talker_state
mtp_state
codec_state
processed_media_cache
```

A profile declares which partitions it uses. State belongs to one request and
cannot be shared across sessions unless an explicit immutable cache key proves
equivalence.

### 8.5 `ArchitectureSummary`

Every built model reports:

- layer types;
- attention/cache type per layer;
- total and active parameter estimates;
- routed/shared/dense parameter counts;
- tokenizer and embedding vocabulary sizes;
- profile and compatibility level;
- unsupported capabilities.

This replaces file-name-based parameter claims.

## 9. Data flow

### 9.1 Understanding and text generation

```text
conversation + media references
        │
profile tokenizer + strict media loader
        │
text tokens + variable-length MediaSequence objects
        │
SequenceAssembler
        │
PositionBuilder
        │
Thinker prefill ──> DecoderState
        │
cached text decode
```

Media encoders run during prefill only. Cached decode consumes the previous
state and new tokens, not the original media tensors.

### 9.2 Speech generation

```text
Thinker hidden/text stream
        │
Talker conditioning + optional voice prompt
        │
main RVQ codebook token
        │
MTP residual codebooks
        │
causal Code2Wav + codec state
        │
incremental waveform chunks
```

The Qwen3.5-inspired profile inserts ARIA scheduling before Talker token
selection. The Qwen3-Omni reference profile follows the open predecessor
semantics.

## 10. Migration order

### D0: identity and contracts

- assign a non-official model type to the legacy prototype;
- introduce profile manifests;
- make config conflicts fail-fast;
- snapshot architecture summaries for all existing YAMLs;
- keep legacy checkpoints readable through an explicit compatibility adapter.

### D1: official Qwen3-Omni oracle

- pin official revisions;
- add config/processor smoke tests;
- add optional checkpoint tests;
- capture official state keys, shapes, logits, code shapes, and media metadata.

### D2: media sequences and assembly

- implement `MediaSequence`;
- replace single-token adapters;
- implement placeholder replacement;
- add video and multi-item support;
- preserve current strict media diagnostics.

### D3: positions and cache

- implement `PositionBuilder`;
- add TM-RoPE and timestamp alignment;
- implement full-attention prefill/decode cache;
- prove cached/uncached parity;
- ensure media encoders run once.

### D4: Qwen3-Omni structure and speech path

- align Thinker structure and state dict;
- add Talker and MTP;
- add Code2Wav and streaming codec state;
- validate official checkpoint and numerical compatibility.

### D5: Qwen3.5-inspired profile

- use the public Qwen3.5 tokenizer/backbone schema;
- implement numerically correct GDN and recurrent state;
- implement 6.25 Hz AuT and explicit timestamps;
- implement ARIA with predecessor codec parameters clearly marked as proxies.

### D6: MiMo experimental profile

- implement SWA/GA and a strict `O(W)` SWA cache;
- implement routed-only SwiGLU MoE;
- add router/load observability;
- add expert parallel before scaling expert count;
- add verified speculative MTP.

### D7: stage-aware training and evaluation

- encoder/projector warmup;
- general multimodal training;
- progressive long-context training;
- distributed Stage-2 support;
- distillation/preference/RL modules only after stable SFT and evaluation.

Each phase gets its own spec, plan, tests, and review. Later phases do not
silently expand an earlier phase's scope.

## 11. Configuration rules

1. A profile owns its tokenizer and token semantics.
2. Bare multimodal token IDs are not accepted as an independent source of
   truth.
3. `top_k` must be smaller than `num_experts` for a model advertised as sparse.
4. Global booleans and layer-index lists cannot contradict each other.
5. Context length cannot exceed the trained/validated length recorded in the
   manifest without an explicit experimental override.
6. Optimized kernels may be optional, but their fallback must preserve
   numerical semantics.
7. Missing performance kernels change performance status, not architecture
   identity.
8. Profile-specific config keys are rejected by other profiles.

## 12. Error handling

### 12.1 Media

- omitted optional media remains valid;
- corrupt referenced media raises the existing structured `MediaLoadError` by
  default;
- explicit quarantine mode records sample ID, path, modality, and error;
- silent zero replacement is prohibited.

### 12.2 Architecture and checkpoint

- model-type collision fails before AutoConfig registration;
- missing/unexpected state keys are errors in compatibility tests;
- shape mismatch reports the profile, source revision, and exact key;
- paper-inspired checkpoints cannot set a checkpoint-compatible flag.

### 12.3 Runtime

- unsupported distributed modes fail before creating models or process groups;
- missing optional performance kernels produce one structured warning and
  record the fallback in runtime metadata;
- missing required semantic dependencies are fatal;
- cache/session ownership violations are fatal.

### 12.4 Numerical behavior

The existing non-finite checks remain active. New modules expose internal
diagnostics that synchronize through the established training-loop mechanism;
individual ranks do not raise before peers reach the same collective boundary.

## 13. Compatibility levels and acceptance gates

### 13.1 `legacy-prototype`

Requirements:

- current behavior is intentionally preserved;
- all existing tests pass;
- no official compatibility claim is present.

### 13.2 `structure-aligned`

Requirements:

- layer count, hidden dimensions, heads, experts, media sequence semantics,
  position encoding, and cache schema match the chosen open reference;
- config and architecture summary tests pass;
- no claim about checkpoint numerics is made.

### 13.3 `checkpoint-compatible`

Requirements:

- official config loads without semantic coercion;
- state-dict keys and tensor shapes match;
- official weights load without unexplained missing/unexpected keys;
- fixed FP32 processor outputs and logits pass a documented tolerance;
- cached and uncached decode agree;
- generation interfaces and output shapes match.

Within this roadmap, only the Qwen3-Omni reference profile targets this level.
MiMo also has open artifacts, but `mimo_v25_experimental` intentionally tests
selected mechanisms and is not designed as a MiMo checkpoint loader.

### 13.4 `paper-inspired`

Requirements:

- every component identifies its paper/source/proxy;
- unpublished parameters are listed as assumptions;
- property and numerical tests validate implemented semantics;
- checkpoint compatibility is explicitly false.

Qwen3.5-Omni remains at this level until official weights and configuration are
released.

### 13.5 `MiMo-style-experiment`

Requirements:

- SWA, MoE, MTP, and cache experiments are independently configurable;
- experiments do not use MiMo class or checkpoint names;
- comparisons report quality, memory, throughput, and active parameters;
- the tokenizer and media pipeline remain profile-specific.

## 14. Test strategy

### 14.1 Contract tests

- profile manifest validation and round-trip;
- tokenizer/profile mismatch;
- invalid MoE sparsity;
- contradictory layer configuration;
- architecture summary snapshots;
- model type isolation.

### 14.2 Golden-oracle tests

- official Qwen3-Omni config/schema;
- processor token, grid, mask, and position outputs;
- state key and shape inventory;
- FP32 logits on fixed tiny inputs;
- Talker and codec output shapes.

Large-weight tests are opt-in and never required for ordinary CPU collection.

### 14.3 Media and position tests

- media sequence lengths;
- image quadrant order;
- audio temporal order;
- dynamic FPS;
- audio/video temporal alignment;
- cross-modal position continuity;
- placeholder source-span mapping;
- media ablation and shuffle.

### 14.4 Cache tests

- cached/uncached logits;
- exact greedy token parity;
- GDN recurrent-state parity;
- SWA window and strict `O(W)` storage;
- one-time media encoding;
- chunked/offline encoder equivalence;
- session isolation.

FP32 cache parity starts with a maximum logit error target of `1e-5`. Profiles
may tighten this based on their official oracle.

### 14.5 MoE and MTP tests

- normalized top-k parity;
- active/total parameter ratio;
- expert token counts, entropy, and max/mean load;
- expert-parallel parity;
- teacher-forced future-token losses;
- speculative accept/reject distribution preservation;
- rejection state repair.

### 14.6 Training tests

- freeze/unfreeze assertions;
- tiny-set overfit;
- deterministic checkpoint resume;
- distributed sampler coverage;
- text and per-modality regression;
- non-finite synchronization.

### 14.7 Streaming and performance tests

On documented hardware and inputs:

- TTFT;
- TTFC;
- RTF;
- decode tokens per second;
- peak device memory;
- 1/4/8 concurrency;
- interruption and backpressure;
- P50/P90 latency for media-cache hit/miss.

Paper latency values are context, not pass/fail thresholds.

## 15. Documentation rules

Every report and implementation note distinguishes:

- paper-explicit;
- public config/source;
- local code fact;
- reasonable inference;
- unpublished/unreproducible.

External links target pinned revisions when possible. Benchmark numbers include
the model variant, precision, hardware, context, and source. Approximate paper
totals are not silently converted into exact configuration values.

## 16. Rollout and compatibility

1. Introduce contracts without changing the legacy path.
2. Add the reference profile and oracle tests.
3. Migrate one media modality at a time behind explicit profile selection.
4. Add cache before any long-context or MTP performance claim.
5. Keep old checkpoint loading in a dedicated adapter with deprecation
   metadata; do not mutate old checkpoints in place.
6. Make new profiles opt-in until their acceptance level is documented.
7. Update README capability tables whenever a profile changes level.

No work in this roadmap deletes user data or automatically downloads large
weights. Weight-based verification is explicit and opt-in.

## 17. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Official and experimental names are confused | independent model types and manifests |
| A large rewrite regresses P0 correctness | legacy profile and continuous full-suite tests |
| Media refactor hides silent truncation | structured spans, masks, and diagnostics |
| Cache improves speed but changes outputs | mandatory cached/uncached parity |
| SWA is implemented with full-size storage | explicit cache-byte and `O(W)` tests |
| MoE is sparse in name only | `top_k < experts`, active-parameter and load tests |
| Qwen3.5 proxy values become “official” | provenance fields and paper-inspired label |
| MiMo inference-only router is copied into training | retain trainable router until a reproducible recipe exists |
| Long-context claims are config-only | progressive length training and evaluation gates |
| Optional kernels change semantics | numerical fallback tests and runtime metadata |

## 18. Design completion criteria

The design phase is complete when:

- all three research reports exist and cite primary sources;
- this specification contains no placeholders or unresolved choices;
- profile boundaries and compatibility labels are unambiguous;
- work is decomposed into dependency-ordered phases;
- error handling and tests cover architecture, training, and streaming;
- the user reviews the written specification;
- a detailed implementation plan is produced with `writing-plans`.

Implementation starts only after the written-spec review gate.
