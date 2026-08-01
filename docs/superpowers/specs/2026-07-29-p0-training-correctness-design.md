# P0 Training Correctness Design

## Goal

Make the current Qwen3-Omni research implementation fail loudly on corrupted or non-finite training data, preserve the intended multimodal attention/loss masks, remove ambiguous double shared-expert execution, and expose reproducible parameter counts.

## Scope

This change deliberately does not replace the placeholder one-token vision/audio encoders or implement Talker/Code2Wav. It stabilizes the current training baseline before the architecture-faithful Omni rewrite.

## Architecture decisions

1. **Explicit Stage-2 supervision**: the collator constructs labels only for `target_text`; prompt and padding tokens use `-100`.
2. **Strict media policy by default**: image/audio decoding errors raise `MediaLoadError`. An opt-in `skip_bad_media=True` mode marks the sample as missing and emits structured metadata instead of silently treating corruption as a real zero-valued modality.
3. **Metadata is not forwarded to models**: batch keys prefixed with `_` carry sample IDs and media errors. The training loop separates these before `model(**batch)` and includes them in failures.
4. **Attention mask preservation**: the multimodal wrapper prepends image/audio availability bits and preserves the original text mask exactly.
5. **Single shared path**: decoder layers already run `shared_mlp`; the routed MoE module therefore contains routed experts only. Legacy `use_shared_expert=true` is rejected with a clear configuration error.
6. **Synchronized fail-fast numerics**: rank-local modules do not throw on NaN/Inf inside the distributed forward graph. Non-finite modality or router states propagate to loss/logits/auxiliary losses, where the training loop all-reduces a failure flag and raises `NonFiniteTrainingError` on every rank.
7. **Parameter accounting**: a utility reports total/trainable parameters and an MoE-aware active-per-token estimate. It must state that the number is an estimate and identify routed modules included in the estimate.

## Error handling

- Media decoding errors include modality, path, sample ID, and original exception.
- Distributed non-finite detection occurs at the model-output and gradient boundaries. It uses an all-reduced flag so every rank raises on the same step; rank-local modules avoid numerical exceptions before peers reach the same collectives.
- The error message includes batch index and `_sample_ids` when available.
- No bare exception silently converts invalid input into a training sample.

## Tests

- Prompt and padding positions are excluded from Stage-2 loss.
- Missing optional modalities remain valid, while corrupt referenced files fail by default.
- Text padding mask survives multimodal prefix construction.
- Top-k weights sum to one when normalization is enabled.
- Internal shared experts cannot be enabled alongside the decoder's shared MLP.
- Non-finite modality/router state propagates to model output, then raises at the synchronized boundary and prevents optimizer updates.
- Parameter statistics match toy dense and MoE modules.

## Compatibility

The Stage-2 model forward signature remains unchanged. New underscore-prefixed collator metadata is removed by the training loop before model invocation. Legacy MoE YAML files are updated to `use_shared_expert: false`.
