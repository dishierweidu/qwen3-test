# P0 Training Correctness Design

## Goal

Make the current Qwen3-Omni research implementation fail loudly on corrupted or non-finite training data, preserve the intended multimodal attention and supervision masks, remove ambiguous double shared-expert execution, and expose reproducible parameter counts.

## Scope

This change deliberately does not replace the placeholder one-token vision/audio encoders or implement Talker/Code2Wav. It stabilizes the current training baseline before the architecture-faithful Omni rewrite.

## Data and supervision

Stage-2 examples are tokenized as separate prompt and target spans. Prompt and padding positions receive label `-100`; target tokens remain supervised. If the combined sequence exceeds the configured length, target tokens receive priority and the oldest prompt tokens are removed first.

Missing optional media is valid. A referenced but unreadable file raises `MediaLoadError` by default. Corpus-audit workflows may explicitly enable `skip_bad_media`, which records structured metadata and marks the modality absent.

## Multimodal wrapper

The wrapper prepends image and audio tokens while preserving the original text attention mask. The resulting mask is `[image_present, audio_present, text_attention_mask...]`. Modality features are checked for NaN/Inf before they enter the Thinker.

## MoE contract

`ThinkerDecoderLayer.shared_mlp` remains the only always-active dense FFN path. `Qwen3OmniMoeMLP` contains routed experts only. Legacy configs enabling an internal shared expert are rejected rather than silently changing parameter count or output scale. Selected top-k router scores are renormalized per token.

## Numerical behavior

Training and evaluation check loss, auxiliary losses, logits, and gradients for NaN/Inf. Distributed ranks synchronize a bad-state flag and raise `NonFiniteTrainingError` on the same batch. Underscore-prefixed batch fields are diagnostic metadata and are not forwarded to `model.forward`.

## Verification

The regression suite covers supervision masks, media errors, multimodal attention composition, MoE routing, non-finite outputs and gradients, and unique/active parameter estimates. The README documents environment setup, tests, known architectural limitations, and the next Omni milestones.
