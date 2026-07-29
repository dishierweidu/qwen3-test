# Qwen3 Omni Architecture Remediation P0 Plan

## Goal
Convert the current research implementation into a structurally correct multimodal foundation before scaling model size.

## Phase 1 - Training correctness fixes

### 1. Multimodal attention mask
- Preserve text attention_mask when adding image/audio/video tokens.
- Add regression test: changing padding token ids must not change valid-token logits.

### 2. Loss masking
- Separate attention_mask and loss_mask.
- Instruction data only computes loss on target response tokens.
- Padding tokens always use -100 labels.

### 3. Data failure handling
- Remove silent zero replacement for corrupted media.
- Add quarantine records containing sample id, path and exception.

### 4. Numerical stability
- Training NaN should stop execution with diagnostics.
- nan_to_num is only allowed for controlled inference fallback.

### 5. MoE validation
- Verify top-k gate normalization.
- Add parameter counter for total parameters and active parameters.

## Phase 2 - Architecture alignment

- Replace single-token image/audio projection with sequence encoders.
- Introduce unified multimodal sequence builder.
- Add temporal and spatial position metadata.
- Implement Thinker/Talker/Codec interfaces.

## Acceptance criteria

- Stage2 loss is reproducible with padding perturbation.
- Corrupted samples are observable and recoverable.
- Parameter statistics match configuration.
- Multimodal inputs preserve temporal/spatial information.
