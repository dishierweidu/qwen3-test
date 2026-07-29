# P0 Training Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current multimodal pretraining baseline fail loudly on invalid data and preserve correct masks, routing scales, and parameter accounting.

**Architecture:** Keep the existing trainer and placeholder multimodal encoders. Correct Stage-2 supervision, preserve text padding masks, make the MoE routed-only because DecoderLayer already owns the shared dense path, and add fail-fast numerical checks plus parameter inspection.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, torchaudio, pytest.

## Global Constraints

- Do not implement the full Qwen3-Omni encoder/Talker rewrite in this change.
- Every behavior change requires a failing regression test first.
- Corrupt media and non-finite tensors must never be silently converted into valid training samples.

---

### Task 1: Stage-2 supervision and media validation

- [x] Preserve target tokens when prompt text exceeds the sequence budget.
- [x] Mask prompt and padding labels with `-100`.
- [x] Raise structured media-loading errors by default.

### Task 2: Multimodal attention masks

- [x] Preserve text `attention_mask` after prepending modality tokens.
- [x] Fail on non-finite modality features.

### Task 3: MoE routing invariants

- [x] Renormalize selected top-k scores per token.
- [x] Remove the second internal shared expert path.
- [x] Reject legacy shared-expert configuration.

### Task 4: Numerical fail-fast behavior

- [ ] Reject non-finite loss, logits, and gradients before optimizer updates.

### Task 5: Parameter accounting and documentation

- [ ] Add total/trainable/active parameter reports.
- [ ] Add reproducible setup and test instructions.
