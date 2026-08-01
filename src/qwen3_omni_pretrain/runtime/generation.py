from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

import torch

from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime.capabilities import (
    CacheCapabilityError,
    CacheErrorCode,
    validate_generation_operations,
)
from qwen3_omni_pretrain.runtime.protocols import (
    CacheCapableModel,
    CausalLMOutput,
    ModelDecodeInputs,
    ModelPrefillInputs,
    legacy_position_ids,
)
from qwen3_omni_pretrain.runtime.state import (
    DecoderState,
    LegacyPositionCursor,
    StateOwner,
)


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class GenerationRequest:
    display_request_id: str
    prefill_inputs: ModelPrefillInputs
    max_new_tokens: int
    eos_token_id: int | None
    num_beams: int = 1
    cancellation: CancellationToken | None = None

    def __post_init__(self) -> None:
        if type(self.display_request_id) is not str:
            raise TypeError("display_request_id must be a string")
        if not self.display_request_id.strip():
            raise ValueError("display_request_id must be nonblank")
        if not isinstance(self.prefill_inputs, ModelPrefillInputs):
            raise TypeError("prefill_inputs must be ModelPrefillInputs")
        if type(self.max_new_tokens) is not int:
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.eos_token_id is not None:
            if type(self.eos_token_id) is not int:
                raise TypeError("eos_token_id must be an integer or None")
            if self.eos_token_id < 0:
                raise ValueError("eos_token_id must be non-negative")
        validate_generation_operations(num_beams=self.num_beams)
        if self.cancellation is not None and not isinstance(
            self.cancellation,
            CancellationToken,
        ):
            raise TypeError("cancellation must be CancellationToken or None")


@dataclass(frozen=True)
class GenerationCheckpoint:
    decoder_state: DecoderState
    pending_token_id: torch.Tensor | None

    def __post_init__(self) -> None:
        if not isinstance(self.decoder_state, DecoderState):
            raise TypeError("decoder_state must be DecoderState")
        state = self.decoder_state.clone_detached()
        pending = self.pending_token_id
        if pending is not None:
            if not isinstance(pending, torch.Tensor):
                raise TypeError("pending_token_id must be a torch.Tensor or None")
            if pending.dtype is not torch.long:
                raise TypeError("pending_token_id must have dtype torch.long")
            if pending.shape != (state.batch_size, 1):
                raise ValueError("pending_token_id must have shape [B, 1]")
            if pending.device != state.device:
                raise ValueError("pending token and state must share a device")
            if bool((pending < 0).any().item()):
                raise ValueError("pending_token_id must be non-negative")
            pending = pending.detach().clone()
        object.__setattr__(self, "decoder_state", state)
        object.__setattr__(self, "pending_token_id", pending)


@dataclass(frozen=True)
class GenerationStep:
    emitted_token_id: torch.Tensor
    emitted_index: int
    checkpoint: GenerationCheckpoint
    finished: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, GenerationCheckpoint):
            raise TypeError("checkpoint must be GenerationCheckpoint")
        checkpoint = GenerationCheckpoint(
            self.checkpoint.decoder_state,
            self.checkpoint.pending_token_id,
        )
        object.__setattr__(self, "checkpoint", checkpoint)
        if not isinstance(self.emitted_token_id, torch.Tensor):
            raise TypeError("emitted_token_id must be a torch.Tensor")
        if self.emitted_token_id.dtype is not torch.long:
            raise TypeError("emitted_token_id must have dtype torch.long")
        if self.emitted_token_id.shape != (
            self.checkpoint.decoder_state.batch_size,
            1,
        ):
            raise ValueError("emitted_token_id must have shape [B, 1]")
        if self.emitted_token_id.device != self.checkpoint.decoder_state.device:
            raise ValueError("emitted token and checkpoint must share a device")
        if type(self.emitted_index) is not int or self.emitted_index < 0:
            raise ValueError("emitted_index must be a non-negative integer")
        if not isinstance(self.finished, torch.Tensor):
            raise TypeError("finished must be a torch.Tensor")
        if self.finished.dtype is not torch.bool or self.finished.shape != (
            self.checkpoint.decoder_state.batch_size,
        ):
            raise ValueError("finished must be a bool [B] tensor")
        if self.finished.device != self.checkpoint.decoder_state.device:
            raise ValueError("finished and checkpoint must share a device")
        object.__setattr__(
            self,
            "emitted_token_id",
            self.emitted_token_id.detach().clone(),
        )
        object.__setattr__(self, "finished", self.finished.detach().clone())


@dataclass(frozen=True)
class GenerationResult:
    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    checkpoint: GenerationCheckpoint
    finish_reason: Literal["eos", "length", "cancelled"]
    prefill_calls: int
    decode_calls: int

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, GenerationCheckpoint):
            raise TypeError("checkpoint must be GenerationCheckpoint")
        checkpoint = GenerationCheckpoint(
            self.checkpoint.decoder_state,
            self.checkpoint.pending_token_id,
        )
        object.__setattr__(self, "checkpoint", checkpoint)
        for name in ("prompt_ids", "generated_ids"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.dtype is not torch.long or tensor.ndim != 2:
                raise TypeError(f"{name} must have dtype long and shape [B, S]")
            if tensor.shape[0] != self.checkpoint.decoder_state.batch_size:
                raise ValueError(f"{name} batch size differs from checkpoint")
            if tensor.device != self.checkpoint.decoder_state.device:
                raise ValueError(f"{name} and checkpoint must share a device")
        if self.prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must not be empty")
        if self.finish_reason not in {"eos", "length", "cancelled"}:
            raise ValueError("invalid finish_reason")
        for name in ("prefill_calls", "decode_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        pending = self.checkpoint.pending_token_id
        if self.generated_ids.shape[1] == 0:
            if pending is not None:
                raise ValueError("empty generation cannot have a pending token")
        elif pending is None or not torch.equal(
            pending,
            self.generated_ids[:, -1:],
        ):
            raise ValueError(
                "checkpoint pending token must equal generated_ids[:, -1:]"
            )
        object.__setattr__(
            self,
            "prompt_ids",
            self.prompt_ids.detach().clone(),
        )
        object.__setattr__(
            self,
            "generated_ids",
            self.generated_ids.detach().clone(),
        )


class GenerationExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        checkpoint: GenerationCheckpoint,
        prefill_calls: int,
        decode_calls: int,
    ) -> None:
        super().__init__(message)
        self.checkpoint = checkpoint
        self.prefill_calls = prefill_calls
        self.decode_calls = decode_calls


class LegacyGreedyPrefillDecodeEngine:
    def __init__(self, model: CacheCapableModel) -> None:
        if not isinstance(model, CacheCapableModel):
            raise TypeError("model must implement CacheCapableModel")
        self.model = model

    def _validate_engine_request(self, request: GenerationRequest) -> torch.Tensor:
        prefill = request.prefill_inputs
        if prefill.input_ids is None:
            raise ValueError("greedy engine requires prompt input_ids")
        if prefill.batch_size != 1:
            raise ValueError("legacy greedy engine supports batch size one")
        if not bool(prefill.key_valid_mask.any().item()):
            raise ValueError("prompt must contain at least one valid token")
        if not bool(prefill.key_valid_mask[0, -1].item()):
            raise ValueError("prompt final query slot must be valid")
        if prefill.position_batch is None:
            raise ValueError("cache engine requires explicit prompt positions")
        concrete = getattr(self.model, "thinker", self.model)
        thinker_config = getattr(concrete, "thinker_cfg", None)
        max_positions = getattr(
            thinker_config,
            "max_position_embeddings",
            None,
        )
        prefix_length = getattr(
            self.model,
            "generation_prefix_storage_length",
            0,
        )
        if type(prefix_length) is not int or prefix_length < 0:
            raise TypeError(
                "generation_prefix_storage_length must be non-negative int"
            )
        if type(max_positions) is int:
            available_text_positions = max_positions - prefix_length
            try:
                prompt_positions = legacy_position_ids(
                    prefill.position_batch,
                    prefill.key_valid_mask,
                    max_position_embeddings=available_text_positions,
                )
            except ValueError as error:
                if (
                    available_text_positions <= 0
                    or "max_position_embeddings" in str(error)
                ):
                    raise CacheCapabilityError(
                        CacheErrorCode.CONTEXT_OVERFLOW,
                        str(error),
                    ) from error
                raise
            valid_positions = prompt_positions.masked_select(
                prefill.key_valid_mask
            )
            next_storage_position = max(
                prefill.query_length + prefix_length,
                int(valid_positions.max().item()) + prefix_length + 1,
            )
            if (
                next_storage_position + request.max_new_tokens - 1
                > max_positions
            ):
                raise CacheCapabilityError(
                    CacheErrorCode.CONTEXT_OVERFLOW,
                    "prompt position cursor plus requested output exceeds "
                    "model context",
                )
        return prefill.input_ids

    @staticmethod
    def _is_cancelled(request: GenerationRequest) -> bool:
        return (
            request.cancellation is not None
            and request.cancellation.is_cancelled
        )

    @staticmethod
    def _result(
        *,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        checkpoint: GenerationCheckpoint,
        finish_reason: Literal["eos", "length", "cancelled"],
        prefill_calls: int,
        decode_calls: int,
    ) -> GenerationResult:
        return GenerationResult(
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            checkpoint=checkpoint,
            finish_reason=finish_reason,
            prefill_calls=prefill_calls,
            decode_calls=decode_calls,
        )

    @staticmethod
    def _decode_inputs(checkpoint: GenerationCheckpoint) -> ModelDecodeInputs:
        state = checkpoint.decoder_state
        pending = checkpoint.pending_token_id
        if pending is None:
            raise ValueError("decode requires a pending token")
        if state.position is None or not isinstance(
            state.position.continuation,
            LegacyPositionCursor,
        ):
            raise ValueError("decode checkpoint requires a legacy position cursor")
        cursor = state.position.continuation
        mask = torch.ones(
            (state.batch_size, 1),
            dtype=torch.bool,
            device=state.device,
        )
        position = PositionBatch(
            position_ids=cursor.next_storage_position.view(1, state.batch_size, 1),
            rope_deltas=state.position.cached.rope_deltas,
            axis_names=("sequence",),
        )
        return ModelDecodeInputs(
            token_ids=pending,
            current_key_valid_mask=mask,
            position_batch=position,
            decoder_state=state,
        )

    @staticmethod
    def _validate_resume(
        request: GenerationRequest,
        resume: GenerationResult,
        prompt_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        GenerationCheckpoint,
        int,
        int,
    ]:
        if not isinstance(resume, GenerationResult):
            raise TypeError("resume must be GenerationResult or None")
        if not torch.equal(resume.prompt_ids, prompt_ids):
            raise ValueError("resume prompt does not match the request")
        if (
            resume.checkpoint.decoder_state.owner.display_request_id
            != request.display_request_id
        ):
            raise ValueError("resume display request ID does not match")
        if resume.generated_ids.shape[1] > request.max_new_tokens:
            raise ValueError("resume generation exceeds requested length")
        return (
            resume.generated_ids.detach().clone(),
            GenerationCheckpoint(
                resume.checkpoint.decoder_state,
                resume.checkpoint.pending_token_id,
            ),
            resume.prefill_calls,
            resume.decode_calls,
        )

    def generate(
        self,
        request: GenerationRequest,
        *,
        resume: GenerationResult | None = None,
        call_budget: int | None = None,
    ) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        if call_budget is not None and (
            type(call_budget) is not int or call_budget <= 0
        ):
            raise ValueError("call_budget must be a positive integer or None")
        prompt_ids = self._validate_engine_request(request)
        if resume is None:
            concrete = getattr(self.model, "thinker", self.model)
            owner_factory = getattr(concrete, "create_state_owner", None)
            owner = (
                owner_factory(request.display_request_id)
                if callable(owner_factory)
                else StateOwner.fresh(request.display_request_id)
            )
            if not isinstance(owner, StateOwner):
                raise TypeError("model owner factory must return StateOwner")
            generated_ids = torch.empty(
                (1, 0),
                dtype=torch.long,
                device=prompt_ids.device,
            )
            checkpoint = GenerationCheckpoint(
                DecoderState.empty(owner, batch_size=1, device=prompt_ids.device),
                None,
            )
            prefill_calls = 0
            decode_calls = 0
        else:
            (
                generated_ids,
                checkpoint,
                prefill_calls,
                decode_calls,
            ) = self._validate_resume(request, resume, prompt_ids)
            owner = checkpoint.decoder_state.owner

        if resume is not None and resume.finish_reason == "eos":
            return self._result(
                prompt_ids=prompt_ids,
                generated_ids=generated_ids,
                checkpoint=checkpoint,
                finish_reason="eos",
                prefill_calls=prefill_calls,
                decode_calls=decode_calls,
            )

        if self._is_cancelled(request):
            return self._result(
                prompt_ids=prompt_ids,
                generated_ids=generated_ids,
                checkpoint=checkpoint,
                finish_reason="cancelled",
                prefill_calls=prefill_calls,
                decode_calls=decode_calls,
            )
        if generated_ids.shape[1] >= request.max_new_tokens:
            return self._result(
                prompt_ids=prompt_ids,
                generated_ids=generated_ids,
                checkpoint=checkpoint,
                finish_reason="length",
                prefill_calls=prefill_calls,
                decode_calls=decode_calls,
            )

        calls_this_run = 0
        if generated_ids.shape[1] == 0:
            prefill_calls += 1
            calls_this_run += 1
            try:
                output = self.model.prefill(
                    inputs=request.prefill_inputs,
                    owner=owner,
                    use_cache=True,
                )
            except Exception as error:
                raise GenerationExecutionError(
                    "prefill failed before candidate commit",
                    checkpoint=checkpoint,
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                ) from error
            if not isinstance(output, CausalLMOutput) or output.decoder_state is None:
                raise GenerationExecutionError(
                    "prefill returned an invalid cache result",
                    checkpoint=checkpoint,
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )
            if self._is_cancelled(request):
                return self._result(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    checkpoint=checkpoint,
                    finish_reason="cancelled",
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )
            output.decoder_state.assert_owner(owner)
            emitted = output.logits[:, -1].argmax(dim=-1, keepdim=True).to(
                dtype=torch.long
            )
            checkpoint = GenerationCheckpoint(output.decoder_state, emitted)
            generated_ids = torch.cat((generated_ids, emitted), dim=1)
            if request.eos_token_id is not None and int(emitted.item()) == request.eos_token_id:
                return self._result(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    checkpoint=checkpoint,
                    finish_reason="eos",
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )
            if generated_ids.shape[1] >= request.max_new_tokens:
                return self._result(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    checkpoint=checkpoint,
                    finish_reason="length",
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )
            if call_budget is not None and calls_this_run >= call_budget:
                return self._result(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    checkpoint=checkpoint,
                    finish_reason="cancelled",
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )

        while generated_ids.shape[1] < request.max_new_tokens:
            if self._is_cancelled(request):
                return self._result(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    checkpoint=checkpoint,
                    finish_reason="cancelled",
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )
            if call_budget is not None and calls_this_run >= call_budget:
                return self._result(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    checkpoint=checkpoint,
                    finish_reason="cancelled",
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )
            decode_calls += 1
            calls_this_run += 1
            try:
                output = self.model.decode(
                    inputs=self._decode_inputs(checkpoint),
                    owner=owner,
                )
            except Exception as error:
                raise GenerationExecutionError(
                    "decode failed before candidate commit",
                    checkpoint=checkpoint,
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                ) from error
            if not isinstance(output, CausalLMOutput) or output.decoder_state is None:
                raise GenerationExecutionError(
                    "decode returned an invalid cache result",
                    checkpoint=checkpoint,
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )
            if self._is_cancelled(request):
                return self._result(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    checkpoint=checkpoint,
                    finish_reason="cancelled",
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )
            output.decoder_state.assert_owner(owner)
            emitted = output.logits[:, -1].argmax(dim=-1, keepdim=True).to(
                dtype=torch.long
            )
            checkpoint = GenerationCheckpoint(output.decoder_state, emitted)
            generated_ids = torch.cat((generated_ids, emitted), dim=1)
            if request.eos_token_id is not None and int(emitted.item()) == request.eos_token_id:
                return self._result(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    checkpoint=checkpoint,
                    finish_reason="eos",
                    prefill_calls=prefill_calls,
                    decode_calls=decode_calls,
                )

        return self._result(
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            checkpoint=checkpoint,
            finish_reason="length",
            prefill_calls=prefill_calls,
            decode_calls=decode_calls,
        )


__all__ = [
    "CancellationToken",
    "GenerationCheckpoint",
    "GenerationExecutionError",
    "GenerationRequest",
    "GenerationResult",
    "GenerationStep",
    "LegacyGreedyPrefillDecodeEngine",
]
