from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from qwen3_omni_pretrain.multimodal.io import DecodedMedia
from qwen3_omni_pretrain.multimodal.prefill import MultimodalPrefillPipeline
from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.cache_adapter import (
    Qwen35CacheAdapter,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    Qwen35InspiredConfig,
)
from qwen3_omni_pretrain.runtime.capabilities import CacheSupport
from qwen3_omni_pretrain.runtime.protocols import (
    CausalLMOutput,
    ModelDecodeInputs,
    ModelPrefillInputs,
)
from qwen3_omni_pretrain.runtime.state import DecoderState, StateOwner


def unique_parameters_by_identity(
    parameters: Iterable[nn.Parameter],
) -> tuple[nn.Parameter, ...]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return tuple(result)


def _projector_parameter_ids(module: nn.Module) -> set[int]:
    result: set[int] = set()
    for name, child in module.named_modules():
        if name and name.rsplit(".", 1)[-1] == "projector":
            result.update(id(parameter) for parameter in child.parameters())
    return result


def assert_disjoint_complete_groups(
    module: nn.Module,
    groups: Mapping[str, tuple[nn.Parameter, ...]],
) -> Mapping[str, tuple[nn.Parameter, ...]]:
    expected_names = {
        "thinker",
        "vision_encoder",
        "audio_encoder",
        "projector",
    }
    if set(groups) != expected_names:
        raise ValueError("Thinker parameter groups must use the four stable names")
    all_ids = [id(parameter) for values in groups.values() for parameter in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Thinker parameter groups must be pairwise disjoint")
    model_ids = {id(parameter) for parameter in module.parameters()}
    if set(all_ids) != model_ids:
        raise ValueError("Thinker parameter groups must cover every parameter exactly once")
    return dict(groups)


def _resolve_owner(
    *,
    owner: StateOwner | None,
    request_id: str | None,
    decoder_state: DecoderState | None,
) -> StateOwner:
    if owner is not None and not isinstance(owner, StateOwner):
        raise TypeError("owner must be StateOwner or None")
    if request_id is not None and (
        not isinstance(request_id, str) or not request_id.strip()
    ):
        raise ValueError("request_id must be non-empty when provided")
    if decoder_state is not None:
        if not isinstance(decoder_state, DecoderState):
            raise TypeError("decoder_state must be DecoderState")
        resolved = decoder_state.owner if owner is None else owner
        decoder_state.assert_owner(resolved)
    else:
        resolved = owner or StateOwner.fresh(request_id or "qwen35-request")
    if request_id is not None and resolved.display_request_id != request_id:
        raise ValueError("request_id contradicts the state owner")
    return resolved


def _default_positions(mask: torch.Tensor) -> PositionBatch:
    values = mask.to(torch.long).cumsum(dim=1) - 1
    values = values.masked_fill(~mask.bool(), 0)
    return PositionBatch(
        position_ids=values.unsqueeze(0),
        rope_deltas=torch.zeros(
            mask.shape[0],
            1,
            dtype=torch.long,
            device=mask.device,
        ),
        axis_names=("sequence",),
    )


def _as_causal_output(
    raw: Mapping[str, object],
) -> CausalLMOutput:
    return CausalLMOutput(
        logits=raw["logits"],
        loss=raw.get("loss"),
        ce_loss=raw.get("ce_loss"),
        aux_loss=raw.get("aux_loss"),
        decoder_state=raw.get("decoder_state"),
        hidden_states=raw.get("hidden_states"),
    )


class Qwen35InspiredThinker(nn.Module):
    cache_support = CacheSupport(
        incremental_decode_state=True,
        streaming_generation=True,
        beam_search=False,
        state_truncate=False,
        speculative_decode=False,
    )

    def __init__(
        self,
        *,
        backbone: nn.Module,
        prefill_pipeline: MultimodalPrefillPipeline,
        cache_adapter: Qwen35CacheAdapter,
        profile_config: Qwen35InspiredConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be an nn.Module")
        if not isinstance(prefill_pipeline, MultimodalPrefillPipeline):
            raise TypeError("prefill_pipeline must be MultimodalPrefillPipeline")
        if not isinstance(cache_adapter, Qwen35CacheAdapter):
            raise TypeError("cache_adapter must be Qwen35CacheAdapter")
        self.backbone = backbone
        self.prefill_pipeline = prefill_pipeline
        self.cache_adapter = cache_adapter
        self.config = profile_config if profile_config is not None else backbone.config

    @classmethod
    def from_public_model(
        cls,
        public: nn.Module,
        *,
        prefill_pipeline: MultimodalPrefillPipeline,
        profile_config: Qwen35InspiredConfig | None = None,
    ) -> Qwen35InspiredThinker:
        return cls(
            backbone=public,
            prefill_pipeline=prefill_pipeline,
            cache_adapter=Qwen35CacheAdapter(public.config),
            profile_config=profile_config,
        )

    @property
    def layers(self) -> nn.ModuleList:
        return self.backbone.model.layers

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.backbone.model.embed_tokens

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def named_parameter_groups(self) -> Mapping[str, tuple[nn.Parameter, ...]]:
        projector_ids = _projector_parameter_ids(self.prefill_pipeline)
        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        vision_parameters = unique_parameters_by_identity(
            parameter
            for encoder in (
                self.prefill_pipeline.image_encoder,
                self.prefill_pipeline.video_encoder,
            )
            for parameter in encoder.parameters()
            if id(parameter) not in projector_ids
        )
        audio_parameters = unique_parameters_by_identity(
            parameter
            for parameter in self.prefill_pipeline.audio_encoder.parameters()
            if id(parameter) not in projector_ids
        )
        projector_parameters = unique_parameters_by_identity(
            parameter
            for parameter in self.prefill_pipeline.parameters()
            if id(parameter) in projector_ids
        )
        groups = {
            "thinker": unique_parameters_by_identity(
                parameter
                for parameter in self.parameters()
                if id(parameter) in backbone_ids
            ),
            "vision_encoder": vision_parameters,
            "audio_encoder": audio_parameters,
            "projector": projector_parameters,
        }
        return assert_disjoint_complete_groups(self, groups)

    @staticmethod
    def _cache_precondition(mask: torch.Tensor) -> None:
        valid = mask.bool()
        if not bool(valid.all().item()) or not bool(
            (valid.sum(dim=1) == valid.sum(dim=1)[0]).all().item()
        ):
            raise ValueError(
                "Qwen3.5 native cache requires an equal-length batch without padding"
            )

    def _run_backbone(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_batch: PositionBatch,
        labels: torch.Tensor | None,
        decoder_state: DecoderState | None,
        owner: StateOwner,
        use_cache: bool,
    ) -> dict[str, object]:
        if type(use_cache) is not bool:
            raise TypeError("use_cache must be a boolean")
        current_mask = attention_mask.bool()
        query_length = current_mask.shape[1]
        native = None
        full_mask = current_mask
        full_positions = position_batch.position_ids
        cache_position: torch.Tensor | None = None
        if decoder_state is not None:
            decoder_state.assert_owner(owner)
            native = self.cache_adapter.to_native(
                decoder_state,
                request_id=owner.display_request_id,
            )
            if decoder_state.position is None:
                raise ValueError("cached decode requires position history")
            full_mask = torch.cat(
                (decoder_state.position.key_valid_mask, current_mask),
                dim=1,
            )
            full_positions = torch.cat(
                (
                    decoder_state.position.cached.position_ids,
                    position_batch.position_ids,
                ),
                dim=2,
            )
            start = int(decoder_state.seen_tokens[0].item())
            cache_position = torch.arange(
                start,
                start + query_length,
                dtype=torch.long,
                device=attention_mask.device,
            )
            use_cache = True
        elif use_cache:
            self._cache_precondition(current_mask)
            cache_position = torch.arange(
                query_length,
                dtype=torch.long,
                device=attention_mask.device,
            )

        public_positions = position_batch.position_ids
        if public_positions.shape[0] == 1:
            public_positions = public_positions[0]
        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            position_ids=public_positions,
            labels=labels,
            past_key_values=native,
            use_cache=use_cache,
            cache_position=cache_position,
            output_hidden_states=True,
            return_dict=True,
        )
        next_state = None
        if use_cache:
            if outputs.past_key_values is None:
                raise RuntimeError("public Qwen3.5 backbone did not return cache state")
            seen_tokens = full_mask.sum(dim=1).to(torch.long)
            next_state = self.cache_adapter.from_native(
                outputs.past_key_values,
                request_id=owner.display_request_id,
                seen_tokens=seen_tokens,
                key_valid_mask=full_mask,
                position_ids=full_positions,
            )._derive(owner=owner)
        hidden_states = (
            None
            if outputs.hidden_states is None
            else tuple(outputs.hidden_states)
        )
        loss = getattr(outputs, "loss", None)
        aux_loss = getattr(outputs, "aux_loss", None)
        return {
            "logits": outputs.logits,
            "loss": loss,
            "ce_loss": loss,
            "aux_loss": aux_loss,
            "decoder_state": next_state,
            "hidden_states": hidden_states,
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoded_media: Sequence[DecodedMedia] = (),
        labels: torch.LongTensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_batch: PositionBatch | None = None,
        decoder_state: DecoderState | None = None,
        owner: StateOwner | None = None,
        request_id: str | None = None,
        use_cache: bool = False,
    ) -> dict[str, object]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one of input_ids or inputs_embeds is required")
        resolved_owner = _resolve_owner(
            owner=owner,
            request_id=request_id,
            decoder_state=decoder_state,
        )
        if decoder_state is not None and decoded_media:
            raise ValueError("cached decode accepts no raw media")
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            if decoded_media:
                prefilled = self.prefill_pipeline(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoded_media=decoded_media,
                    text_embedding=self.get_input_embeddings(),
                )
                input_ids = None
                inputs_embeds = prefilled.assembled.inputs_embeds
                attention_mask = prefilled.assembled.attention_mask
                labels = prefilled.assembled.labels
                position_batch = prefilled.positions
        else:
            assert inputs_embeds is not None
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )
        assert attention_mask is not None
        attention_mask = attention_mask.bool()
        if position_batch is None:
            position_batch = _default_positions(attention_mask)
        position_batch.validate(attention_mask)
        return self._run_backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_batch=position_batch,
            labels=labels,
            decoder_state=decoder_state,
            owner=resolved_owner,
            use_cache=use_cache,
        )

    def prefill(
        self,
        *,
        inputs: ModelPrefillInputs,
        owner: StateOwner,
        use_cache: bool,
    ) -> CausalLMOutput:
        if not isinstance(inputs, ModelPrefillInputs):
            raise TypeError("inputs must be ModelPrefillInputs")
        positions = inputs.position_batch or _default_positions(inputs.key_valid_mask)
        raw = self._run_backbone(
            input_ids=inputs.input_ids,
            inputs_embeds=inputs.inputs_embeds,
            attention_mask=inputs.key_valid_mask,
            position_batch=positions,
            labels=None,
            decoder_state=None,
            owner=owner,
            use_cache=use_cache,
        )
        return _as_causal_output(raw)

    def decode(
        self,
        *,
        inputs: ModelDecodeInputs,
        owner: StateOwner,
    ) -> CausalLMOutput:
        if not isinstance(inputs, ModelDecodeInputs):
            raise TypeError("inputs must be ModelDecodeInputs")
        raw = self._run_backbone(
            input_ids=inputs.token_ids,
            inputs_embeds=None,
            attention_mask=inputs.current_key_valid_mask,
            position_batch=inputs.position_batch,
            labels=None,
            decoder_state=inputs.decoder_state,
            owner=owner,
            use_cache=True,
        )
        return _as_causal_output(raw)


__all__ = [
    "Qwen35InspiredThinker",
    "assert_disjoint_complete_groups",
    "unique_parameters_by_identity",
]
