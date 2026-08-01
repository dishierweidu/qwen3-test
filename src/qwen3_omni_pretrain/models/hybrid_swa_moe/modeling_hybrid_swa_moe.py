"""Complete text-only Hybrid SWA routed-MoE causal language model."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import nn
from torch.nn import functional as F

from qwen3_omni_pretrain.models.hybrid_swa_moe.attention import (
    HybridSelfAttention,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.configuration_hybrid_swa_moe import (
    HybridSwaMoeConfig,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.moe import (
    RoutedSwiGLUMoE,
    RouterStats,
    SwiGLU,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.mtp import MultiTokenPredictor
from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.parallel.expert_parallel import ExpertParallelContext
from qwen3_omni_pretrain.runtime.capabilities import CacheSupport
from qwen3_omni_pretrain.runtime.protocols import (
    CausalLMOutput,
    ModelDecodeInputs,
    ModelPrefillInputs,
)
from qwen3_omni_pretrain.runtime.state import (
    AttentionKV,
    DecoderPositionState,
    DecoderState,
    LegacyPositionCursor,
    SlidingWindowKV,
    StateOwner,
)


def _bool_mask(
    value: torch.Tensor | None,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError("attention_mask must have shape [B, Q]")
    if value.device != device:
        raise ValueError("attention_mask and model inputs must share a device")
    if value.dtype is torch.bool:
        return value
    if not bool(((value == 0) | (value == 1)).all().item()):
        raise ValueError("attention_mask values must be 0 or 1")
    return value.to(torch.bool)


def _position_batch(position_ids: torch.Tensor) -> PositionBatch:
    return PositionBatch(
        position_ids=position_ids.unsqueeze(0),
        rope_deltas=torch.zeros(
            (position_ids.shape[0], 1),
            dtype=position_ids.dtype,
            device=position_ids.device,
        ),
        axis_names=("sequence",),
    )


class HybridDecoderLayer(nn.Module):
    def __init__(
        self,
        config: HybridSwaMoeConfig,
        layer_index: int,
        *,
        expert_parallel_context: ExpertParallelContext | None = None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attention_type = config.attention_layer_types[layer_index]
        self.ffn_type = config.ffn_layer_types[layer_index]
        self.input_norm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = HybridSelfAttention(
            config, layer_index=layer_index
        )
        self.post_attention_norm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.ffn_type == "dense":
            self.ffn: SwiGLU | RoutedSwiGLUMoE = SwiGLU(
                config.hidden_size, config.dense_intermediate_size
            )
        else:
            self.ffn = RoutedSwiGLUMoE(
                config.hidden_size,
                config.expert_intermediate_size,
                config.num_experts,
                config.num_experts_per_token,
                expert_parallel_context=expert_parallel_context,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        current_key_valid_mask: torch.Tensor,
        cache: AttentionKV | SlidingWindowKV | None,
        use_cache: bool,
        collect_router_stats: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        RouterStats | None,
        AttentionKV | SlidingWindowKV | None,
    ]:
        attention_output, next_cache = self.self_attn(
            self.input_norm(hidden_states),
            position_ids=position_ids,
            current_key_valid_mask=current_key_valid_mask,
            cache=cache,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        ffn_input = self.post_attention_norm(hidden_states)
        if isinstance(self.ffn, RoutedSwiGLUMoE):
            routed = self.ffn(
                ffn_input,
                token_mask=current_key_valid_mask,
                collect_stats=collect_router_stats,
            )
            ffn_output = routed.hidden_states
            aux_loss = routed.aux_loss
            stats = routed.stats
        else:
            ffn_output = self.ffn(ffn_input)
            aux_loss = None
            stats = None
        hidden_states = hidden_states + ffn_output
        hidden_states = hidden_states * current_key_valid_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )
        return hidden_states, aux_loss, stats, next_cache


class HybridSwaMoeForCausalLM(nn.Module):
    """Small, fully trainable experiment model with typed cache methods."""

    config_class = HybridSwaMoeConfig

    def __init__(
        self,
        config: HybridSwaMoeConfig,
        *,
        expert_parallel_context: ExpertParallelContext | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, HybridSwaMoeConfig):
            raise TypeError("config must be HybridSwaMoeConfig")
        self.config = config
        if expert_parallel_context is not None and not isinstance(
            expert_parallel_context, ExpertParallelContext
        ):
            raise TypeError(
                "expert_parallel_context must be ExpertParallelContext"
            )
        self.expert_parallel_context = expert_parallel_context
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            HybridDecoderLayer(
                config,
                layer_index,
                expert_parallel_context=expert_parallel_context,
            )
            for layer_index in range(config.num_hidden_layers)
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.mtp = (
            MultiTokenPredictor(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                num_predictors=config.mtp_num_predictors,
                rms_norm_eps=config.rms_norm_eps,
            )
            if config.mtp_num_predictors > 0
            else None
        )
        self.cache_support = CacheSupport(
            incremental_decode_state=True,
            streaming_generation=True,
            beam_search=False,
            state_truncate=False,
            speculative_decode=True,
        )
        self.architecture_capabilities: Mapping[str, bool] = MappingProxyType(
            {
                "attention_sink": config.attention_sink,
                "expert_parallel": expert_parallel_context is not None,
                "full_attention": "full" in config.attention_layer_types,
                "mtp_enabled": self.mtp is not None,
                "routed_moe": "routed_moe" in config.ffn_layer_types,
                "sliding_window_attention": "swa" in config.attention_layer_types,
            }
        )
        self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    @property
    def full_attention_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, layer_type in enumerate(
                self.config.attention_layer_types
            )
            if layer_type == "full"
        )

    @property
    def swa_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, layer_type in enumerate(
                self.config.attention_layer_types
            )
            if layer_type == "swa"
        )

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype

    def _validate_state(self, state: DecoderState) -> None:
        device, dtype = self._device_dtype()
        if state.device != device:
            raise ValueError("decoder state and model must share a device")
        if tuple(state.full_attention_kv) != self.full_attention_layer_indices:
            raise ValueError("full-attention cache layer set does not match model")
        if tuple(state.swa_kv) != self.swa_layer_indices:
            raise ValueError("SWA cache layer set does not match model")
        if state.position is None:
            raise ValueError("Hybrid decode state requires position history")
        if not isinstance(state.position.continuation, LegacyPositionCursor):
            raise ValueError("Hybrid decode state requires a legacy text cursor")
        for index, cache in (
            tuple(state.full_attention_kv.items())
            + tuple(state.swa_kv.items())
        ):
            attention = self.layers[index].self_attn
            if (
                cache.key.dtype != dtype
                or cache.key.shape[1] != attention.num_key_value_heads
                or cache.key.shape[-1] != attention.qk_head_dim
                or cache.value.shape[-1] != attention.v_head_dim
            ):
                raise ValueError(f"cache topology mismatch at layer {index}")

    def _resolve_positions(
        self,
        *,
        position_ids: torch.Tensor | None,
        current_mask: torch.Tensor,
        decoder_state: DecoderState | None,
    ) -> torch.Tensor:
        batch_size, query_length = current_mask.shape
        if position_ids is not None:
            if (
                not isinstance(position_ids, torch.Tensor)
                or position_ids.dtype is not torch.long
                or position_ids.shape != (batch_size, query_length)
                or position_ids.device != current_mask.device
            ):
                raise ValueError(
                    "position_ids must have dtype long and shape [B, Q]"
                )
            resolved = position_ids
        else:
            relative = current_mask.to(torch.long).cumsum(dim=1) - 1
            if decoder_state is None:
                base = torch.zeros(
                    (batch_size, 1),
                    dtype=torch.long,
                    device=current_mask.device,
                )
            else:
                assert decoder_state.position is not None
                continuation = decoder_state.position.continuation
                assert isinstance(continuation, LegacyPositionCursor)
                base = continuation.next_storage_position.unsqueeze(1)
            resolved = (relative + base).masked_fill(~current_mask, 0)
        valid = resolved.masked_select(current_mask)
        if valid.numel() and (
            bool((valid < 0).any().item())
            or int(valid.max().item()) >= self.config.max_position_embeddings
        ):
            raise ValueError("valid position IDs exceed the configured context")
        return resolved

    @staticmethod
    def _next_cursor(
        *,
        position_ids: torch.Tensor,
        current_mask: torch.Tensor,
        previous: LegacyPositionCursor | None,
    ) -> LegacyPositionCursor:
        if previous is None:
            next_positions = torch.zeros(
                current_mask.shape[0],
                dtype=torch.long,
                device=current_mask.device,
            )
        else:
            next_positions = previous.next_storage_position.clone()
        for row in range(current_mask.shape[0]):
            valid = position_ids[row].masked_select(current_mask[row])
            if valid.numel():
                next_positions[row] = torch.maximum(
                    next_positions[row], valid.max() + 1
                )
        return LegacyPositionCursor(next_positions)

    def _make_state(
        self,
        *,
        owner: StateOwner,
        old_state: DecoderState | None,
        current_mask: torch.Tensor,
        current_positions: PositionBatch,
        position_ids: torch.Tensor,
        full_caches: dict[int, AttentionKV],
        swa_caches: dict[int, SlidingWindowKV],
    ) -> DecoderState:
        previous_cursor = (
            None
            if old_state is None or old_state.position is None
            else old_state.position.continuation
        )
        if previous_cursor is not None and not isinstance(
            previous_cursor, LegacyPositionCursor
        ):
            raise ValueError("Hybrid state cursor type changed")
        continuation = self._next_cursor(
            position_ids=position_ids,
            current_mask=current_mask,
            previous=previous_cursor,
        )
        if old_state is None:
            position = DecoderPositionState(
                cached=current_positions,
                key_valid_mask=current_mask,
                continuation=continuation,
            )
            seen_tokens = current_mask.sum(dim=1).to(torch.long)
        else:
            assert old_state.position is not None
            position = old_state.position.append(
                current=current_positions,
                current_key_valid_mask=current_mask,
                continuation=continuation,
            )
            seen_tokens = old_state.seen_tokens + current_mask.sum(dim=1).to(
                torch.long
            )
        return DecoderState(
            owner=owner,
            seen_tokens=seen_tokens,
            position=position,
            full_attention_kv=full_caches,
            swa_kv=swa_caches,
            processed_media=None if old_state is None else old_state.processed_media,
            gdn_state=None if old_state is None else old_state.gdn_state,
            talker_state=None if old_state is None else old_state.talker_state,
            mtp_state=None if old_state is None else old_state.mtp_state,
            codec_state=None if old_state is None else old_state.codec_state,
        )

    def _forward_impl(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        labels: torch.Tensor | None,
        decoder_state: DecoderState | None,
        owner: StateOwner | None,
        use_cache: bool,
        collect_router_stats: bool,
        current_position_batch: PositionBatch | None,
    ) -> dict[str, object]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one of input_ids or inputs_embeds is required")
        device, dtype = self._device_dtype()
        if input_ids is not None:
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.dtype is not torch.long
                or input_ids.ndim != 2
            ):
                raise ValueError("input_ids must have dtype long and shape [B, Q]")
            if input_ids.device != device:
                raise ValueError("input_ids and model must share a device")
            if bool(((input_ids < 0) | (input_ids >= self.vocab_size)).any().item()):
                raise ValueError("input_ids are outside the model vocabulary")
            hidden_states = self.embed_tokens(input_ids)
        else:
            assert inputs_embeds is not None
            if (
                inputs_embeds.ndim != 3
                or inputs_embeds.shape[-1] != self.config.hidden_size
                or inputs_embeds.device != device
                or inputs_embeds.dtype != dtype
            ):
                raise ValueError("inputs_embeds shape/dtype/device does not match model")
            hidden_states = inputs_embeds
        batch_size, query_length = hidden_states.shape[:2]
        if query_length == 0:
            raise ValueError("query length must be positive")
        current_mask = _bool_mask(
            attention_mask,
            shape=(batch_size, query_length),
            device=device,
        )
        if bool((~current_mask.any(dim=1)).any().item()):
            raise ValueError("every batch row must contain a valid token")
        if type(use_cache) is not bool or type(collect_router_stats) is not bool:
            raise TypeError("cache and router-stat flags must be boolean")
        if decoder_state is not None:
            if not use_cache:
                raise ValueError("decoder_state requires use_cache=True")
            self._validate_state(decoder_state)
            if owner is None:
                owner = decoder_state.owner
            decoder_state.assert_owner(owner)
        elif use_cache and owner is None:
            raise ValueError("cache creation requires a StateOwner")
        resolved_positions = self._resolve_positions(
            position_ids=position_ids,
            current_mask=current_mask,
            decoder_state=decoder_state,
        )
        if current_position_batch is None:
            current_position_batch = _position_batch(resolved_positions)
        else:
            current_position_batch.validate(current_mask)
            if current_position_batch.axis_names != ("sequence",):
                raise ValueError("Hybrid text positions require one sequence axis")
            if not torch.equal(
                current_position_batch.position_ids[0].to(torch.long),
                resolved_positions,
            ):
                raise ValueError("position_batch and position_ids disagree")

        if labels is not None:
            if (
                not isinstance(labels, torch.Tensor)
                or labels.dtype is not torch.long
                or labels.shape != (batch_size, query_length)
                or labels.device != device
            ):
                raise ValueError("labels must have dtype long and shape [B, Q]")
            labels = labels.masked_fill(~current_mask, -100)

        aux_losses: list[torch.Tensor] = []
        router_stats: list[RouterStats] = []
        next_full: dict[int, AttentionKV] = {}
        next_swa: dict[int, SlidingWindowKV] = {}
        for index, layer in enumerate(self.layers):
            cache = None
            if decoder_state is not None:
                cache = (
                    decoder_state.full_attention_kv.get(index)
                    if layer.attention_type == "full"
                    else decoder_state.swa_kv.get(index)
                )
            hidden_states, aux_loss, stats, next_cache = layer(
                hidden_states,
                position_ids=resolved_positions,
                current_key_valid_mask=current_mask,
                cache=cache,
                use_cache=use_cache,
                collect_router_stats=collect_router_stats,
            )
            if aux_loss is not None:
                aux_losses.append(aux_loss)
            if stats is not None:
                router_stats.append(stats)
            if use_cache:
                if layer.attention_type == "full":
                    if not isinstance(next_cache, AttentionKV):
                        raise RuntimeError("full layer did not return AttentionKV")
                    next_full[index] = next_cache
                else:
                    if not isinstance(next_cache, SlidingWindowKV):
                        raise RuntimeError("SWA layer did not return SlidingWindowKV")
                    next_swa[index] = next_cache

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None

        ce_loss: torch.Tensor | None = None
        if labels is not None and query_length > 1:
            targets = labels[:, 1:]
            valid_targets = int((targets != -100).sum().item())
            if valid_targets:
                ce_loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, self.vocab_size).float(),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ) / valid_targets

        mtp_loss: torch.Tensor | None = None
        if labels is not None and self.mtp is not None:
            mtp_loss = self.mtp(hidden_states, labels=labels).loss

        loss = ce_loss
        if loss is not None:
            if self.config.router_aux_loss_weight != 0.0 and aux_loss is not None:
                loss = loss + self.config.router_aux_loss_weight * aux_loss
            if self.config.mtp_loss_weight != 0.0 and mtp_loss is not None:
                loss = loss + self.config.mtp_loss_weight * mtp_loss

        next_state = None
        if use_cache:
            assert owner is not None
            next_state = self._make_state(
                owner=owner,
                old_state=decoder_state,
                current_mask=current_mask,
                current_positions=current_position_batch,
                position_ids=resolved_positions,
                full_caches=next_full,
                swa_caches=next_swa,
            )
        return {
            "logits": logits,
            "loss": loss,
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
            "mtp_loss": mtp_loss,
            "router_stats": tuple(router_stats) if collect_router_stats else None,
            "decoder_state": next_state,
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        *,
        decoder_state: DecoderState | None = None,
        request_id: str | None = None,
        use_cache: bool = False,
        collect_router_stats: bool = False,
        expert_parallel_context: ExpertParallelContext | None = None,
    ) -> dict[str, object]:
        if (
            expert_parallel_context is not None
            and expert_parallel_context is not self.expert_parallel_context
        ):
            raise ValueError(
                "forward expert_parallel_context must be the construction context"
            )
        owner: StateOwner | None = None
        if decoder_state is not None:
            owner = decoder_state.owner
            if request_id is not None and request_id != owner.display_request_id:
                raise ValueError("request_id does not match decoder state owner")
        elif use_cache:
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("use_cache requires a non-empty request_id")
            owner = StateOwner.fresh(request_id)
        return self._forward_impl(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            decoder_state=decoder_state,
            owner=owner,
            use_cache=use_cache,
            collect_router_stats=collect_router_stats,
            current_position_batch=None,
        )

    @staticmethod
    def _to_causal_output(raw: Mapping[str, object]) -> CausalLMOutput:
        return CausalLMOutput(
            logits=raw["logits"],  # type: ignore[arg-type]
            loss=raw["loss"],  # type: ignore[arg-type]
            ce_loss=raw["ce_loss"],  # type: ignore[arg-type]
            aux_loss=raw["aux_loss"],  # type: ignore[arg-type]
            decoder_state=raw["decoder_state"],  # type: ignore[arg-type]
            hidden_states=None,
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
        if not isinstance(owner, StateOwner):
            raise TypeError("owner must be StateOwner")
        if inputs.media is not None:
            raise ValueError("Hybrid text model accepts assembled text/embeddings only")
        if use_cache and inputs.position_batch is None:
            raise ValueError("cached prefill requires an explicit PositionBatch")
        position_ids = (
            None
            if inputs.position_batch is None
            else inputs.position_batch.position_ids[0].to(torch.long)
        )
        with torch.inference_mode():
            raw = self._forward_impl(
                input_ids=inputs.input_ids,
                inputs_embeds=inputs.inputs_embeds,
                attention_mask=inputs.key_valid_mask,
                position_ids=position_ids,
                labels=None,
                decoder_state=None,
                owner=owner,
                use_cache=use_cache,
                collect_router_stats=False,
                current_position_batch=inputs.position_batch,
            )
        return self._to_causal_output(raw)

    def decode(
        self,
        *,
        inputs: ModelDecodeInputs,
        owner: StateOwner,
    ) -> CausalLMOutput:
        if not isinstance(inputs, ModelDecodeInputs):
            raise TypeError("inputs must be ModelDecodeInputs")
        if not isinstance(owner, StateOwner):
            raise TypeError("owner must be StateOwner")
        inputs.decoder_state.assert_owner(owner)
        with torch.inference_mode():
            raw = self._forward_impl(
                input_ids=inputs.token_ids,
                inputs_embeds=None,
                attention_mask=inputs.current_key_valid_mask,
                position_ids=inputs.position_batch.position_ids[0].to(torch.long),
                labels=None,
                decoder_state=inputs.decoder_state,
                owner=owner,
                use_cache=True,
                collect_router_stats=False,
                current_position_batch=inputs.position_batch,
            )
        return self._to_causal_output(raw)


__all__ = ["HybridDecoderLayer", "HybridSwaMoeForCausalLM"]
