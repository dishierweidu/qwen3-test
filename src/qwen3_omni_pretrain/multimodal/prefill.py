from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch
from torch import nn

from qwen3_omni_pretrain.multimodal.encoders import (
    AudioWindowEncoder,
    PatchVisionEncoder,
    TemporalVideoEncoder,
)
from qwen3_omni_pretrain.multimodal.io import DecodedMedia
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.positions import PositionBuilder
from qwen3_omni_pretrain.multimodal.sequence_assembler import (
    EmbeddingLookup,
    MediaExpansionPolicy,
    SequenceAssembler,
)
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    ResolvedMultimodalTokens,
)
from qwen3_omni_pretrain.multimodal.types import (
    AssembledSequence,
    MediaSequence,
    MediaSource,
    PositionBatch,
)


_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)

_MODALITY_ORDER = (
    MediaModality.IMAGE,
    MediaModality.VIDEO,
    MediaModality.AUDIO,
)


def _exact_non_negative_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_constructor_tokens(
    tokens: object,
    *,
    pad_token_id: object,
    joint_separator_token_ids: object,
    max_assembled_length: object,
) -> tuple[
    ResolvedMultimodalTokens,
    int,
    frozenset[int],
    int,
]:
    if type(tokens) is not ResolvedMultimodalTokens:
        raise TypeError(
            "tokens must have exact type ResolvedMultimodalTokens"
        )

    present_ids: list[int] = []
    for field in fields(ResolvedMultimodalTokens):
        token_id = getattr(tokens, field.name)
        if token_id is None:
            continue
        present_ids.append(
            _exact_non_negative_integer(token_id, field.name)
        )
    for sentinel_name in ("image_pad", "video_pad", "audio_pad"):
        if getattr(tokens, sentinel_name) is None:
            raise ValueError(f"{sentinel_name} must be present")
    if len(set(present_ids)) != len(present_ids):
        raise ValueError("all present multimodal token IDs must be distinct")

    validated_pad = _exact_non_negative_integer(
        pad_token_id,
        "pad_token_id",
    )
    sentinel_ids = {
        tokens.image_pad,
        tokens.video_pad,
        tokens.audio_pad,
    }
    if validated_pad in sentinel_ids:
        raise ValueError("pad_token_id must differ from media sentinel IDs")

    if type(max_assembled_length) is not int:
        raise TypeError("max_assembled_length must be an integer")
    if max_assembled_length <= 0:
        raise ValueError("max_assembled_length must be positive")

    if type(joint_separator_token_ids) is not frozenset:
        raise TypeError("joint_separator_token_ids must be a frozenset")
    for token_id in joint_separator_token_ids:
        _exact_non_negative_integer(token_id, "joint separator token ID")
        if token_id == validated_pad or token_id in sentinel_ids:
            raise ValueError(
                "joint separator token IDs must differ from pad and "
                "media sentinel IDs"
            )

    return (
        tokens,
        validated_pad,
        joint_separator_token_ids,
        max_assembled_length,
    )


def _validate_text_preflight(
    *,
    input_ids: object,
    attention_mask: object,
    labels: object,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")
    if labels is not None and not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor or None")

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
        raise ValueError("input_ids must have non-empty shape [B, T]")
    if input_ids.dtype is not torch.long:
        raise TypeError("input_ids must have dtype torch.long")
    if bool((input_ids < 0).any().item()):
        raise ValueError("input_ids must be non-negative")

    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    if (
        attention_mask.dtype is not torch.bool
        and attention_mask.dtype not in _INTEGER_DTYPES
    ):
        raise TypeError(
            "attention_mask must have a boolean or integer dtype"
        )
    if not bool(
        ((attention_mask == 0) | (attention_mask == 1)).all().item()
    ):
        raise ValueError("attention_mask values must be 0 or 1")
    if attention_mask.device != input_ids.device:
        raise ValueError(
            "input_ids and attention_mask must be on the same device"
        )

    if labels is not None:
        if labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids shape")
        if labels.dtype is not torch.long:
            raise TypeError("labels must have dtype torch.long")
        if labels.device != input_ids.device:
            raise ValueError(
                "input_ids, attention_mask, and labels must share a device"
            )
        if bool(((labels < 0) & (labels != -100)).any().item()):
            raise ValueError(
                "labels must contain only -100 or non-negative values"
            )

    mask = attention_mask.to(dtype=torch.bool)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    for sample_index in range(input_ids.shape[0]):
        valid_length = int(torch.count_nonzero(mask[sample_index]).item())
        if valid_length <= 0:
            raise ValueError(
                "every attention_mask row must contain a non-empty prefix"
            )
        expected = positions < valid_length
        if not torch.equal(mask[sample_index], expected):
            raise ValueError(
                "attention_mask rows must be right-padded prefixes"
            )

    masked = ~mask
    if bool((input_ids.masked_select(masked) != pad_token_id).any().item()):
        raise ValueError("masked input_ids must equal pad_token_id")
    if labels is not None and bool(
        (labels.masked_select(masked) != -100).any().item()
    ):
        raise ValueError("masked labels must equal -100")
    return input_ids, attention_mask, labels


def _snapshot_decoded_media(
    decoded_media: object,
    *,
    batch_size: int,
) -> tuple[DecodedMedia, ...]:
    if isinstance(decoded_media, (str, bytes)) or not isinstance(
        decoded_media,
        Sequence,
    ):
        raise TypeError("decoded_media must be a non-string Sequence")
    snapshot = tuple(decoded_media)
    if any(not isinstance(item, DecodedMedia) for item in snapshot):
        raise TypeError("decoded_media must contain DecodedMedia values")

    item_keys: set[tuple[int, int]] = set()
    source_keys: set[tuple[int, str]] = set()
    indices_by_sample: dict[int, list[int]] = defaultdict(list)
    for item in snapshot:
        item.__post_init__()
        item.request.__post_init__()
        request = item.request
        if request.sample_index >= batch_size:
            raise ValueError(
                "decoded media sample_index is outside the text batch"
            )
        item_key = (request.sample_index, request.item_index)
        if item_key in item_keys:
            raise ValueError(
                "decoded media (sample_index, item_index) keys must be unique"
            )
        item_keys.add(item_key)
        source_key = (request.sample_index, request.source_id)
        if source_key in source_keys:
            raise ValueError(
                "decoded media (sample_index, source_id) keys must be unique"
            )
        source_keys.add(source_key)
        indices_by_sample[request.sample_index].append(request.item_index)

    for sample_index, indices in indices_by_sample.items():
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(
                "decoded media item_index values must be dense from zero "
                f"within sample {sample_index}"
            )
    return tuple(
        sorted(
            snapshot,
            key=lambda item: (
                item.request.sample_index,
                item.request.item_index,
            ),
        )
    )


def _source_from_item(item: DecodedMedia) -> MediaSource:
    request = item.request
    return MediaSource(
        sample_index=request.sample_index,
        item_index=request.item_index,
        source_id=request.source_id,
    )


@dataclass(frozen=True)
class MultimodalPrefillOutput:
    assembled: AssembledSequence
    positions: PositionBatch
    media_sequences: tuple[MediaSequence, ...]


class MultimodalPrefillPipeline(nn.Module):
    def __init__(
        self,
        *,
        tokens: ResolvedMultimodalTokens,
        image_encoder: PatchVisionEncoder,
        video_encoder: TemporalVideoEncoder,
        audio_encoder: nn.Module,
        assembler: SequenceAssembler,
        expansion_policy: MediaExpansionPolicy,
        position_builder: PositionBuilder,
        pad_token_id: int,
        joint_separator_token_ids: frozenset[int],
        max_assembled_length: int,
    ) -> None:
        super().__init__()
        (
            validated_tokens,
            validated_pad,
            validated_separators,
            validated_limit,
        ) = _validate_constructor_tokens(
            tokens,
            pad_token_id=pad_token_id,
            joint_separator_token_ids=joint_separator_token_ids,
            max_assembled_length=max_assembled_length,
        )
        if not isinstance(image_encoder, PatchVisionEncoder):
            raise TypeError("image_encoder must be a PatchVisionEncoder")
        if not isinstance(video_encoder, TemporalVideoEncoder):
            raise TypeError("video_encoder must be a TemporalVideoEncoder")
        if not isinstance(audio_encoder, nn.Module) or not callable(
            getattr(audio_encoder, "forward", None)
        ):
            raise TypeError(
                "audio_encoder must be a registered media nn.Module"
            )
        audio_hidden_size = getattr(audio_encoder, "hidden_size", None)
        if type(audio_hidden_size) is not int or audio_hidden_size <= 0:
            raise TypeError(
                "audio_encoder must expose a positive integer hidden_size"
            )
        hidden_sizes = (
            image_encoder.hidden_size,
            video_encoder.hidden_size,
            audio_hidden_size,
        )
        if any(type(size) is not int or size <= 0 for size in hidden_sizes):
            raise TypeError("all media encoders must expose positive hidden_size")
        if len(set(hidden_sizes)) != 1:
            raise ValueError("all media encoder hidden sizes must match")
        if not isinstance(assembler, SequenceAssembler):
            raise TypeError("assembler must be a SequenceAssembler")
        if any(
            isinstance(component, nn.Module)
            for component in (assembler, expansion_policy, position_builder)
        ):
            raise TypeError(
                "assembler, expansion_policy, and position_builder must "
                "be non-nn.Module objects"
            )
        if expansion_policy is None or not callable(
            getattr(expansion_policy, "expand_sample", None)
        ):
            raise TypeError(
                "expansion_policy must define callable expand_sample"
            )
        if position_builder is None or not callable(
            getattr(position_builder, "build", None)
        ):
            raise TypeError("position_builder must define callable build")

        self.image_encoder = image_encoder
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.tokens = validated_tokens
        self.assembler = assembler
        self.expansion_policy = expansion_policy
        self.position_builder = position_builder
        self.pad_token_id = validated_pad
        self.joint_separator_token_ids = validated_separators
        self.max_assembled_length = validated_limit
        self.hidden_size = hidden_sizes[0]

    def _encoder_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        floating_parameters = tuple(
            parameter
            for encoder in (
                self.image_encoder,
                self.video_encoder,
                self.audio_encoder,
            )
            for parameter in encoder.parameters()
            if parameter.is_floating_point()
        )
        if not floating_parameters:
            raise ValueError(
                "media encoders must expose floating-point parameters"
            )
        placements = {
            (parameter.device, parameter.dtype)
            for parameter in floating_parameters
        }
        if len(placements) != 1:
            raise ValueError(
                "all media encoder parameters must share one device and dtype"
            )
        return next(iter(placements))

    def _validate_text_embeddings(
        self,
        result: object,
        *,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(result, torch.Tensor):
            raise TypeError("text_embedding must return a torch.Tensor")
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            self.hidden_size,
        )
        if result.shape != expected_shape:
            raise ValueError(
                "text_embedding output must have exact shape [B, T, H]"
            )
        if not result.is_floating_point():
            raise TypeError(
                "text_embedding output must have a floating non-complex dtype"
            )
        if result.device != input_ids.device:
            raise ValueError(
                "text_embedding output must be on input_ids.device"
            )
        encoder_device, encoder_dtype = self._encoder_device_and_dtype()
        if result.device != encoder_device or result.dtype != encoder_dtype:
            raise ValueError(
                "text embeddings and all media encoders must share one "
                "device and dtype"
            )
        return result

    @staticmethod
    def _validate_media_sequence(
        result: object,
        *,
        modality: MediaModality,
        items: tuple[DecodedMedia, ...],
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> MediaSequence:
        if not isinstance(result, MediaSequence):
            raise TypeError("media encoder must return a MediaSequence")
        result.validate()
        if result.modality is not modality:
            raise ValueError("media encoder returned the wrong modality")
        expected_sources = tuple(_source_from_item(item) for item in items)
        if result.sources != expected_sources:
            raise ValueError(
                "media encoder sources must match canonical requests"
            )
        if result.embeddings.shape[2] != hidden_size:
            raise ValueError(
                "media embeddings must match the common hidden size"
            )
        if result.embeddings.dtype != dtype:
            raise ValueError("media embeddings must match text dtype")
        if result.embeddings.device != device:
            raise ValueError("media embeddings must match text device")
        return result

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor | None,
        decoded_media: Sequence[DecodedMedia],
        text_embedding: EmbeddingLookup,
    ) -> MultimodalPrefillOutput:
        input_ids, attention_mask, labels = _validate_text_preflight(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pad_token_id=self.pad_token_id,
        )
        canonical_items = _snapshot_decoded_media(
            decoded_media,
            batch_size=input_ids.shape[0],
        )
        partitions = {
            modality: tuple(
                item
                for item in canonical_items
                if item.request.modality is modality
            )
            for modality in _MODALITY_ORDER
        }

        if not callable(text_embedding):
            raise TypeError("text_embedding must be callable")
        text_embeddings = self._validate_text_embeddings(
            text_embedding(input_ids),
            input_ids=input_ids,
        )

        sequences: list[MediaSequence] = []
        for modality in _MODALITY_ORDER:
            items = partitions[modality]
            if not items:
                continue
            if modality is MediaModality.IMAGE:
                encoded = self.image_encoder(items)
            elif modality is MediaModality.VIDEO:
                encoded = self.video_encoder(
                    items,
                    patch_encoder=self.image_encoder,
                )
            else:
                encoded = self.audio_encoder(items)
            sequences.append(
                self._validate_media_sequence(
                    encoded,
                    modality=modality,
                    items=items,
                    hidden_size=self.hidden_size,
                    dtype=text_embeddings.dtype,
                    device=text_embeddings.device,
                )
            )
        media_sequences = tuple(sequences)

        assembled = self.assembler.assemble(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            media_sequences=media_sequences,
            tokens=self.tokens,
            expansion_policy=self.expansion_policy,
            embedding_lookup=text_embedding,
            pad_token_id=self.pad_token_id,
            joint_separator_token_ids=self.joint_separator_token_ids,
            max_assembled_length=self.max_assembled_length,
        )
        if not isinstance(assembled, AssembledSequence):
            raise TypeError("assembler must return an AssembledSequence")
        assembled.validate()

        positions = self.position_builder.build(assembled)
        if not isinstance(positions, PositionBatch):
            raise TypeError("position_builder must return a PositionBatch")
        positions.validate(assembled.attention_mask)
        return MultimodalPrefillOutput(
            assembled=assembled,
            positions=positions,
            media_sequences=media_sequences,
        )

    def encode_and_assemble(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor | None,
        decoded_media: Sequence[DecodedMedia],
        text_embedding: EmbeddingLookup,
    ) -> MultimodalPrefillOutput:
        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoded_media=decoded_media,
            text_embedding=text_embedding,
        )


__all__ = [
    "MultimodalPrefillOutput",
    "MultimodalPrefillPipeline",
]
