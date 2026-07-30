from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from qwen3_omni_pretrain.data.collators import MediaLoadError
from qwen3_omni_pretrain.multimodal.io import (
    DecodedMedia,
    MediaRequest,
    StrictMediaLoader,
)
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    MultimodalTokenSchema,
    resolve_token_schema,
)


class AllSamplesQuarantinedError(RuntimeError):
    """Cause attached to the aggregate error for an empty retained batch."""

    def __init__(self, quarantined_count: int) -> None:
        self.quarantined_count = int(quarantined_count)
        super().__init__(
            f"all {self.quarantined_count} samples were quarantined"
        )


@dataclass(frozen=True)
class _MediaItem:
    source_id: str
    modality: MediaModality
    path: str
    timeline_offset_seconds: float
    timestamps: tuple[float, ...] | None

    def request(
        self,
        *,
        sample_id: str,
        sample_index: int,
        original_sample_index: int,
        item_index: int,
    ) -> MediaRequest:
        return MediaRequest(
            sample_id=sample_id,
            sample_index=sample_index,
            item_index=item_index,
            source_id=self.source_id,
            modality=self.modality,
            path=self.path,
            original_sample_index=original_sample_index,
            timeline_offset_seconds=self.timeline_offset_seconds,
            timestamps=self.timestamps,
        )


@dataclass(frozen=True)
class _StagedRow:
    sample_id: str
    original_sample_index: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    media_items: tuple[_MediaItem, ...]


class ProfileStage2Collator:
    """Build strict, variable-length multimodal batches for profile runtimes."""

    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int = 2048,
        *,
        token_schema: MultimodalTokenSchema | None = None,
        media_loader: StrictMediaLoader | None = None,
        quarantine_bad_samples: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                "Tokenizer must define pad_token_id or eos_token_id"
            )
        self.pad_token_id = int(pad_id)

        self.token_schema = token_schema or MultimodalTokenSchema.qwen3()
        vocab = tokenizer.get_vocab()
        if not isinstance(vocab, Mapping) or not vocab:
            raise ValueError("tokenizer.get_vocab() must return a non-empty map")
        try:
            vocab_size = max(int(token_id) for token_id in vocab.values()) + 1
        except (TypeError, ValueError) as cause:
            raise ValueError("tokenizer vocabulary IDs must be integers") from cause
        self.resolved_tokens = resolve_token_schema(
            tokenizer,
            self.token_schema,
            vocab_size,
        )
        self.media_loader = (
            StrictMediaLoader() if media_loader is None else media_loader
        )
        if not isinstance(quarantine_bad_samples, bool):
            raise TypeError("quarantine_bad_samples must be a boolean")
        self.quarantine_bad_samples = quarantine_bad_samples

    def _encode_text(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        if torch.is_tensor(ids):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token_id) for token_id in ids]

    def _sentinel_modalities(
        self,
        token_ids: Sequence[int],
    ) -> tuple[MediaModality, ...]:
        by_id = {
            self.resolved_tokens.image_pad: MediaModality.IMAGE,
            self.resolved_tokens.video_pad: MediaModality.VIDEO,
            self.resolved_tokens.audio_pad: MediaModality.AUDIO,
        }
        return tuple(
            by_id[token_id] for token_id in token_ids if token_id in by_id
        )

    @staticmethod
    def _row_error(
        *,
        sample_id: str,
        cause: BaseException,
        modality: str = "sequence",
        path: str = "<multiple>",
    ) -> MediaLoadError:
        return MediaLoadError(
            modality=modality,
            path=path,
            sample_id=sample_id,
            cause=cause,
        )

    @staticmethod
    def _parse_media_items(
        example: Mapping[str, Any],
    ) -> tuple[_MediaItem, ...]:
        raw_items = example.get("media")
        if raw_items is None:
            return ()
        if (
            not isinstance(raw_items, Sequence)
            or isinstance(raw_items, (str, bytes, bytearray))
        ):
            raise TypeError("media must be a sequence of item mappings")

        parsed: list[_MediaItem] = []
        for item_index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, Mapping):
                raise TypeError(f"media[{item_index}] must be a mapping")

            source_id = raw_item.get("id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(
                    f"media[{item_index}].id must be a non-empty string"
                )
            path = raw_item.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(
                    f"media[{item_index}].path must be a non-empty string"
                )

            raw_modality = raw_item.get("modality")
            try:
                modality = (
                    raw_modality
                    if isinstance(raw_modality, MediaModality)
                    else MediaModality(raw_modality)
                )
            except (TypeError, ValueError) as cause:
                raise ValueError(
                    f"media[{item_index}].modality is unsupported: "
                    f"{raw_modality!r}"
                ) from cause

            raw_timestamps = raw_item.get("timestamps")
            timestamps: tuple[float, ...] | None
            if raw_timestamps is None:
                timestamps = None
            elif (
                isinstance(raw_timestamps, Sequence)
                and not isinstance(
                    raw_timestamps,
                    (str, bytes, bytearray),
                )
            ):
                timestamps = tuple(raw_timestamps)
            else:
                raise TypeError(
                    f"media[{item_index}].timestamps must be a sequence or None"
                )

            parsed.append(
                _MediaItem(
                    source_id=source_id,
                    modality=modality,
                    path=path,
                    timeline_offset_seconds=raw_item.get(
                        "timeline_offset_seconds",
                        0.0,
                    ),
                    timestamps=timestamps,
                )
            )
        return tuple(parsed)

    def _validate_sentinel_order(
        self,
        *,
        sample_id: str,
        token_ids: Sequence[int],
        media_items: Sequence[_MediaItem],
        phase: str,
    ) -> None:
        sentinels = self._sentinel_modalities(token_ids)
        items = tuple(item.modality for item in media_items)
        if sentinels != items:
            raise self._row_error(
                sample_id=sample_id,
                cause=ValueError(
                    f"{phase} media sentinel/item order mismatch: "
                    f"sentinels={[value.value for value in sentinels]!r}, "
                    f"items={[value.value for value in items]!r}"
                ),
            )

    def _stage_row(
        self,
        example: Mapping[str, Any],
        original_sample_index: int,
    ) -> _StagedRow:
        sample_id = str(
            example.get("id", f"batch-index-{original_sample_index}")
        )
        try:
            media_items = self._parse_media_items(example)
            prompt_ids = self._encode_text(
                example.get("input_text", "") or ""
            )
            target_ids = self._encode_text(
                example.get("target_text", "") or ""
            )
            self._validate_sentinel_order(
                sample_id=sample_id,
                token_ids=prompt_ids + target_ids,
                media_items=media_items,
                phase="untruncated",
            )

            kept_target = target_ids[: self.max_seq_length]
            prompt_budget = self.max_seq_length - len(kept_target)
            kept_prompt = (
                prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            )
            final_ids = kept_prompt + kept_target
            self._validate_sentinel_order(
                sample_id=sample_id,
                token_ids=final_ids,
                media_items=media_items,
                phase="final truncated",
            )
            labels = ([-100] * len(kept_prompt)) + kept_target
        except MediaLoadError:
            raise
        except Exception as cause:
            raise self._row_error(
                sample_id=sample_id,
                cause=cause,
                modality="media",
                path="<unknown>",
            ) from cause

        return _StagedRow(
            sample_id=sample_id,
            original_sample_index=original_sample_index,
            input_ids=tuple(final_ids),
            labels=tuple(labels),
            media_items=media_items,
        )

    def _decode_row(
        self,
        row: _StagedRow,
    ) -> tuple[DecodedMedia, ...]:
        decoded_items: list[DecodedMedia] = []
        for item_index, item in enumerate(row.media_items):
            try:
                request = item.request(
                    sample_id=row.sample_id,
                    sample_index=row.original_sample_index,
                    original_sample_index=row.original_sample_index,
                    item_index=item_index,
                )
                decoded = self.media_loader.load(request)
                if not isinstance(decoded, DecodedMedia):
                    raise TypeError(
                        "media loader must return DecodedMedia"
                    )
                decoded_items.append(
                    replace(decoded, request=request)
                )
            except MediaLoadError:
                raise
            except Exception as cause:
                raise self._row_error(
                    sample_id=row.sample_id,
                    cause=cause,
                    modality=item.modality.value,
                    path=item.path,
                ) from cause
        return tuple(decoded_items)

    def __call__(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not batch:
            raise ValueError("batch must contain at least one sample")

        retained_rows: list[_StagedRow] = []
        provisional_media: list[tuple[DecodedMedia, ...]] = []
        media_errors: list[dict[str, str]] = []

        for original_index, example in enumerate(batch):
            sample_id = (
                str(example.get("id", f"batch-index-{original_index}"))
                if isinstance(example, Mapping)
                else f"batch-index-{original_index}"
            )
            try:
                if not isinstance(example, Mapping):
                    raise self._row_error(
                        sample_id=sample_id,
                        cause=TypeError("each batch sample must be a mapping"),
                        modality="batch",
                        path="<unknown>",
                    )
                row = self._stage_row(example, original_index)
                row_media = self._decode_row(row)
            except MediaLoadError as error:
                if not self.quarantine_bad_samples:
                    raise
                media_errors.append(error.to_dict())
                continue

            retained_rows.append(row)
            provisional_media.append(row_media)

        if not retained_rows:
            cause = AllSamplesQuarantinedError(len(media_errors))
            raise MediaLoadError(
                modality="batch",
                path="<multiple>",
                sample_id="<all-quarantined>",
                cause=cause,
            ) from cause

        decoded_media: list[DecodedMedia] = []
        for sample_index, row_media in enumerate(provisional_media):
            for decoded in row_media:
                final_request = replace(
                    decoded.request,
                    sample_index=sample_index,
                )
                decoded_media.append(
                    replace(decoded, request=final_request)
                )

        width = min(
            self.max_seq_length,
            max(len(row.input_ids) for row in retained_rows),
        )
        input_ids = torch.full(
            (len(retained_rows), width),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(retained_rows), width),
            dtype=torch.long,
        )
        labels = torch.full(
            (len(retained_rows), width),
            -100,
            dtype=torch.long,
        )
        for index, row in enumerate(retained_rows):
            length = len(row.input_ids)
            if length == 0:
                continue
            input_ids[index, :length] = torch.tensor(
                row.input_ids,
                dtype=torch.long,
            )
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(
                row.labels,
                dtype=torch.long,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoded_media": tuple(decoded_media),
            "_sample_ids": [row.sample_id for row in retained_rows],
            "_media_errors": media_errors,
        }


__all__ = [
    "AllSamplesQuarantinedError",
    "ProfileStage2Collator",
]
