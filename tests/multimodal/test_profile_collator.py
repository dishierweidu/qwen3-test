from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from qwen3_omni_pretrain.data.collators import MediaLoadError
from qwen3_omni_pretrain.data.profile_collator import (
    AllSamplesQuarantinedError,
    ProfileStage2Collator,
)
from qwen3_omni_pretrain.multimodal.io import (
    DecodedMedia,
    MediaRequest,
)
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    MultimodalTokenSchema,
)


SPECIAL_IDS = {
    "<|image_pad|>": 101,
    "<|video_pad|>": 102,
    "<|audio_pad|>": 103,
    "<|vision_start|>": 104,
    "<|vision_end|>": 105,
    "<|audio_start|>": 106,
    "<|audio_end|>": 107,
}


class SentinelTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def get_vocab(self):
        return dict(SPECIAL_IDS)

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        ids = []
        position = 0
        while position < len(text):
            token = next(
                (
                    candidate
                    for candidate in SPECIAL_IDS
                    if text.startswith(candidate, position)
                ),
                None,
            )
            if token is not None:
                ids.append(SPECIAL_IDS[token])
                position += len(token)
            else:
                ids.append(2 + (ord(text[position]) % 90))
                position += 1
        return {"input_ids": ids}


class ControlledLoader:
    def __init__(self, *, bad_paths=()):
        self.bad_paths = set(bad_paths)
        self.seen = []

    def load(self, request):
        self.seen.append(request)
        if request.path in self.bad_paths:
            cause = ValueError(f"corrupt fixture: {request.path}")
            raise MediaLoadError(
                modality=request.modality.value,
                path=request.path,
                sample_id=request.sample_id,
                cause=cause,
            ) from cause
        if request.modality is MediaModality.IMAGE:
            tensor = torch.zeros(3, 2, 3)
            length = 1
            timestamps = None
            metadata = {
                "original_width": 3,
                "original_height": 2,
            }
        elif request.modality is MediaModality.AUDIO:
            length = 12_345 if "long" in request.path else 7
            tensor = torch.zeros(1, length)
            timestamps = None
            metadata = {"sample_rate": 16_000}
        else:
            tensor = torch.zeros(2, 3, 2, 3)
            length = 2
            timestamps = torch.tensor([0.0, 0.5])
            metadata = {"width": 3, "height": 2}
        return DecodedMedia(
            request=request,
            tensor=tensor,
            length=length,
            timestamps=timestamps,
            seconds_per_grid=(
                0.5
                if request.modality is MediaModality.VIDEO
                else None
            ),
            metadata=metadata,
        )


class WrongModalityPayloadLoader:
    def load(self, request):
        wrong_request = replace(
            request,
            modality=MediaModality.AUDIO,
        )
        return DecodedMedia(
            request=wrong_request,
            tensor=torch.zeros(1, 7),
            length=7,
            timestamps=None,
            seconds_per_grid=None,
            metadata={"sample_rate": 16_000},
        )


def media_item(
    source_id,
    modality,
    *,
    path=None,
    timeline_offset_seconds=0.0,
    timestamps=None,
):
    return {
        "id": source_id,
        "modality": modality,
        "path": path or f"{source_id}.media",
        "timeline_offset_seconds": timeline_offset_seconds,
        **({"timestamps": timestamps} if timestamps is not None else {}),
    }


def collator(
    loader=None,
    *,
    max_seq_length=32,
    quarantine_bad_samples=False,
):
    return ProfileStage2Collator(
        SentinelTokenizer(),
        max_seq_length=max_seq_length,
        token_schema=MultimodalTokenSchema.qwen3(),
        media_loader=loader or ControlledLoader(),
        quarantine_bad_samples=quarantine_bad_samples,
    )


def test_omitted_optional_media_needs_no_sentinel():
    batch = collator()(
        [
            {
                "id": "text-only",
                "input_text": "A",
                "target_text": "B",
            }
        ]
    )

    assert batch["_sample_ids"] == ["text-only"]
    assert batch["decoded_media"] == ()
    assert batch["_media_errors"] == []


def test_two_images_and_two_audio_items_keep_canonical_prompt_order():
    batch = collator()(
        [
            {
                "id": "many",
                "input_text": (
                    "a<|image_pad|>b<|image_pad|>"
                    "c<|audio_pad|>d<|audio_pad|>"
                ),
                "target_text": "T",
                "media": [
                    media_item("image-0", "image"),
                    media_item("image-1", "image"),
                    media_item("audio-0", "audio"),
                    media_item("audio-1", "audio"),
                ],
            }
        ]
    )

    assert [
        (
            item.request.modality,
            item.request.source_id,
            item.request.sample_index,
            item.request.item_index,
        )
        for item in batch["decoded_media"]
    ] == [
        (MediaModality.IMAGE, "image-0", 0, 0),
        (MediaModality.IMAGE, "image-1", 0, 1),
        (MediaModality.AUDIO, "audio-0", 0, 2),
        (MediaModality.AUDIO, "audio-1", 0, 3),
    ]


def test_two_rows_keep_distinct_sample_and_item_indices():
    batch = collator()(
        [
            {
                "id": "row-0",
                "input_text": "<|image_pad|><|audio_pad|>",
                "target_text": "A",
                "media": [
                    media_item("image-0", "image"),
                    media_item("audio-0", "audio"),
                ],
            },
            {
                "id": "row-1",
                "input_text": "<|video_pad|>",
                "target_text": "B",
                "media": [media_item("video-0", "video")],
            },
        ]
    )

    assert [
        (item.request.sample_index, item.request.item_index)
        for item in batch["decoded_media"]
    ] == [(0, 0), (0, 1), (1, 0)]
    assert [
        item.request.original_sample_index
        for item in batch["decoded_media"]
    ] == [0, 0, 1]


@pytest.mark.parametrize(
    ("input_text", "items"),
    [
        ("<|image_pad|>", []),
        ("plain", [media_item("image-0", "image")]),
        (
            "<|image_pad|><|audio_pad|>",
            [
                media_item("audio-0", "audio"),
                media_item("image-0", "image"),
            ],
        ),
    ],
)
def test_sentinel_and_item_sequences_must_match_exactly(
    input_text,
    items,
):
    with pytest.raises(MediaLoadError, match="sentinel"):
        collator()(
            [
                {
                    "id": "mismatch",
                    "input_text": input_text,
                    "target_text": "T",
                    "media": items,
                }
            ]
        )


def test_strict_mode_propagates_first_structured_decode_error():
    loader = ControlledLoader(bad_paths={"bad-image"})

    with pytest.raises(MediaLoadError) as captured:
        collator(loader)(
            [
                {
                    "id": "bad-row",
                    "input_text": "<|image_pad|>",
                    "target_text": "T",
                    "media": [
                        media_item(
                            "image-0",
                            "image",
                            path="bad-image",
                        )
                    ],
                }
            ]
        )

    assert captured.value.to_dict()["sample_id"] == "bad-row"
    assert captured.value.to_dict()["path"] == "bad-image"
    assert captured.value.to_dict()["modality"] == "image"


@pytest.mark.parametrize(
    "item",
    [
        media_item(
            "invalid-offset",
            "video",
            timeline_offset_seconds=-0.1,
        ),
        media_item(
            "invalid-timestamp",
            "video",
            timestamps=[0.0, float("nan")],
        ),
    ],
)
def test_invalid_request_metadata_is_a_structured_strict_error(item):
    with pytest.raises(MediaLoadError) as captured:
        collator()(
            [
                {
                    "id": "invalid-request",
                    "input_text": "<|video_pad|>",
                    "target_text": "T",
                    "media": [item],
                }
            ]
        )

    error = captured.value
    assert error.to_dict()["sample_id"] == "invalid-request"
    assert error.to_dict()["modality"] == "video"
    assert error.to_dict()["path"] == item["path"]
    assert error.to_dict()["error_type"] == "ValueError"


@pytest.mark.parametrize(
    "item",
    [
        media_item(
            "invalid-offset",
            "video",
            timeline_offset_seconds=-0.1,
        ),
        media_item(
            "invalid-timestamp",
            "video",
            timestamps=[0.0, float("nan")],
        ),
    ],
)
def test_invalid_request_metadata_quarantines_the_entire_row(item):
    batch = collator(quarantine_bad_samples=True)(
        [
            {
                "id": "invalid-request",
                "input_text": "<|video_pad|>",
                "target_text": "X",
                "media": [item],
            },
            {
                "id": "good-row",
                "input_text": "Q",
                "target_text": "Y",
            },
        ]
    )

    assert batch["_sample_ids"] == ["good-row"]
    assert batch["decoded_media"] == ()
    assert batch["_media_errors"][0]["sample_id"] == "invalid-request"
    assert batch["_media_errors"][0]["path"] == item["path"]


def test_wrong_modality_payload_is_structured_and_quarantined():
    batch = collator(
        WrongModalityPayloadLoader(),
        quarantine_bad_samples=True,
    )(
        [
            {
                "id": "wrong-payload",
                "input_text": "<|image_pad|>",
                "target_text": "X",
                "media": [media_item("image-0", "image")],
            },
            {
                "id": "good-row",
                "input_text": "Q",
                "target_text": "Y",
            },
        ]
    )

    assert batch["_sample_ids"] == ["good-row"]
    assert batch["decoded_media"] == ()
    assert batch["_media_errors"][0]["sample_id"] == "wrong-payload"
    assert batch["_media_errors"][0]["modality"] == "image"
    assert batch["_media_errors"][0]["error_type"] == "ValueError"


def test_bad_first_row_is_removed_atomically_and_good_row_is_reindexed():
    loader = ControlledLoader(bad_paths={"bad-image"})
    batch = collator(loader, quarantine_bad_samples=True)(
        [
            {
                "id": "bad-row",
                "input_text": "<|image_pad|>",
                "target_text": "X",
                "media": [
                    media_item(
                        "bad-source",
                        "image",
                        path="bad-image",
                    )
                ],
            },
            {
                "id": "good-row",
                "input_text": "a<|audio_pad|>",
                "target_text": "Y",
                "media": [
                    media_item(
                        "good-source",
                        "audio",
                        path="good-audio",
                    )
                ],
            },
        ]
    )

    assert batch["input_ids"].shape[0] == 1
    assert batch["attention_mask"].shape[0] == 1
    assert batch["labels"].shape[0] == 1
    assert batch["_sample_ids"] == ["good-row"]
    assert len(batch["decoded_media"]) == 1
    request = batch["decoded_media"][0].request
    assert request.sample_index == 0
    assert request.original_sample_index == 1
    assert request.item_index == 0
    assert batch["_media_errors"] == [
        {
            "sample_id": "bad-row",
            "modality": "image",
            "path": "bad-image",
            "error_type": "ValueError",
            "error": "corrupt fixture: bad-image",
        }
    ]
    assert SPECIAL_IDS["<|image_pad|>"] not in batch["input_ids"]


def test_middle_bad_row_produces_dense_final_sample_indices():
    loader = ControlledLoader(bad_paths={"bad-middle"})
    batch = collator(loader, quarantine_bad_samples=True)(
        [
            {
                "id": "first",
                "input_text": "<|image_pad|>",
                "target_text": "A",
                "media": [media_item("first-image", "image")],
            },
            {
                "id": "middle",
                "input_text": "<|image_pad|>",
                "target_text": "B",
                "media": [
                    media_item(
                        "bad-middle-image",
                        "image",
                        path="bad-middle",
                    )
                ],
            },
            {
                "id": "last",
                "input_text": "<|audio_pad|>",
                "target_text": "C",
                "media": [media_item("last-audio", "audio")],
            },
        ]
    )

    assert batch["_sample_ids"] == ["first", "last"]
    assert [
        (
            item.request.sample_index,
            item.request.original_sample_index,
            item.request.item_index,
        )
        for item in batch["decoded_media"]
    ] == [(0, 0, 0), (1, 2, 0)]


def test_loader_observes_original_identity_before_final_dense_reindex():
    loader = ControlledLoader(
        bad_paths={"bad-first", "bad-second"},
    )
    batch = collator(loader, quarantine_bad_samples=True)(
        [
            {
                "id": "first-bad",
                "input_text": "<|image_pad|>",
                "target_text": "A",
                "media": [
                    media_item(
                        "first-bad-item",
                        "image",
                        path="bad-first",
                    )
                ],
            },
            {
                "id": "second-bad",
                "input_text": "<|image_pad|><|audio_pad|>",
                "target_text": "B",
                "media": [
                    media_item("decoded-then-dropped", "image"),
                    media_item(
                        "second-bad-item",
                        "audio",
                        path="bad-second",
                    ),
                ],
            },
            {
                "id": "third-good",
                "input_text": "<|video_pad|>",
                "target_text": "C",
                "media": [media_item("final-video", "video")],
            },
        ]
    )

    observed = {
        request.source_id: (
            request.sample_index,
            request.original_sample_index,
        )
        for request in loader.seen
    }
    assert observed["decoded-then-dropped"] == (1, 1)
    assert observed["second-bad-item"] == (1, 1)
    assert observed["final-video"] == (2, 2)
    assert batch["_sample_ids"] == ["third-good"]
    final_request = batch["decoded_media"][0].request
    assert final_request.sample_index == 0
    assert final_request.original_sample_index == 2


def test_corrupt_second_item_removes_every_item_from_that_row():
    loader = ControlledLoader(bad_paths={"bad-second"})
    batch = collator(loader, quarantine_bad_samples=True)(
        [
            {
                "id": "multi-bad",
                "input_text": "<|image_pad|><|audio_pad|>",
                "target_text": "A",
                "media": [
                    media_item("decoded-first", "image"),
                    media_item(
                        "broken-second",
                        "audio",
                        path="bad-second",
                    ),
                ],
            },
            {
                "id": "only-good",
                "input_text": "<|video_pad|>",
                "target_text": "B",
                "media": [media_item("kept-video", "video")],
            },
        ]
    )

    assert batch["_sample_ids"] == ["only-good"]
    assert [
        item.request.source_id for item in batch["decoded_media"]
    ] == ["kept-video"]
    assert batch["decoded_media"][0].request.sample_index == 0
    assert all(
        item.request.sample_id != "multi-bad"
        for item in batch["decoded_media"]
    )


def test_final_prompt_truncation_cannot_leave_an_extra_media_item():
    loader = ControlledLoader()
    batch = collator(
        loader,
        max_seq_length=4,
        quarantine_bad_samples=True,
    )(
        [
            {
                "id": "truncated",
                "input_text": "<|image_pad|>ABCDE",
                "target_text": "T",
                "media": [media_item("image-0", "image")],
            },
            {
                "id": "kept",
                "input_text": "Q",
                "target_text": "R",
            },
        ]
    )

    assert batch["_sample_ids"] == ["kept"]
    assert batch["decoded_media"] == ()
    assert batch["_media_errors"][0]["sample_id"] == "truncated"
    assert batch["_media_errors"][0]["error_type"] == "ValueError"
    assert all(
        request.sample_id != "truncated" for request in loader.seen
    )


def test_all_quarantined_raises_aggregate_error_without_source_paths():
    loader = ControlledLoader(bad_paths={"/secret/one", "/secret/two"})

    with pytest.raises(MediaLoadError) as captured:
        collator(loader, quarantine_bad_samples=True)(
            [
                {
                    "id": "bad-1",
                    "input_text": "<|image_pad|>",
                    "target_text": "A",
                    "media": [
                        media_item(
                            "image-1",
                            "image",
                            path="/secret/one",
                        )
                    ],
                },
                {
                    "id": "bad-2",
                    "input_text": "<|audio_pad|>",
                    "target_text": "B",
                    "media": [
                        media_item(
                            "audio-2",
                            "audio",
                            path="/secret/two",
                        )
                    ],
                },
            ]
        )

    error = captured.value
    assert error.to_dict() == {
        "sample_id": "<all-quarantined>",
        "modality": "batch",
        "path": "<multiple>",
        "error_type": "AllSamplesQuarantinedError",
        "error": "all 2 samples were quarantined",
    }
    assert isinstance(error.cause, AllSamplesQuarantinedError)
    assert error.cause.quarantined_count == 2
    assert "/secret/one" not in str(error)
    assert "/secret/two" not in str(error)
    assert "/secret/one" not in str(error.to_dict())
    assert "/secret/two" not in str(error.to_dict())


def test_target_supervision_is_reserved_before_prompt_truncation():
    batch = collator(max_seq_length=4)(
        [
            {
                "id": "long-prompt",
                "input_text": "ABCDEFG",
                "target_text": "xy",
            }
        ]
    )

    assert batch["input_ids"].shape == (1, 4)
    assert batch["labels"][0, :2].tolist() == [-100, -100]
    assert batch["labels"][0, 2:].tolist() == (
        batch["input_ids"][0, 2:].tolist()
    )


def test_media_tensor_is_not_truncated_or_padded_by_collator():
    batch = collator()(
        [
            {
                "id": "long-audio",
                "input_text": "<|audio_pad|>",
                "target_text": "T",
                "media": [
                    media_item(
                        "long-audio-0",
                        "audio",
                        path="long-audio",
                    )
                ],
            }
        ]
    )

    assert batch["decoded_media"][0].tensor.shape == (1, 12_345)
    assert batch["decoded_media"][0].length == 12_345


def test_timeline_metadata_is_forwarded_without_rewriting():
    batch = collator()(
        [
            {
                "id": "offset-video",
                "input_text": "<|video_pad|>",
                "target_text": "T",
                "media": [
                    media_item(
                        "video-0",
                        "video",
                        timeline_offset_seconds=3.25,
                        timestamps=[0.0, 0.5],
                    )
                ],
            }
        ]
    )

    request = batch["decoded_media"][0].request
    assert request.timeline_offset_seconds == 3.25
    assert request.timestamps == (0.0, 0.5)


def test_legacy_single_media_fields_are_not_consumed_by_profile_collator():
    batch = collator()(
        [
            {
                "id": "legacy-fields",
                "input_text": "plain",
                "target_text": "T",
                "image_path": "legacy.png",
                "audio_path": "legacy.wav",
            }
        ]
    )

    assert batch["decoded_media"] == ()


def test_quarantine_flag_must_be_a_boolean():
    with pytest.raises(TypeError, match="quarantine_bad_samples"):
        ProfileStage2Collator(
            SentinelTokenizer(),
            token_schema=MultimodalTokenSchema.qwen3(),
            quarantine_bad_samples=1,
        )
