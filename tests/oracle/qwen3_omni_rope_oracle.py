from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from types import SimpleNamespace


PINNED_TRANSFORMERS_VERSION = "5.2.0"
EXPECTED_ROPE_IMPLEMENTATION_SHA256 = (
    "4b0de5c1b83a32c7fbb57b16c5fc0c1c6e5a707585cae9188fc87d7283b44ee3"
)

IMAGE_TOKEN_ID = 151_655
VIDEO_TOKEN_ID = 151_656
AUDIO_TOKEN_ID = 151_675
VISION_START_TOKEN_ID = 151_652
VISION_END_TOKEN_ID = 151_653
AUDIO_START_TOKEN_ID = 151_669
AUDIO_END_TOKEN_ID = 151_670


@dataclass(frozen=True)
class ProjectMediaVector:
    sample_index: int
    item_index: int
    source_id: str
    modality: str
    start: int
    end: int
    grid: tuple[int, int, int] | None = None
    timestamps: tuple[float, ...] | None = None
    seconds_per_grid: float | None = None


@dataclass(frozen=True)
class RopeVector:
    name: str
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    expected_positions: tuple[
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
    ]
    expected_deltas: tuple[tuple[float], ...]
    project_media: tuple[ProjectMediaVector, ...] = ()
    image_grid_thw: tuple[tuple[int, int, int], ...] | None = None
    video_grid_thw: tuple[tuple[int, int, int], ...] | None = None
    audio_seqlens: tuple[int, ...] | None = None
    second_per_grids: tuple[float, ...] | None = None


def rope_facade():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoePreTrainedModelForConditionalGeneration,
    )

    base = Qwen3OmniMoePreTrainedModelForConditionalGeneration

    class RopeFacade:
        spatial_merge_size = 2
        config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=13,
        )
        get_llm_pos_ids_for_vision = base.get_llm_pos_ids_for_vision
        get_rope_index = base.get_rope_index

    return RopeFacade(), base


def rope_implementation_sha256(base) -> str:
    source = (
        inspect.getsource(base.get_llm_pos_ids_for_vision)
        + inspect.getsource(base.get_rope_index)
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def fixed_rope_vectors() -> tuple[RopeVector, ...]:
    text_three = (
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
    )
    padded_raw = (
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    )
    image_positions = (
        (0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0, 5.0),
        (0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0),
        (0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0),
    )
    mixed_positions = tuple(
        (
            image_positions[axis],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        for axis in range(3)
    )
    integer_video = (
        (0.0, 1.0, 1.0, 1.0, 1.0, 14.0, 14.0, 14.0, 14.0, 15.0),
        (0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 15.0),
        (0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 15.0),
    )
    fractional_video = (
        (
            0.0,
            1.0,
            2.0399999618530273,
            3.0799999237060547,
            4.119999885559082,
            5.159999847412109,
            6.199999809265137,
            7.239999771118164,
            8.280000686645508,
            9.280000686645508,
        ),
        (
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            9.280000686645508,
        ),
        (
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            9.280000686645508,
        ),
    )
    audio_positions = (
        (0.0, 1.0, 2.0, 3.0),
        (0.0, 1.0, 2.0, 3.0),
        (0.0, 1.0, 2.0, 3.0),
    )

    return (
        RopeVector(
            name="text_no_padding",
            input_ids=((11, 12, 13),),
            attention_mask=((1, 1, 1),),
            expected_positions=tuple((axis,) for axis in text_three),
            expected_deltas=((0.0,),),
        ),
        RopeVector(
            name="text_one_valid_with_padding",
            input_ids=((11, 0, 0),),
            attention_mask=((1, 0, 0),),
            expected_positions=tuple((axis,) for axis in padded_raw),
            expected_deltas=((1.0,),),
        ),
        RopeVector(
            name="text_two_valid_with_padding",
            input_ids=((11, 12, 0),),
            attention_mask=((1, 1, 0),),
            expected_positions=tuple((axis,) for axis in padded_raw),
            expected_deltas=((0.0,),),
        ),
        RopeVector(
            name="image",
            input_ids=(
                (
                    7,
                    VISION_START_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    VISION_END_TOKEN_ID,
                    8,
                ),
            ),
            attention_mask=((1, 1, 1, 1, 1, 1, 1, 1),),
            image_grid_thw=((1, 4, 4),),
            expected_positions=tuple((axis,) for axis in image_positions),
            expected_deltas=((-2.0,),),
            project_media=(
                ProjectMediaVector(
                    sample_index=0,
                    item_index=0,
                    source_id="image-0",
                    modality="image",
                    start=2,
                    end=6,
                    grid=(1, 2, 2),
                ),
            ),
        ),
        RopeVector(
            name="mixed_image_and_padded_text",
            input_ids=(
                (
                    7,
                    VISION_START_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    VISION_END_TOKEN_ID,
                    8,
                ),
                (9, 0, 0, 0, 0, 0, 0, 0),
            ),
            attention_mask=(
                (1, 1, 1, 1, 1, 1, 1, 1),
                (1, 0, 0, 0, 0, 0, 0, 0),
            ),
            image_grid_thw=((1, 4, 4),),
            expected_positions=mixed_positions,
            expected_deltas=((-2.0,), (0.0,)),
            project_media=(
                ProjectMediaVector(
                    sample_index=0,
                    item_index=0,
                    source_id="image-0",
                    modality="image",
                    start=2,
                    end=6,
                    grid=(1, 2, 2),
                ),
            ),
        ),
        RopeVector(
            name="audio",
            input_ids=(
                (
                    AUDIO_START_TOKEN_ID,
                    AUDIO_TOKEN_ID,
                    AUDIO_TOKEN_ID,
                    AUDIO_END_TOKEN_ID,
                ),
            ),
            attention_mask=((1, 1, 1, 1),),
            image_grid_thw=(),
            audio_seqlens=(16,),
            expected_positions=tuple((axis,) for axis in audio_positions),
            expected_deltas=((0.0,),),
            project_media=(
                ProjectMediaVector(
                    sample_index=0,
                    item_index=0,
                    source_id="audio-0",
                    modality="audio",
                    start=1,
                    end=3,
                    timestamps=(0.0, 0.08),
                ),
            ),
        ),
        RopeVector(
            name="video_integer_cadence",
            input_ids=(
                (
                    VISION_START_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VISION_END_TOKEN_ID,
                ),
            ),
            attention_mask=((1,) * 10,),
            video_grid_thw=((2, 4, 4),),
            second_per_grids=(1.0,),
            expected_positions=tuple((axis,) for axis in integer_video),
            expected_deltas=((6.0,),),
            project_media=(
                ProjectMediaVector(
                    sample_index=0,
                    item_index=0,
                    source_id="video-0",
                    modality="video",
                    start=1,
                    end=9,
                    grid=(2, 2, 2),
                    timestamps=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
                    seconds_per_grid=1.0,
                ),
            ),
        ),
        RopeVector(
            name="video_fractional_cadence",
            input_ids=(
                (
                    VISION_START_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VISION_END_TOKEN_ID,
                ),
            ),
            attention_mask=((1,) * 10,),
            video_grid_thw=((8, 2, 2),),
            second_per_grids=(0.08,),
            expected_positions=tuple((axis,) for axis in fractional_video),
            expected_deltas=((0.2800006866455078,),),
            project_media=(
                ProjectMediaVector(
                    sample_index=0,
                    item_index=0,
                    source_id="video-0",
                    modality="video",
                    start=1,
                    end=9,
                    grid=(8, 1, 1),
                    timestamps=(0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56),
                    seconds_per_grid=0.08,
                ),
            ),
        ),
    )


__all__ = [
    "EXPECTED_ROPE_IMPLEMENTATION_SHA256",
    "PINNED_TRANSFORMERS_VERSION",
    "ProjectMediaVector",
    "RopeVector",
    "fixed_rope_vectors",
    "rope_facade",
    "rope_implementation_sha256",
]
