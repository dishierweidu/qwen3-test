from __future__ import annotations

import pytest
import torch
import transformers

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.positions import (
    Qwen3DisjointPositionBuilder,
    Qwen3DisjointPositionConfig,
)
from qwen3_omni_pretrain.multimodal.types import (
    AssembledSequence,
    MediaGrid,
    MediaSource,
    SequenceSpan,
    SequenceSpanKind,
)
from tests.oracle.qwen3_omni_rope_oracle import (
    EXPECTED_ROPE_IMPLEMENTATION_SHA256,
    PINNED_TRANSFORMERS_VERSION,
    ProjectMediaVector,
    RopeVector,
    fixed_rope_vectors,
    rope_facade,
    rope_implementation_sha256,
)


pytestmark = pytest.mark.reference


def _optional_grid(
    values: tuple[tuple[int, int, int], ...] | None,
) -> torch.Tensor | None:
    if values is None:
        return None
    if not values:
        return torch.empty((0, 3), dtype=torch.long)
    return torch.tensor(values, dtype=torch.long)


def _official_result(vector: RopeVector) -> tuple[torch.Tensor, torch.Tensor]:
    facade, _ = rope_facade()
    return facade.get_rope_index(
        input_ids=torch.tensor(vector.input_ids, dtype=torch.long),
        image_grid_thw=_optional_grid(vector.image_grid_thw),
        video_grid_thw=_optional_grid(vector.video_grid_thw),
        attention_mask=torch.tensor(
            vector.attention_mask,
            dtype=torch.long,
        ),
        audio_seqlens=(
            None
            if vector.audio_seqlens is None
            else torch.tensor(vector.audio_seqlens, dtype=torch.long)
        ),
        second_per_grids=(
            None
            if vector.second_per_grids is None
            else torch.tensor(
                vector.second_per_grids,
                dtype=torch.float32,
            )
        ),
        use_audio_in_video=False,
    )


def _media_span(spec: ProjectMediaVector) -> SequenceSpan:
    modality = MediaModality(spec.modality)
    source = MediaSource(
        spec.sample_index,
        spec.item_index,
        spec.source_id,
    )
    return SequenceSpan(
        sample_index=spec.sample_index,
        start=spec.start,
        end=spec.end,
        kind=SequenceSpanKind.MEDIA,
        modality=modality,
        grid=(None if spec.grid is None else MediaGrid(*spec.grid)),
        timestamps=(
            None
            if spec.timestamps is None
            else torch.tensor(spec.timestamps, dtype=torch.float32)
        ),
        seconds_per_grid=spec.seconds_per_grid,
        source=source,
        source_token_indices=tuple(range(spec.end - spec.start)),
    )


def _project_assembled(vector: RopeVector) -> AssembledSequence:
    input_ids = torch.tensor(vector.input_ids, dtype=torch.long)
    mask = torch.tensor(vector.attention_mask, dtype=torch.bool)
    spans: list[SequenceSpan] = []
    media_by_sample: dict[int, list[ProjectMediaVector]] = {}
    for spec in vector.project_media:
        media_by_sample.setdefault(spec.sample_index, []).append(spec)

    for sample_index in range(input_ids.shape[0]):
        valid_length = int(mask[sample_index].sum().item())
        cursor = 0
        for spec in sorted(
            media_by_sample.get(sample_index, []),
            key=lambda value: value.start,
        ):
            if cursor < spec.start:
                spans.append(
                    SequenceSpan(
                        sample_index=sample_index,
                        start=cursor,
                        end=spec.start,
                        kind=SequenceSpanKind.TEXT,
                        modality=None,
                        grid=None,
                        timestamps=None,
                        seconds_per_grid=None,
                        source=None,
                        source_token_indices=None,
                    )
                )
            spans.append(_media_span(spec))
            cursor = spec.end
        if cursor < valid_length:
            spans.append(
                SequenceSpan(
                    sample_index=sample_index,
                    start=cursor,
                    end=valid_length,
                    kind=SequenceSpanKind.TEXT,
                    modality=None,
                    grid=None,
                    timestamps=None,
                    seconds_per_grid=None,
                    source=None,
                    source_token_indices=None,
                )
            )

    result = AssembledSequence(
        expanded_input_ids=input_ids,
        inputs_embeds=torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            4,
        ),
        attention_mask=mask,
        labels=None,
        spans=tuple(
            sorted(spans, key=lambda span: (span.sample_index, span.start))
        ),
    )
    result.validate()
    return result


def test_qwen3_rope_implementation_version_and_hash_are_pinned():
    assert transformers.__version__ == PINNED_TRANSFORMERS_VERSION
    _, base = rope_facade()
    assert rope_implementation_sha256(base) == (
        EXPECTED_ROPE_IMPLEMENTATION_SHA256
    )


@pytest.mark.parametrize(
    "vector",
    fixed_rope_vectors(),
    ids=lambda vector: vector.name,
)
def test_official_facade_and_project_builder_match_fixed_vectors(vector):
    official_positions, official_deltas = _official_result(vector)
    expected_positions = torch.tensor(
        vector.expected_positions,
        dtype=torch.float32,
    )
    expected_deltas = torch.tensor(
        vector.expected_deltas,
        dtype=torch.float32,
    )
    assert torch.equal(official_positions, expected_positions)
    assert torch.equal(official_deltas, expected_deltas)

    assembled = _project_assembled(vector)
    project = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig(
            position_id_per_seconds=13.0,
            rotary_sections=(24, 20, 20),
        )
    ).build(assembled)
    valid = (
        assembled.attention_mask
        .unsqueeze(0)
        .expand_as(project.position_ids)
    )
    assert torch.equal(
        project.position_ids.masked_select(valid),
        expected_positions.masked_select(valid),
    )
    assert torch.count_nonzero(
        project.position_ids.masked_select(~valid)
    ) == 0
    assert torch.equal(project.rope_deltas, expected_deltas)
    assert project.position_ids.dtype is torch.float32
    assert project.position_ids.shape == expected_positions.shape


def test_oracle_maps_official_premerge_grids_to_project_output_grids():
    for vector in fixed_rope_vectors():
        image_index = 0
        video_index = 0
        for media in vector.project_media:
            if media.modality == "image":
                assert vector.image_grid_thw is not None
                official = vector.image_grid_thw[image_index]
                image_index += 1
            elif media.modality == "video":
                assert vector.video_grid_thw is not None
                official = vector.video_grid_thw[video_index]
                video_index += 1
            else:
                continue
            assert media.grid is not None
            assert media.grid == (
                official[0],
                official[1] // 2,
                official[2] // 2,
            )
