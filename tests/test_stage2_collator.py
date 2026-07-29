from pathlib import Path

import pytest
import torch

from conftest import TinyTokenizer
from qwen3_omni_pretrain.data import collators


def test_stage2_supervises_only_target_and_masks_padding():
    tokenizer = TinyTokenizer()
    collator = collators.OmniStage2Collator(
        tokenizer,
        max_seq_length=8,
        image_size=2,
        max_audio_len=4,
    )

    batch = collator(
        [
            {"id": "long", "input_text": "AB", "target_text": "xy"},
            {"id": "short", "input_text": "C", "target_text": "z"},
        ]
    )

    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["labels"][0, :2].tolist() == [-100, -100]
    assert batch["labels"][0, 2:4].tolist() == batch["input_ids"][0, 2:4].tolist()
    assert torch.all(batch["labels"][0, 4:] == -100)
    assert batch["labels"][1, 0].item() == -100
    assert batch["labels"][1, 1].item() == batch["input_ids"][1, 1].item()
    assert torch.all(batch["labels"][1, 2:] == -100)
    assert batch["_sample_ids"] == ["long", "short"]


def test_missing_optional_media_is_valid():
    collator = collators.OmniStage2Collator(
        TinyTokenizer(), max_seq_length=4, image_size=2, max_audio_len=4
    )
    batch = collator(
        [{"id": "no-media", "input_text": "A", "target_text": "B"}]
    )
    assert batch["has_image"].tolist() == [0]
    assert batch["has_audio"].tolist() == [0]
    assert batch["_media_errors"] == []


@pytest.mark.parametrize("create_corrupt_file", [False, True])
def test_referenced_invalid_media_raises_by_default(
    tmp_path: Path, create_corrupt_file: bool
):
    bad_image = tmp_path / "bad.jpg"
    if create_corrupt_file:
        bad_image.write_text("not an image", encoding="utf-8")
    stage2_collator = collators.OmniStage2Collator(
        TinyTokenizer(), max_seq_length=4, image_size=2, max_audio_len=4
    )

    with pytest.raises(collators.MediaLoadError) as exc_info:
        stage2_collator(
            [{
                    "id": "bad-image",
                    "input_text": "A",
                    "target_text": "B",
                    "image_path": str(bad_image),
            }]
        )

    assert "bad-image" in str(exc_info.value)
    assert str(bad_image) in str(exc_info.value)


@pytest.mark.parametrize("create_corrupt_file", [False, True])
def test_skip_bad_media_records_each_invalid_reference(
    tmp_path: Path, create_corrupt_file: bool
):
    bad_audio = tmp_path / "bad.wav"
    if create_corrupt_file:
        bad_audio.write_text("not audio", encoding="utf-8")
    stage2_collator = collators.OmniStage2Collator(
        TinyTokenizer(),
        max_seq_length=4,
        image_size=2,
        max_audio_len=4,
        skip_bad_media=True,
    )

    batch = stage2_collator(
        [{
                "id": "bad-audio",
                "input_text": "A",
                "target_text": "B",
                "audio_path": str(bad_audio),
        }]
    )

    assert batch["has_audio"].tolist() == [0]
    error = batch["_media_errors"][0]
    assert error["sample_id"] == "bad-audio"
    assert error["modality"] == "audio"
    assert error["path"] == str(bad_audio)
    assert error["error_type"]
    assert error["error"]


def test_long_prompt_is_truncated_before_target_supervision():
    collator = collators.OmniStage2Collator(
        TinyTokenizer(), max_seq_length=4, image_size=2, max_audio_len=4
    )

    batch = collator(
        [{"id": "long-prompt", "input_text": "ABCDEFG", "target_text": "xy"}]
    )

    assert batch["input_ids"].shape == (1, 4)
    assert batch["labels"][0, :2].tolist() == [-100, -100]
    assert batch["labels"][0, 2:].tolist() == batch["input_ids"][0, 2:].tolist()


def test_media_loader_distinguishes_omitted_from_missing(tmp_path: Path):
    loader = collators.Stage2MediaLoader(image_size=2, max_audio_len=4)
    omitted = loader.load_optional(
        modality="image", path=None, sample_id="omitted"
    )
    assert omitted.present == 0
    assert omitted.error is None

    missing = tmp_path / "missing.jpg"
    with pytest.raises(collators.MediaLoadError, match="missing"):
        loader.load_optional(
            modality="image", path=str(missing), sample_id="referenced"
        )


def test_media_loader_skip_returns_structured_error(tmp_path: Path):
    missing_audio = tmp_path / "missing.wav"
    loader = collators.Stage2MediaLoader(
        image_size=2, max_audio_len=4, skip_bad_media=True
    )
    result = loader.load_optional(
        modality="audio",
        path=str(missing_audio),
        sample_id="audio-7",
    )
    assert result.present == 0
    assert result.error["sample_id"] == "audio-7"
    assert result.error["modality"] == "audio"
    assert result.error["path"] == str(missing_audio)
    assert result.error["error_type"]
    assert result.error["error"]


def test_trainer_collator_helper_uses_the_shared_media_policy():
    from types import SimpleNamespace

    from qwen3_omni_pretrain.training import trainer_thinker

    runtime = SimpleNamespace(max_seq_length=4, skip_bad_media=True)
    stage2_collator = trainer_thinker._build_stage2_collator(
        runtime, TinyTokenizer()
    )
    assert isinstance(
        stage2_collator.media_loader, collators.Stage2MediaLoader
    )
    assert stage2_collator.media_loader.skip_bad_media is True
