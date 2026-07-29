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


def test_corrupt_referenced_media_raises_by_default(tmp_path: Path):
    bad_image = tmp_path / "bad.jpg"
    bad_image.write_text("not an image", encoding="utf-8")
    collator = collators.OmniStage2Collator(
        TinyTokenizer(), max_seq_length=4, image_size=2, max_audio_len=4
    )

    with pytest.raises(collators.MediaLoadError) as exc_info:
        collator(
            [
                {
                    "id": "bad-image",
                    "input_text": "A",
                    "target_text": "B",
                    "image_path": str(bad_image),
                }
            ]
        )

    message = str(exc_info.value)
    assert "bad-image" in message
    assert str(bad_image) in message
    assert "image" in message


def test_skip_bad_media_is_explicit_and_records_error(tmp_path: Path):
    bad_audio = tmp_path / "bad.wav"
    bad_audio.write_text("not audio", encoding="utf-8")
    collator = collators.OmniStage2Collator(
        TinyTokenizer(),
        max_seq_length=4,
        image_size=2,
        max_audio_len=4,
        skip_bad_media=True,
    )

    batch = collator(
        [
            {
                "id": "bad-audio",
                "input_text": "A",
                "target_text": "B",
                "audio_path": str(bad_audio),
            }
        ]
    )

    assert batch["has_audio"].tolist() == [0]
    assert len(batch["_media_errors"]) == 1
    error = batch["_media_errors"][0]
    assert error["sample_id"] == "bad-audio"
    assert error["modality"] == "audio"
    assert error["path"] == str(bad_audio)


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
