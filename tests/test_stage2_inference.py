import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from qwen3_omni_pretrain import cli_infer_thinker as cli
from qwen3_omni_pretrain.data.collators import (
    MediaLoadError,
    Stage2MediaLoader,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)


class TinyInferenceTokenizer:
    eos_token_id = None
    _vocab = {
        "<|image_pad|>": 10,
        "<|video_pad|>": 11,
        "<|audio_pad|>": 12,
        "<|audio_start|>": 13,
        "<|audio_end|>": 14,
    }

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "x"

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self):
        return 15


class TinyInferenceModel(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), 4)
        logits[:, -1, 2] = 1
        return {"logits": logits}


def test_parse_args_accepts_explicit_skip_bad_media(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3-omni-infer",
            "--stage",
            "stage2",
            "--checkpoint",
            "checkpoint",
            "--skip_bad_media",
        ],
    )
    assert cli.parse_args().skip_bad_media is True


def test_stage2_inference_omitted_media_generates_without_errors():
    result = cli.greedy_decode_stage2(
        model=TinyInferenceModel(),
        tokenizer=TinyInferenceTokenizer(),
        media_loader=Stage2MediaLoader(image_size=2, max_audio_len=4),
        device=torch.device("cpu"),
        sample={"id": "sample-0", "input_text": "A"},
        max_new_tokens=1,
        image_root="",
        audio_root="",
        max_seq_length=8,
    )
    assert result["text"] == "x"
    assert result["media_errors"] == []


@pytest.mark.parametrize("create_corrupt_file", [False, True])
def test_stage2_inference_is_strict_for_referenced_invalid_media(
    tmp_path: Path, create_corrupt_file: bool
):
    bad_image = tmp_path / "bad.jpg"
    if create_corrupt_file:
        bad_image.write_text("not an image", encoding="utf-8")
    with pytest.raises(MediaLoadError, match="sample-1"):
        cli.greedy_decode_stage2(
            model=TinyInferenceModel(),
            tokenizer=TinyInferenceTokenizer(),
            media_loader=Stage2MediaLoader(image_size=2, max_audio_len=4),
            device=torch.device("cpu"),
            sample={"id": "sample-1", "input_text": "A", "image": str(bad_image)},
            max_new_tokens=1,
            image_root="",
            audio_root="",
            max_seq_length=8,
        )


@pytest.mark.parametrize("create_corrupt_file", [False, True])
def test_stage2_inference_skip_reports_each_invalid_media_and_generates(
    tmp_path: Path, create_corrupt_file: bool
):
    bad_audio = tmp_path / "bad.wav"
    if create_corrupt_file:
        bad_audio.write_text("not audio", encoding="utf-8")
    result = cli.greedy_decode_stage2(
        model=TinyInferenceModel(),
        tokenizer=TinyInferenceTokenizer(),
        media_loader=Stage2MediaLoader(
            image_size=2, max_audio_len=4, skip_bad_media=True
        ),
        device=torch.device("cpu"),
        sample={
            "id": "sample-2",
            "input_text": "A",
            "audio": str(bad_audio),
        },
        max_new_tokens=1,
        image_root="",
        audio_root="",
        max_seq_length=8,
    )
    assert result["text"] == "x"
    assert result["stats"]["output_tokens"] == 1
    error = result["media_errors"][0]
    assert error["sample_id"] == "sample-2"
    assert error["modality"] == "audio"
    assert error["path"] == str(bad_audio)
    assert error["error_type"]
    assert error["error"]


def test_run_stage2_uses_skip_flag_and_emits_json_error(
    monkeypatch, capsys, tmp_path: Path
):
    missing_audio = tmp_path / "missing.wav"
    config = Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": False,
            "moe_shared_expert": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_ensure_tokenizer",
        lambda tokenizer_name_or_path, checkpoint: TinyInferenceTokenizer(),
    )
    monkeypatch.setattr(
        cli.Qwen3OmniMoeConfig,
        "from_pretrained",
        classmethod(lambda cls, checkpoint: config),
    )
    monkeypatch.setattr(
        cli.Qwen3OmniMoeThinkerVisionAudioModel,
        "from_pretrained",
        classmethod(
            lambda cls, checkpoint, **kwargs: TinyInferenceModel()
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_jsonl",
        lambda path, limit: [
            {
                "id": "observable",
                "input_text": "A",
                "audio": str(missing_audio),
            }
        ],
    )
    args = Namespace(
        checkpoint="checkpoint",
        tokenizer_name_or_path=None,
        dtype="auto",
        max_seq_length=8,
        max_new_tokens=1,
        image_root="",
        audio_root="",
        skip_bad_media=True,
        chat=False,
        prompt=None,
        jsonl="samples.jsonl",
        num_samples=1,
    )

    cli.run_stage2(args)

    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"media_error"')
    ]
    assert len(payloads) == 1
    error = payloads[0]["media_error"]
    assert error["sample_id"] == "observable"
    assert error["modality"] == "audio"
    assert error["path"] == str(missing_audio)
    assert error["error_type"]
    assert error["error"]
