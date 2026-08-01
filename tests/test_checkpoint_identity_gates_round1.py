from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch
import yaml

from qwen3_omni_pretrain import cli_infer_thinker as inference_cli
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.training import trainer_thinker


class IdentityRejected(RuntimeError):
    pass


class FullAllocationReached(RuntimeError):
    pass


class _Backend:
    def to_str(self) -> str:
        return json.dumps(
            {
                "model": {
                    "type": "WordLevel",
                    "vocab": {str(index): index for index in range(5)},
                },
                "pre_tokenizer": {"type": "Whitespace"},
            }
        )


class _Tokenizer:
    backend_tokenizer = _Backend()
    additional_special_tokens_ids: list[int] = []
    pad_token_id = 0
    eos_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        return {
            "<|image_pad|>": 1,
            "<|video_pad|>": 2,
            "<|audio_pad|>": 3,
            "<|audio_start|>": 4,
            "<|audio_end|>": 5,
        }

    def __len__(self) -> int:
        return 6


def _config() -> Qwen3OmniMoeConfig:
    return Qwen3OmniMoeConfig(
        vocab_size=8,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 8,
            "use_moe": False,
        },
    )


def test_stage1_inference_identity_gate_runs_before_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    tokenizer = _Tokenizer()
    config = _config()
    monkeypatch.setattr(
        inference_cli,
        "_ensure_tokenizer",
        lambda *args, **kwargs: events.append("tokenizer") or tokenizer,
    )
    monkeypatch.setattr(
        inference_cli.Qwen3OmniMoeConfig,
        "from_legacy_pretrained_config",
        classmethod(
            lambda cls, checkpoint: events.append("config") or config
        ),
    )

    def reject(*args, **kwargs):
        events.append("identity-gate")
        raise IdentityRejected("stage1 mismatch")

    monkeypatch.setattr(
        inference_cli,
        "load_checkpoint_metadata",
        reject,
        raising=False,
    )
    monkeypatch.setattr(
        inference_cli.Qwen3OmniMoeThinkerTextModel,
        "from_pretrained",
        classmethod(
            lambda cls, *args, **kwargs: events.append("weight-load")
            or (_ for _ in ()).throw(
                AssertionError("weights loaded before identity gate")
            )
        ),
    )

    with pytest.raises(IdentityRejected, match="stage1 mismatch"):
        inference_cli.run_stage1(
            Namespace(
                checkpoint="checkpoint",
                tokenizer_name_or_path=None,
                dtype="auto",
            )
        )

    assert events == ["tokenizer", "config", "identity-gate"]


def test_stage2_inference_identity_gate_runs_before_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    tokenizer = _Tokenizer()
    config = _config()
    monkeypatch.setattr(
        inference_cli.Qwen3OmniMoeConfig,
        "from_legacy_pretrained_config",
        classmethod(
            lambda cls, checkpoint: events.append("config") or config
        ),
    )

    def reject(*args, **kwargs):
        events.append("identity-gate")
        raise IdentityRejected("stage2 mismatch")

    monkeypatch.setattr(
        inference_cli,
        "load_checkpoint_metadata",
        reject,
        raising=False,
    )
    monkeypatch.setattr(
        inference_cli.Qwen3OmniMoeThinkerVisionAudioModel,
        "from_pretrained",
        classmethod(
            lambda cls, *args, **kwargs: events.append("weight-load")
            or (_ for _ in ()).throw(
                AssertionError("weights loaded before identity gate")
            )
        ),
    )

    with pytest.raises(IdentityRejected, match="stage2 mismatch"):
        inference_cli._load_reconciled_stage2_model(
            "checkpoint",
            tokenizer,
        )

    assert events == ["config", "identity-gate"]


def test_stage2_training_resume_gate_runs_before_full_model_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 8,
                "thinker_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 8,
                    "use_moe": False,
                    "routing_kind": "dense",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trainer_thinker.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    events: list[str] = []

    def reject_preflight(*args, **kwargs):
        events.append("identity-gate")
        raise IdentityRejected("resume mismatch")

    monkeypatch.setattr(
        trainer_thinker,
        "_preflight_training_resume",
        reject_preflight,
        raising=False,
    )
    monkeypatch.setattr(
        trainer_thinker,
        "_build_reconciled_stage2_model",
        lambda *args, **kwargs: events.append("full-allocation")
        or (_ for _ in ()).throw(FullAllocationReached()),
    )
    raw = {
        "experiment_name": "identity-order",
        "stage1_init_ckpt": "unused-stage1",
        "model": {"model_config_path": str(model_config)},
        "data": {
            "train_corpus_path": "train.jsonl",
            "val_corpus_path": "val.jsonl",
            "image_root": "images",
            "audio_root": "audio",
        },
        "train": {
            "output_dir": str(tmp_path / "output"),
            "resume_from_checkpoint": str(tmp_path / "resume"),
        },
    }

    with pytest.raises(IdentityRejected, match="resume mismatch"):
        trainer_thinker.train_thinker_stage2(
            raw,
            tokenizer_name_or_path="tokenizer",
        )

    assert events == ["identity-gate"]


def test_preflight_descriptor_builds_only_meta_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    observed_devices: list[set[torch.device]] = []

    def describe(model):
        observed_devices.append(
            {parameter.device for parameter in model.parameters()}
        )
        return object()

    monkeypatch.setattr(
        trainer_thinker,
        "describe_model_topology",
        describe,
        raising=False,
    )
    monkeypatch.setattr(
        trainer_thinker,
        "load_checkpoint_metadata",
        lambda *args, **kwargs: None,
    )

    trainer_thinker._preflight_training_resume(
        checkpoint=str(tmp_path),
        model_config=_config(),
        tokenizer=_Tokenizer(),
        artifact_kind="stage2-training",
        model_class=(
            trainer_thinker.Qwen3OmniMoeThinkerVisionAudioModel
        ),
    )

    assert observed_devices
    assert observed_devices[0] == {torch.device("meta")}
