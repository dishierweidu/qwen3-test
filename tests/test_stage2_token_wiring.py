from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import yaml

from qwen3_omni_pretrain import cli_infer_thinker as cli
from qwen3_omni_pretrain.training import trainer_thinker


class WiringReached(RuntimeError):
    pass


def test_training_builder_reconciles_before_model_construction(monkeypatch):
    events = []
    config = SimpleNamespace(vocab_size=16, image_token_id=None)
    tokenizer = object()

    def reconcile(config_arg, tokenizer_arg):
        events.append("reconcile")
        assert config_arg is config
        assert tokenizer_arg is tokenizer
        config_arg.image_token_id = 7
        return {"image_token_id": 7}

    class SpyModel:
        def __init__(self, config_arg):
            events.append("construct")
            assert config_arg is config
            assert config_arg.image_token_id == 7

    monkeypatch.setattr(
        trainer_thinker,
        "reconcile_multimodal_token_ids",
        reconcile,
        raising=False,
    )
    monkeypatch.setattr(
        trainer_thinker,
        "Qwen3OmniMoeThinkerVisionAudioModel",
        SpyModel,
    )

    model = trainer_thinker._build_reconciled_stage2_model(
        config, tokenizer
    )
    assert isinstance(model, SpyModel)
    assert events == ["reconcile", "construct"]


def test_training_entry_calls_reconciled_builder(
    monkeypatch, tmp_path
):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 16,
                "thinker_config": {
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 16,
                    "use_moe": False,
                    "moe_shared_expert": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trainer_thinker.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        trainer_thinker,
        "_build_reconciled_stage2_model",
        lambda config, tokenizer: (_ for _ in ()).throw(WiringReached()),
        raising=False,
    )
    monkeypatch.setattr(
        trainer_thinker,
        "Qwen3OmniMoeThinkerVisionAudioModel",
        lambda config: pytest.fail("entry bypassed reconciled builder"),
    )
    raw = {
        "experiment_name": "token-wiring-test",
        "stage1_init_ckpt": "unused-stage1",
        "model": {"model_config_path": str(model_yaml)},
        "data": {
            "train_corpus_path": "train.jsonl",
            "val_corpus_path": "val.jsonl",
            "image_root": "images",
            "audio_root": "audio",
        },
        "train": {"output_dir": str(tmp_path / "output")},
    }

    with pytest.raises(WiringReached):
        trainer_thinker.train_thinker_stage2(
            raw, tokenizer_name_or_path="tokenizer"
        )


def test_inference_loads_config_then_reconciles_before_weights(monkeypatch):
    events = []
    config = SimpleNamespace(vocab_size=16, image_token_id=99)
    tokenizer = object()
    captured = {}

    class SpyConfig:
        @classmethod
        def from_legacy_pretrained_config(cls, checkpoint):
            events.append("legacy-adapt")
            return config

    def reconcile(config_arg, tokenizer_arg):
        events.append("reconcile")
        config_arg.image_token_id = 7
        return {"image_token_id": 7}

    class SpyModel:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            events.append("weight-load")
            captured.update(kwargs)
            assert kwargs["config"] is config
            assert config.image_token_id == 7
            return cls()

    monkeypatch.setattr(cli, "Qwen3OmniMoeConfig", SpyConfig, raising=False)
    monkeypatch.setattr(
        cli, "reconcile_multimodal_token_ids", reconcile, raising=False
    )
    monkeypatch.setattr(
        cli, "Qwen3OmniMoeThinkerVisionAudioModel", SpyModel
    )

    model = cli._load_reconciled_stage2_model(
        "checkpoint-x",
        tokenizer,
        load_kwargs={"torch_dtype": torch.float32},
    )
    assert isinstance(model, SpyModel)
    assert events == ["legacy-adapt", "reconcile", "weight-load"]
    assert captured["torch_dtype"] is torch.float32


def test_inference_stage1_adapts_config_before_weights(monkeypatch):
    events = []
    config = object()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)

    monkeypatch.setattr(
        cli,
        "_ensure_tokenizer",
        lambda tokenizer_name_or_path, checkpoint: tokenizer,
    )

    class SpyConfig:
        @classmethod
        def from_legacy_pretrained_config(cls, checkpoint):
            events.append("legacy-adapt")
            return config

    class SpyModel:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            events.append("weight-load")
            assert kwargs["config"] is config
            raise WiringReached()

    monkeypatch.setattr(cli, "Qwen3OmniMoeConfig", SpyConfig)
    monkeypatch.setattr(cli, "Qwen3OmniMoeThinkerTextModel", SpyModel)
    args = Namespace(
        checkpoint="checkpoint",
        tokenizer_name_or_path=None,
        dtype="auto",
    )

    with pytest.raises(WiringReached):
        cli.run_stage1(args)

    assert events == ["legacy-adapt", "weight-load"]


def test_training_stage1_initialization_adapts_config_before_weights(
    monkeypatch,
):
    events = []
    expected_architecture = object()
    expected_tokenizer_sha256 = "a" * 64
    manifest = SimpleNamespace(
        architecture_profile="legacy-profile",
        compatibility_level="legacy-compatibility",
    )

    class AdaptedConfig:
        profile_manifest = manifest

        def to_dict(self):
            return {
                "model_type": "qwen3_omni_prototype",
                "architecture_profile": "legacy_prototype",
            }

    config = AdaptedConfig()

    class SpyConfig:
        @classmethod
        def from_legacy_pretrained_config(cls, checkpoint):
            events.append("legacy-adapt")
            return config

    class SpyModel:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            events.append("weight-load")
            assert kwargs["config"] is config
            return cls()

    def gate(checkpoint, **kwargs):
        events.append("metadata-gate")
        assert checkpoint == "checkpoint"
        assert kwargs == {
            "expected_profile": "legacy-profile",
            "expected_compatibility": "legacy-compatibility",
            "expected_architecture": expected_architecture,
            "expected_tokenizer_sha256": expected_tokenizer_sha256,
            "legacy_config": config.to_dict(),
        }

    monkeypatch.setattr(trainer_thinker, "Qwen3OmniMoeConfig", SpyConfig)
    monkeypatch.setattr(
        trainer_thinker, "Qwen3OmniMoeThinkerTextModel", SpyModel
    )
    monkeypatch.setattr(
        trainer_thinker,
        "load_checkpoint_metadata",
        gate,
    )

    model = trainer_thinker._load_legacy_stage1_model(
        "checkpoint",
        expected_profile=manifest.architecture_profile,
        expected_compatibility=manifest.compatibility_level,
        expected_architecture=expected_architecture,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
    )

    assert isinstance(model, SpyModel)
    assert events == ["legacy-adapt", "metadata-gate", "weight-load"]


def test_inference_entry_calls_reconciled_loader(monkeypatch):
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    monkeypatch.setattr(
        cli,
        "_ensure_tokenizer",
        lambda tokenizer_name_or_path, checkpoint: tokenizer,
    )
    monkeypatch.setattr(
        cli,
        "_load_reconciled_stage2_model",
        lambda checkpoint, tokenizer, load_kwargs=None: (
            (_ for _ in ()).throw(WiringReached())
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli.Qwen3OmniMoeThinkerVisionAudioModel,
        "from_pretrained",
        classmethod(
            lambda cls, checkpoint, **kwargs: pytest.fail(
                "entry bypassed reconciled loader"
            )
        ),
    )
    args = Namespace(
        checkpoint="checkpoint",
        tokenizer_name_or_path=None,
        dtype="auto",
        max_seq_length=8,
        skip_bad_media=False,
    )

    with pytest.raises(WiringReached):
        cli.run_stage2(args)
