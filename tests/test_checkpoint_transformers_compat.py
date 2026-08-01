import importlib
import json

import pytest

from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)


def test_checkpoint_helpers_import_with_supported_transformers():
    checkpoint = importlib.import_module(
        "qwen3_omni_pretrain.training.checkpoint"
    )
    assert callable(checkpoint.load_sharded_checkpoint)


def test_legacy_config_loader_accepts_old_checkpoint_directory(tmp_path):
    old_config = {
        "model_type": "qwen3_omni_moe",
        "vocab_size": 32,
        "thinker_config": {"use_moe": False},
    }
    (tmp_path / "config.json").write_text(
        json.dumps(old_config),
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match="qwen3_omni_moe"):
        config = Qwen3OmniMoeConfig.from_legacy_pretrained_config(
            str(tmp_path)
        )

    assert config.model_type == "qwen3_omni_prototype"
    assert config.thinker_config.routing_kind == "dense"
