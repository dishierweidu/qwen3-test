from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from qwen3_omni_pretrain import cli_train_thinker as cli
from qwen3_omni_pretrain.training import distributed as distributed_module
from qwen3_omni_pretrain.training.stage2_config import (
    Stage2RuntimeConfig,
)


def cli_args(**updates):
    values = {
        "config": "stage2.yaml",
        "tokenizer_name_or_path": "tokenizer",
        "stage": "stage2",
        "tensorboard": False,
        "log_dir": "./runs",
        "resume_from_checkpoint": "cli-checkpoint",
        "local_rank": None,
        "deepspeed": None,
        "accelerator_config": None,
        "use_accelerator": False,
        "use_tensor_parallel": False,
        "tp_size": 1,
        "pp_size": 1,
    }
    values.update(updates)
    return Namespace(**values)


def checked_stage2_config():
    return yaml.safe_load(
        Path(
            "configs/train/stage2_omni_vision_audio.yaml"
        ).read_text(encoding="utf-8")
    )


def test_stage2_cli_rejects_ddp_before_any_distributed_context(
    monkeypatch,
):
    raw = checked_stage2_config()
    raw["train"]["ddp"] = True
    monkeypatch.setattr(cli, "parse_args", cli_args)
    monkeypatch.setattr(cli, "load_yaml", lambda path: raw)
    monkeypatch.setattr(
        cli,
        "distributed_context",
        lambda: pytest.fail("distributed context entered"),
    )
    monkeypatch.setattr(
        distributed_module,
        "distributed_tensor_parallel_context",
        lambda **kwargs: pytest.fail("TP context entered"),
    )
    monkeypatch.setattr(
        cli,
        "train_thinker_stage2",
        lambda **kwargs: pytest.fail("trainer entered"),
    )

    with pytest.raises(NotImplementedError, match="Stage2 ddp=true"):
        cli.main()


def test_stage2_cli_bypasses_context_and_passes_one_runtime(
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(cli, "parse_args", cli_args)
    monkeypatch.setattr(
        cli, "load_yaml", lambda path: checked_stage2_config()
    )
    monkeypatch.setattr(
        cli,
        "distributed_context",
        lambda: pytest.fail("distributed context entered"),
    )
    monkeypatch.setattr(
        cli,
        "train_thinker_stage2",
        lambda **kwargs: captured.update(kwargs),
    )

    cli.main()

    assert isinstance(captured["cfg"], Stage2RuntimeConfig)
    assert captured["cfg"].resume_from_checkpoint == "cli-checkpoint"
    assert captured["resume_from_checkpoint"] is None


@pytest.mark.parametrize(
    "updates",
    [
        {"use_accelerator": True},
        {"accelerator_config": "accelerate.yaml"},
        {"deepspeed": "deepspeed.json"},
        {"use_tensor_parallel": True, "tp_size": 2},
        {"pp_size": 2},
        {"local_rank": 0},
    ],
)
def test_stage2_cli_rejects_unsupported_execution_modes(
    monkeypatch, updates
):
    monkeypatch.setattr(
        cli, "parse_args", lambda: cli_args(**updates)
    )
    monkeypatch.setattr(
        cli, "load_yaml", lambda path: checked_stage2_config()
    )
    monkeypatch.setattr(
        cli,
        "distributed_context",
        lambda: pytest.fail("distributed context entered"),
    )
    monkeypatch.setattr(
        distributed_module,
        "distributed_tensor_parallel_context",
        lambda **kwargs: pytest.fail("TP context entered"),
    )
    monkeypatch.setattr(
        cli,
        "train_thinker_stage2",
        lambda **kwargs: pytest.fail("trainer entered"),
    )

    with pytest.raises(
        NotImplementedError,
        match="Stage2 distributed execution",
    ):
        cli.main()


def test_stage2_cli_rejects_launcher_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(cli, "parse_args", cli_args)
    monkeypatch.setattr(
        cli, "load_yaml", lambda path: checked_stage2_config()
    )
    monkeypatch.setattr(
        cli,
        "distributed_context",
        lambda: pytest.fail("distributed context entered"),
    )
    monkeypatch.setattr(
        distributed_module,
        "distributed_tensor_parallel_context",
        lambda **kwargs: pytest.fail("TP context entered"),
    )
    monkeypatch.setattr(
        cli,
        "train_thinker_stage2",
        lambda **kwargs: pytest.fail("trainer entered"),
    )

    with pytest.raises(
        NotImplementedError,
        match="Stage2 distributed execution",
    ):
        cli.main()
