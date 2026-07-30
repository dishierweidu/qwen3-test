from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from qwen3_omni_pretrain.utils.model_stats import collect_parameter_stats
from scripts import inspect_architecture as inspection


class _Tokenizer:
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


def _write_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "model_type": "qwen3_omni_prototype",
                "architecture_profile": "legacy_prototype",
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
    return path


@pytest.mark.parametrize(
    ("allow_network", "allow_remote_code", "expected"),
    [
        (
            False,
            False,
            {"local_files_only": True, "trust_remote_code": False},
        ),
        (
            True,
            False,
            {"local_files_only": False, "trust_remote_code": False},
        ),
        (
            False,
            True,
            {"local_files_only": True, "trust_remote_code": True},
        ),
    ],
)
def test_inspection_tokenizer_network_and_remote_code_are_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_network: bool,
    allow_remote_code: bool,
    expected: dict[str, bool],
):
    calls: list[dict[str, object]] = []

    def load_tokenizer(source, **kwargs):
        calls.append(kwargs)
        return _Tokenizer()

    monkeypatch.setattr(
        inspection.AutoTokenizer,
        "from_pretrained",
        load_tokenizer,
    )

    summary = inspection.inspect_architecture(
        _write_config(tmp_path / "model.yaml"),
        tokenizer_path=Path("tokenizer"),
        allow_network=allow_network,
        allow_remote_code=allow_remote_code,
    )

    assert calls == [expected]
    assert summary.tokenizer_vocab_size == 6
    assert summary.embedding_vocab_size == 8


@pytest.mark.parametrize(
    "script",
    [
        "scripts/inspect_architecture.py",
        "scripts/inspect_model_parameters.py",
    ],
)
def test_inspection_script_help_exposes_separate_safety_flags(script: str):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [str(root / script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--tokenizer" in completed.stdout
    assert "--allow-network" in completed.stdout
    assert "--allow-remote-code" in completed.stdout


def test_inspection_json_and_parameter_wrapper_execute_and_agree(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    config_path = _write_config(tmp_path / "model.yaml")
    architecture = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_architecture.py",
            str(config_path),
            "--json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    parameters = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_model_parameters.py",
            str(config_path),
            "--json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert architecture.returncode == 0, architecture.stderr
    assert parameters.returncode == 0, parameters.stderr
    full = json.loads(architecture.stdout)
    wrapped = json.loads(parameters.stdout)
    assert wrapped == {
        key: full[key]
        for key in (
            "total_parameters",
            "active_parameters_per_token",
            "routed_parameters",
            "shared_parameters",
            "dense_parameters",
        )
    }


def test_parameter_stats_deduplicate_tied_embedding_and_head():
    class TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 3)
            self.lm_head = torch.nn.Linear(3, 5, bias=False)
            self.lm_head.weight = self.embedding.weight

    stats = collect_parameter_stats(TiedModel())

    assert stats.total_parameters == 15
    assert stats.trainable_parameters == 15
    assert stats.dense_parameters == 15


def test_parameter_stats_deduplicate_shared_module_references():
    class SharedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared_mlp = torch.nn.Linear(3, 3, bias=False)
            self.second_reference = self.shared_mlp

    stats = collect_parameter_stats(SharedModel())

    assert stats.total_parameters == 9
    assert stats.shared_parameters == 9
    assert stats.dense_parameters == 0
