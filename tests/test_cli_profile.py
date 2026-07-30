from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from qwen3_omni_pretrain import cli_profile
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.pins import (
    QWEN3_OMNI_MODEL_ID,
)


def _write_tiny_legacy_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "model_type": "qwen3_omni_prototype",
                "architecture_profile": "legacy_prototype",
                "vocab_size": 32,
                "thinker_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 16,
                    "use_moe": False,
                    "routing_kind": "dense",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_validate_prints_manifest_without_allocating_model(
    tmp_path,
    monkeypatch,
    capsys,
):
    from qwen3_omni_pretrain.profiles.legacy_prototype import (
        factory as factory_module,
    )

    config_path = _write_tiny_legacy_config(tmp_path / "legacy.yaml")
    monkeypatch.setattr(
        factory_module,
        "Qwen3OmniMoeThinkerTextModel",
        lambda *args, **kwargs: pytest.fail("model was allocated"),
    )

    result = cli_profile.main(
        [
            "validate",
            "--profile",
            "legacy_prototype",
            "--config-or-checkpoint",
            str(config_path),
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["architecture_profile"] == "legacy_prototype"
    assert output["compatibility_level"] == "legacy-prototype"
    assert output["exact_official_checkpoint_compatible"] is False


def test_validate_subcommand_invokes_factory_validation(
    tmp_path,
    monkeypatch,
    capsys,
):
    from qwen3_omni_pretrain.profiles.legacy_prototype import (
        factory as factory_module,
    )

    config_path = _write_tiny_legacy_config(tmp_path / "legacy.yaml")
    calls = []
    real_validate = factory_module.LegacyPrototypeFactory.validate

    def record_validate(self, request):
        calls.append(request)
        return real_validate(self, request)

    monkeypatch.setattr(
        factory_module.LegacyPrototypeFactory,
        "validate",
        record_validate,
    )

    result = cli_profile.main(
        [
            "validate",
            "--profile",
            "legacy_prototype",
            "--config-or-checkpoint",
            str(config_path),
            "--json",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0].config_or_checkpoint == str(config_path)
    json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--capability", "streaming_generation"),
        ("--tokenizer", "unused-tokenizer"),
        ("--allow-network", None),
    ],
)
def test_validate_rejects_inspection_only_options(
    tmp_path,
    capsys,
    option,
    value,
):
    config_path = _write_tiny_legacy_config(tmp_path / "legacy.yaml")
    extra_arguments = [option]
    if value is not None:
        extra_arguments.append(value)

    with pytest.raises(SystemExit) as raised:
        cli_profile.main(
            [
                "validate",
                "--profile",
                "legacy_prototype",
                "--config-or-checkpoint",
                str(config_path),
                *extra_arguments,
            ]
        )

    assert raised.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_inspect_builds_on_meta_and_reports_unsupported_capability(
    tmp_path,
    capsys,
):
    config_path = _write_tiny_legacy_config(tmp_path / "legacy.yaml")

    result = cli_profile.main(
        [
            "inspect",
            "--profile",
            "legacy_prototype",
            "--config-or-checkpoint",
            str(config_path),
            "--capability",
            "streaming_generation",
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    summary = output["architecture_summary"]
    assert result == 0
    assert output["manifest"]["compatibility_level"] == (
        "legacy-prototype"
    )
    assert output["artifact_type"] == "Qwen3OmniMoeThinkerTextModel"
    assert summary["model_type"] == "qwen3_omni_prototype"
    assert summary["total_parameters"] > 0
    assert summary["unsupported_capabilities"] == [
        "streaming_generation"
    ]


def test_reference_inspection_is_offline_and_honest(
    monkeypatch,
    capsys,
):
    from qwen3_omni_pretrain.profiles.qwen3_omni_reference import (
        factory as factory_module,
    )

    monkeypatch.setattr(
        factory_module,
        "load_reference_config",
        lambda *args, **kwargs: pytest.fail("config loader was invoked"),
    )
    monkeypatch.setattr(
        factory_module,
        "load_reference_processor",
        lambda *args, **kwargs: pytest.fail("processor loader was invoked"),
    )

    result = cli_profile.main(
        [
            "inspect",
            "--profile",
            "qwen3_omni_reference",
            "--config-or-checkpoint",
            QWEN3_OMNI_MODEL_ID,
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["artifact_type"] == "Qwen3OracleArtifact"
    assert output["manifest"]["compatibility_level"] == (
        "structure-aligned"
    )
    assert (
        output["manifest"]["exact_official_checkpoint_compatible"]
        is False
    )
    assert (
        output["architecture_summary"]["capabilities"][
            "inference_runtime"
        ]
        is False
    )
    assert "inference_runtime" in output[
        "architecture_summary"
    ]["unsupported_capabilities"]


def test_checked_in_legacy_inspect_example_executes_offline():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "qwen3_omni_pretrain.cli_profile",
            "inspect",
            "--profile",
            "legacy_prototype",
            "--config-or-checkpoint",
            "configs/model/qwen3_omni_1_3b_moe.yaml",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["manifest"]["architecture_profile"] == (
        "legacy_prototype"
    )
    assert output["architecture_summary"]["embedding_vocab_size"] == (
        152_064
    )


def test_cli_rejects_planned_profile_without_factory(capsys):
    with pytest.raises(SystemExit) as raised:
        cli_profile.main(
            [
                "validate",
                "--profile",
                "qwen35_omni_inspired",
                "--config-or-checkpoint",
                "planned",
            ]
        )

    assert raised.value.code == 2
    assert "is not registered" in capsys.readouterr().err
