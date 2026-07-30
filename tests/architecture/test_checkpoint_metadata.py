from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from qwen3_omni_pretrain.architecture.checkpoint_metadata import (
    CheckpointMetadata,
    load_checkpoint_metadata,
    write_checkpoint_metadata,
)
from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from qwen3_omni_pretrain.architecture.summary import (
    ArchitectureSummary,
    LayerArchitecture,
)


def legacy_manifest() -> ProfileManifest:
    return ProfileManifest(
        architecture_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
        compatibility_level=CompatibilityLevel.LEGACY_PROTOTYPE,
        sources={},
        assumptions=("custom research architecture",),
        exact_official_checkpoint_compatible=False,
    )


def toy_architecture_summary() -> ArchitectureSummary:
    return ArchitectureSummary(
        profile="legacy_prototype",
        compatibility_level="legacy-prototype",
        model_type="qwen3_omni_prototype",
        tokenizer_vocab_size=32,
        embedding_vocab_size=32,
        total_parameters=24,
        active_parameters_per_token=16,
        routed_parameters=8,
        shared_parameters=4,
        dense_parameters=12,
        capabilities={"text": True},
        layers=(
            LayerArchitecture(
                index=0,
                attention_type="full-attention",
                cache_type="kv-cache",
                ffn_type="dense",
                routed_experts=0,
                experts_per_token=0,
            ),
        ),
        unsupported_capabilities=("audio",),
    )


def checkpoint_metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        manifest=legacy_manifest(),
        architecture=toy_architecture_summary(),
        tokenizer_sha256="a" * 64,
        implementation_commit="dc71e7ca1d03666798ecdbee5143e132e49210f7",
    )


def adapted_legacy_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_omni_prototype",
        "architecture_profile": "legacy_prototype",
        "thinker_config": {"use_moe": False, "routing_kind": "dense"},
    }


def test_checkpoint_metadata_round_trip_is_atomic(tmp_path):
    metadata = checkpoint_metadata()

    write_checkpoint_metadata(tmp_path, metadata)

    assert load_checkpoint_metadata(tmp_path) == metadata
    assert not (tmp_path / "architecture.json.tmp").exists()


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    [
        (
            lambda raw: raw.update({"schema_version": 2}),
            ValueError,
            "schema_version",
        ),
        (
            lambda raw: raw.update({"unknown": True}),
            ValueError,
            "checkpoint metadata keys",
        ),
        (
            lambda raw: raw.update({"tokenizer_sha256": "not-a-digest"}),
            ValueError,
            "tokenizer_sha256",
        ),
        (
            lambda raw: raw["architecture"].update({"profile": "other"}),
            ValueError,
            "architecture profile",
        ),
    ],
)
def test_checkpoint_metadata_rejects_malformed_or_contradictory_payload(
    tmp_path,
    mutation,
    exception,
    message,
):
    raw = checkpoint_metadata().to_dict()
    mutation(raw)
    (tmp_path / "architecture.json").write_text(
        json.dumps(raw),
        encoding="utf-8",
    )

    with pytest.raises(exception, match=message):
        load_checkpoint_metadata(tmp_path)


def test_checkpoint_metadata_restores_nested_contract_types(tmp_path):
    write_checkpoint_metadata(tmp_path, checkpoint_metadata())

    restored = load_checkpoint_metadata(tmp_path)

    assert restored is not None
    assert isinstance(restored.manifest, ProfileManifest)
    assert isinstance(restored.manifest.assumptions, tuple)
    assert isinstance(restored.architecture.layers, tuple)
    assert isinstance(restored.architecture.layers[0], LayerArchitecture)
    assert isinstance(restored.architecture.unsupported_capabilities, tuple)


def test_checkpoint_metadata_validates_expected_identity(tmp_path):
    write_checkpoint_metadata(tmp_path, checkpoint_metadata())

    with pytest.raises(ValueError, match="expected profile"):
        load_checkpoint_metadata(
            tmp_path,
            expected_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        )
    with pytest.raises(ValueError, match="expected compatibility"):
        load_checkpoint_metadata(
            tmp_path,
            expected_compatibility=CompatibilityLevel.STRUCTURE_ALIGNED,
        )
    different_architecture = toy_architecture_summary().to_dict()
    different_architecture["model_type"] = "different-model"
    with pytest.raises(ValueError, match="expected architecture"):
        load_checkpoint_metadata(
            tmp_path,
            expected_architecture=ArchitectureSummary.from_dict(
                different_architecture
            ),
        )
    with pytest.raises(ValueError, match="expected tokenizer"):
        load_checkpoint_metadata(
            tmp_path,
            expected_tokenizer_sha256="b" * 64,
        )


def test_nonlegacy_checkpoint_requires_metadata(tmp_path):
    with pytest.raises(ValueError, match="architecture.json"):
        load_checkpoint_metadata(
            tmp_path,
            expected_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        )


def test_missing_legacy_metadata_requires_adapted_config(tmp_path):
    with pytest.raises(ValueError, match="adapted legacy config"):
        load_checkpoint_metadata(
            tmp_path,
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
        )


def test_old_legacy_checkpoint_warns_returns_none_and_is_not_modified(tmp_path):
    with pytest.warns(DeprecationWarning, match="architecture.json") as warnings:
        restored = load_checkpoint_metadata(
            tmp_path,
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            legacy_config=adapted_legacy_config(),
        )

    assert restored is None
    assert len(warnings) == 1
    assert list(tmp_path.iterdir()) == []


def test_old_legacy_checkpoint_rejects_unadapted_or_contradictory_config(
    tmp_path,
):
    raw = adapted_legacy_config()
    raw["architecture_profile"] = "qwen3_omni_reference"

    with pytest.raises(ValueError, match="legacy prototype"):
        load_checkpoint_metadata(
            tmp_path,
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            legacy_config=raw,
        )


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def save_pretrained(self, directory, **kwargs) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")


class FakeAccelerator:
    is_main_process = True
    num_processes = 1

    def __init__(self) -> None:
        self.loaded = False

    def wait_for_everyone(self) -> None:
        pass

    def save_state(self, directory) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "accelerator-state").write_text("saved", encoding="utf-8")

    def load_state(self, directory) -> None:
        self.loaded = True

    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        return model.state_dict()


def _standard_save_arguments(tmp_path) -> dict[str, object]:
    return {
        "checkpoint_dir": str(tmp_path),
        "model": FakeModel(),
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "epoch": 1,
        "global_step": 2,
        "best_val_loss": 3.0,
        "metadata": checkpoint_metadata(),
    }


@pytest.mark.parametrize("entrypoint", ["atomic_save_checkpoint", "save_checkpoint"])
def test_standard_checkpoint_writers_persist_metadata(
    tmp_path,
    entrypoint,
):
    from qwen3_omni_pretrain.training import checkpoint

    arguments = _standard_save_arguments(tmp_path)
    if entrypoint == "atomic_save_checkpoint":
        arguments["verify"] = False
    getattr(checkpoint, entrypoint)(**arguments)

    assert load_checkpoint_metadata(tmp_path) == checkpoint_metadata()


def test_checkpoint_accelerator_writer_persists_metadata(tmp_path):
    from qwen3_omni_pretrain.training.checkpoint import (
        save_checkpoint_accelerator,
    )

    save_checkpoint_accelerator(
        FakeAccelerator(),
        str(tmp_path),
        epoch=1,
        global_step=2,
        best_val_loss=3.0,
        metadata=checkpoint_metadata(),
    )

    assert load_checkpoint_metadata(tmp_path) == checkpoint_metadata()


def test_model_only_accelerator_writer_persists_metadata(tmp_path):
    from qwen3_omni_pretrain.training.checkpoint import (
        save_model_only_accelerator,
    )

    save_model_only_accelerator(
        FakeAccelerator(),
        FakeModel(),
        str(tmp_path),
        metadata=checkpoint_metadata(),
    )

    assert load_checkpoint_metadata(tmp_path) == checkpoint_metadata()


def test_accelerator_utils_standard_writer_persists_metadata(tmp_path):
    from qwen3_omni_pretrain.training.accelerator_utils import (
        save_accelerator_checkpoint,
    )

    save_accelerator_checkpoint(
        FakeAccelerator(),
        str(tmp_path),
        epoch=1,
        global_step=2,
        best_val_loss=3.0,
        metadata=checkpoint_metadata(),
    )

    assert load_checkpoint_metadata(tmp_path) == checkpoint_metadata()


def test_accelerator_utils_tp_writer_persists_metadata(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import accelerator_utils

    monkeypatch.setattr(
        accelerator_utils,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        accelerator_utils,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    accelerator_utils.save_tp_sharded_checkpoint(
        FakeAccelerator(),
        FakeModel(),
        optimizer=None,
        scheduler=None,
        checkpoint_dir=str(tmp_path),
        epoch=1,
        global_step=2,
        best_val_loss=3.0,
        metadata=checkpoint_metadata(),
    )

    assert load_checkpoint_metadata(tmp_path) == checkpoint_metadata()


def test_standard_tensor_loader_rejects_identity_before_reading_weights(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import checkpoint

    torch.save({}, tmp_path / "pytorch_model.bin")
    monkeypatch.setattr(
        checkpoint.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("tensor read must be gated"),
    )

    with pytest.raises(ValueError, match="architecture.json"):
        checkpoint.load_checkpoint(
            str(tmp_path),
            FakeModel(),
            expected_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        )


def test_accelerator_tensor_loader_rejects_identity_before_loading_state(
    tmp_path,
):
    from qwen3_omni_pretrain.training.accelerator_utils import (
        load_accelerator_checkpoint,
    )

    accelerator = FakeAccelerator()
    with pytest.raises(ValueError, match="architecture.json"):
        load_accelerator_checkpoint(
            accelerator,
            str(tmp_path),
            expected_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        )

    assert accelerator.loaded is False


def test_checkpoint_module_accelerator_loader_gates_tensor_state(tmp_path):
    from qwen3_omni_pretrain.training.checkpoint import (
        load_checkpoint_accelerator,
    )

    accelerator = FakeAccelerator()
    with pytest.raises(ValueError, match="architecture.json"):
        load_checkpoint_accelerator(
            accelerator,
            str(tmp_path),
            expected_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        )

    assert accelerator.loaded is False


def test_deepspeed_atomic_writer_persists_metadata(tmp_path):
    from qwen3_omni_pretrain.training import trainer_thinker

    class FakeEngine:
        def save_checkpoint(self, directory, client_state):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model-state").write_text("saved", encoding="utf-8")

    class FakeScheduler:
        def state_dict(self):
            return {"step": 2}

    class FakeTokenizer:
        def save_pretrained(self, directory):
            (Path(directory) / "tokenizer.json").write_text(
                "{}",
                encoding="utf-8",
            )

    destination = tmp_path / "checkpoint"
    trainer_thinker._save_deepspeed_checkpoint_atomic(
        model_engine=FakeEngine(),
        checkpoint_dir=str(destination),
        client_state={"step": 2},
        scheduler=FakeScheduler(),
        tokenizer=FakeTokenizer(),
        metadata=checkpoint_metadata(),
        is_main=True,
    )

    assert load_checkpoint_metadata(destination) == checkpoint_metadata()
    assert not Path(f"{destination}.tmp").exists()


def test_trainer_builds_metadata_from_live_graph_and_tokenizer_evidence():
    from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeConfig,
    )
    from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
        Qwen3OmniMoeThinkerTextModel,
    )
    from qwen3_omni_pretrain.training import trainer_thinker

    config = Qwen3OmniMoeConfig(
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

    class EvidenceTokenizer:
        def get_vocab(self):
            return {"<eos>": 1, "hello": 0}

        def get_added_vocab(self):
            return {"<eos>": 1}

    metadata = trainer_thinker._build_checkpoint_metadata(
        Qwen3OmniMoeThinkerTextModel(config),
        EvidenceTokenizer(),
        implementation_commit="d" * 40,
    )

    assert metadata.manifest == config.profile_manifest
    assert metadata.architecture.layers[0].ffn_type == "dense"
    assert len(metadata.tokenizer_sha256) == 64


def test_trainer_refuses_to_invent_missing_tokenizer_evidence():
    from qwen3_omni_pretrain.training import trainer_thinker

    with pytest.raises(ValueError, match="tokenizer identity evidence"):
        trainer_thinker._tokenizer_identity_sha256(object())


def test_tokenizer_identity_includes_special_token_roles():
    from qwen3_omni_pretrain.training import trainer_thinker

    class EvidenceTokenizer:
        def __init__(self, eos_token_id):
            self.eos_token_id = eos_token_id

        def get_vocab(self):
            return {"first": 0, "second": 1}

        def get_added_vocab(self):
            return {}

    assert trainer_thinker._tokenizer_identity_sha256(
        EvidenceTokenizer(0)
    ) != trainer_thinker._tokenizer_identity_sha256(
        EvidenceTokenizer(1)
    )
