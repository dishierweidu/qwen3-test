from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


class SerializedBackend:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_str(self) -> str:
        return json.dumps(self.payload)


class EvidenceTokenizer:
    def __init__(
        self,
        *,
        backend_payload: dict[str, object] | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.backend_tokenizer = SerializedBackend(
            backend_payload
            or {
                "model": {
                    "type": "WordLevel",
                    "vocab": {"<eos>": 1, "hello": 0},
                },
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
            }
        )
        self.eos_token_id = eos_token_id

    def get_vocab(self):
        return {"<eos>": 1, "hello": 0}

    def get_added_vocab(self):
        return {"<eos>": 1}


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


def test_old_legacy_checkpoint_validates_only_already_adapted_identity(
    tmp_path,
):
    raw = adapted_legacy_config()
    raw["thinker_config"] = {
        "use_moe": "not-a-boolean",
        "moe_layer_indices": "0",
    }

    with pytest.warns(DeprecationWarning, match="architecture.json") as warnings:
        restored = load_checkpoint_metadata(
            tmp_path,
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            legacy_config=raw,
        )

    assert restored is None
    assert len(warnings) == 1


def test_old_legacy_checkpoint_rejects_config_objects(tmp_path):
    with pytest.raises(TypeError, match="mapping"):
        load_checkpoint_metadata(
            tmp_path,
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            legacy_config=SimpleNamespace(
                model_type="qwen3_omni_prototype",
                architecture_profile="legacy_prototype",
                profile_manifest=legacy_manifest(),
            ),
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


class RecordingAccelerator(FakeAccelerator):
    def __init__(
        self,
        *,
        is_main_process: bool,
        events: list[str],
        peer_failed: bool = False,
        peer_failure_on_status: int | None = None,
    ) -> None:
        super().__init__()
        self.is_main_process = is_main_process
        self.num_processes = 2
        self.device = torch.device("cpu")
        self.events = events
        self.peer_failed = peer_failed
        self.peer_failure_on_status = peer_failure_on_status
        self.status_calls = 0

    def wait_for_everyone(self) -> None:
        self.events.append("barrier")

    def save_state(self, directory) -> None:
        self.events.append("payload")
        super().save_state(directory)

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        self.status_calls += 1
        self.events.append(
            f"status:{int(tensor.item())}"
        )
        failed = bool(tensor.item()) or (
            self.peer_failed
            and (
                self.peer_failure_on_status is None
                or self.status_calls == self.peer_failure_on_status
            )
        )
        return torch.tensor(
            [int(failed)],
            dtype=tensor.dtype,
            device=tensor.device,
        )


class RecordingTokenizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def save_pretrained(self, directory) -> None:
        self.events.append("tokenizer")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


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


def test_checkpoint_accelerator_synchronizes_after_payload_before_publication(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import checkpoint

    events: list[str] = []
    accelerator = RecordingAccelerator(
        is_main_process=True,
        events=events,
    )
    real_write = checkpoint.write_checkpoint_metadata

    def record_metadata(directory, metadata):
        events.append("metadata")
        real_write(directory, metadata)

    monkeypatch.setattr(
        checkpoint,
        "write_checkpoint_metadata",
        record_metadata,
    )

    checkpoint.save_checkpoint_accelerator(
        accelerator,
        str(tmp_path / "checkpoint"),
        epoch=1,
        global_step=2,
        best_val_loss=3.0,
        tokenizer=RecordingTokenizer(events),
        metadata=checkpoint_metadata(),
    )

    payload_index = events.index("payload")
    tokenizer_index = events.index("tokenizer")
    metadata_index = events.index("metadata")
    assert "barrier" in events[tokenizer_index + 1 : metadata_index]
    assert payload_index < tokenizer_index < metadata_index


def test_checkpoint_accelerator_propagates_peer_publication_failure(
    tmp_path,
):
    from qwen3_omni_pretrain.training import checkpoint

    accelerator = RecordingAccelerator(
        is_main_process=False,
        events=[],
        peer_failed=True,
        peer_failure_on_status=3,
    )

    with pytest.raises(RuntimeError, match="another rank"):
        checkpoint.save_checkpoint_accelerator(
            accelerator,
            str(tmp_path / "checkpoint"),
            epoch=1,
            global_step=2,
            best_val_loss=3.0,
            metadata=checkpoint_metadata(),
        )


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


def test_model_only_accelerator_synchronizes_payload_before_metadata(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import checkpoint

    events: list[str] = []
    accelerator = RecordingAccelerator(
        is_main_process=True,
        events=events,
    )

    class RecordingModel(FakeModel):
        def save_pretrained(self, directory, **kwargs) -> None:
            events.append("model")
            super().save_pretrained(directory, **kwargs)

    real_write = checkpoint.write_checkpoint_metadata

    def record_metadata(directory, metadata):
        events.append("metadata")
        real_write(directory, metadata)

    monkeypatch.setattr(
        checkpoint,
        "write_checkpoint_metadata",
        record_metadata,
    )

    checkpoint.save_model_only_accelerator(
        accelerator,
        RecordingModel(),
        str(tmp_path / "model"),
        metadata=checkpoint_metadata(),
    )

    model_index = events.index("model")
    metadata_index = events.index("metadata")
    assert "barrier" in events[model_index + 1 : metadata_index]


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


def test_accelerator_utils_writer_synchronizes_payload_before_metadata(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import accelerator_utils

    events: list[str] = []
    accelerator = RecordingAccelerator(
        is_main_process=True,
        events=events,
    )
    real_write = accelerator_utils.write_checkpoint_metadata

    def record_metadata(directory, metadata):
        events.append("metadata")
        real_write(directory, metadata)

    monkeypatch.setattr(
        accelerator_utils,
        "write_checkpoint_metadata",
        record_metadata,
    )

    accelerator_utils.save_accelerator_checkpoint(
        accelerator,
        str(tmp_path / "checkpoint"),
        epoch=1,
        global_step=2,
        best_val_loss=3.0,
        tokenizer=RecordingTokenizer(events),
        metadata=checkpoint_metadata(),
    )

    tokenizer_index = events.index("tokenizer")
    metadata_index = events.index("metadata")
    assert "barrier" in events[tokenizer_index + 1 : metadata_index]


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


def test_accelerator_utils_tp_non_main_rank_never_publishes_metadata(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import accelerator_utils

    events: list[str] = []
    accelerator = RecordingAccelerator(
        is_main_process=False,
        events=events,
    )
    monkeypatch.setattr(
        accelerator_utils,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        accelerator_utils,
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )
    monkeypatch.setattr(
        accelerator_utils,
        "write_checkpoint_metadata",
        lambda *args, **kwargs: pytest.fail(
            "non-main rank must not publish metadata"
        ),
    )

    accelerator_utils.save_tp_sharded_checkpoint(
        accelerator,
        FakeModel(),
        optimizer=None,
        scheduler=None,
        checkpoint_dir=str(tmp_path / "checkpoint"),
        epoch=1,
        global_step=2,
        best_val_loss=3.0,
        metadata=checkpoint_metadata(),
    )

    assert "barrier" in events
    assert any(event.startswith("status:") for event in events)


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


def test_standard_loader_uses_valid_backup_when_primary_metadata_is_invalid(
    tmp_path,
):
    from qwen3_omni_pretrain.training import checkpoint

    primary = tmp_path / "checkpoint"
    backup = tmp_path / "checkpoint.backup"
    primary.mkdir()
    backup.mkdir()
    (primary / "architecture.json").write_text("{}", encoding="utf-8")
    write_checkpoint_metadata(backup, checkpoint_metadata())
    torch.save(FakeModel().state_dict(), backup / "pytorch_model.bin")
    torch.save(
        {
            "epoch": 7,
            "global_step": 11,
            "best_val_loss": 0.5,
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
        },
        backup / "trainer_state.pt",
    )

    with pytest.warns(UserWarning, match="Trying backup"):
        restored = checkpoint.load_checkpoint(
            str(primary),
            FakeModel(),
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            expected_compatibility=CompatibilityLevel.LEGACY_PROTOTYPE,
            expected_architecture=toy_architecture_summary(),
            expected_tokenizer_sha256="a" * 64,
        )

    assert restored == (7, 11, 0.5)


def test_accelerator_loader_uses_valid_backup_when_primary_metadata_is_invalid(
    tmp_path,
):
    from qwen3_omni_pretrain.training.checkpoint import (
        load_checkpoint_accelerator,
    )

    primary = tmp_path / "checkpoint"
    backup = tmp_path / "checkpoint.backup"
    primary.mkdir()
    backup.mkdir()
    (primary / "architecture.json").write_text("{}", encoding="utf-8")
    write_checkpoint_metadata(backup, checkpoint_metadata())
    torch.save(
        {"epoch": 7, "global_step": 11, "best_val_loss": 0.5},
        backup / "trainer_state.pt",
    )
    accelerator = FakeAccelerator()
    loaded_paths: list[str] = []

    def load_state(directory):
        loaded_paths.append(str(directory))
        accelerator.loaded = True

    accelerator.load_state = load_state

    with pytest.warns(UserWarning, match="Trying backup"):
        restored = load_checkpoint_accelerator(
            accelerator,
            str(primary),
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            expected_compatibility=CompatibilityLevel.LEGACY_PROTOTYPE,
            expected_architecture=toy_architecture_summary(),
            expected_tokenizer_sha256="a" * 64,
        )

    assert restored == (7, 11, 0.5)
    assert loaded_paths == [str(backup)]


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


def test_deepspeed_writer_synchronizes_payload_before_publication(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import trainer_thinker

    events: list[str] = []

    class FakeEngine:
        def save_checkpoint(self, directory, client_state):
            events.append("payload")
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model-state").write_text("saved", encoding="utf-8")

    class FakeScheduler:
        def state_dict(self):
            return {"step": 2}

    class FakeTokenizer:
        def save_pretrained(self, directory):
            events.append("tokenizer")

    real_write = trainer_thinker.write_checkpoint_metadata

    def record_metadata(directory, metadata):
        events.append("metadata")
        real_write(directory, metadata)

    def gather(output, value):
        events.append("status")
        for index in range(len(output)):
            output[index] = value

    monkeypatch.setattr(trainer_thinker.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_thinker.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_thinker.dist,
        "barrier",
        lambda: events.append("barrier"),
    )
    monkeypatch.setattr(
        trainer_thinker.dist,
        "all_gather_object",
        gather,
    )
    monkeypatch.setattr(
        trainer_thinker,
        "write_checkpoint_metadata",
        record_metadata,
    )

    trainer_thinker._save_deepspeed_checkpoint_atomic(
        model_engine=FakeEngine(),
        checkpoint_dir=str(tmp_path / "checkpoint"),
        client_state={"step": 2},
        scheduler=FakeScheduler(),
        tokenizer=FakeTokenizer(),
        metadata=checkpoint_metadata(),
        is_main=True,
    )

    payload_index = events.index("payload")
    tokenizer_index = events.index("tokenizer")
    assert "barrier" in events[payload_index + 1 : tokenizer_index]


def test_deepspeed_writer_cleans_temp_and_restores_backup_on_publish_failure(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import trainer_thinker

    destination = tmp_path / "checkpoint"
    destination.mkdir()
    (destination / "previous").write_text("kept", encoding="utf-8")

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
            pass

    real_rename = trainer_thinker.os.rename

    def fail_publication(source, target):
        if (
            Path(source) == Path(f"{destination}.tmp")
            and Path(target) == destination
        ):
            raise OSError("injected publication failure")
        return real_rename(source, target)

    monkeypatch.setattr(trainer_thinker.os, "rename", fail_publication)

    with pytest.raises(RuntimeError, match="publication"):
        trainer_thinker._save_deepspeed_checkpoint_atomic(
            model_engine=FakeEngine(),
            checkpoint_dir=str(destination),
            client_state={"step": 2},
            scheduler=FakeScheduler(),
            tokenizer=FakeTokenizer(),
            metadata=checkpoint_metadata(),
            is_main=True,
        )

    assert (destination / "previous").read_text(encoding="utf-8") == "kept"
    assert not Path(f"{destination}.tmp").exists()


def test_deepspeed_writer_propagates_failure_from_another_rank(
    tmp_path,
    monkeypatch,
):
    from qwen3_omni_pretrain.training import trainer_thinker

    class FakeEngine:
        def save_checkpoint(self, directory, client_state):
            Path(directory).mkdir(parents=True, exist_ok=True)

    gather_calls = 0

    def gather(output, value):
        nonlocal gather_calls
        gather_calls += 1
        if gather_calls == 3:
            output[:] = [None, "publication failed on rank 1"]
        else:
            output[:] = [None, None]

    monkeypatch.setattr(trainer_thinker.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_thinker.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(trainer_thinker.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        trainer_thinker.dist,
        "all_gather_object",
        gather,
    )

    with pytest.raises(RuntimeError, match="another rank"):
        trainer_thinker._save_deepspeed_checkpoint_atomic(
            model_engine=FakeEngine(),
            checkpoint_dir=str(tmp_path / "checkpoint"),
            client_state={"step": 2},
            scheduler=object(),
            tokenizer=object(),
            metadata=checkpoint_metadata(),
            is_main=False,
        )

    assert gather_calls == 3


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


def test_trainer_refuses_vocab_only_tokenizer_identity():
    from qwen3_omni_pretrain.training import trainer_thinker

    class VocabOnlyTokenizer:
        def get_vocab(self):
            return {"first": 0, "second": 1}

    with pytest.raises(ValueError, match="full serialization"):
        trainer_thinker._tokenizer_identity_sha256(VocabOnlyTokenizer())


def test_tokenizer_identity_can_use_complete_saved_artifacts():
    from qwen3_omni_pretrain.training import trainer_thinker

    class ArtifactTokenizer:
        def save_pretrained(self, directory):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "WordLevel",
                            "vocab": {"first": 0, "second": 1},
                        },
                        "pre_tokenizer": {"type": "Whitespace"},
                    }
                ),
                encoding="utf-8",
            )

    assert len(
        trainer_thinker._tokenizer_identity_sha256(ArtifactTokenizer())
    ) == 64


def test_tokenizer_identity_includes_special_token_roles():
    from qwen3_omni_pretrain.training import trainer_thinker

    assert trainer_thinker._tokenizer_identity_sha256(
        EvidenceTokenizer(eos_token_id=0)
    ) != trainer_thinker._tokenizer_identity_sha256(
        EvidenceTokenizer(eos_token_id=1)
    )


def test_tokenizer_identity_distinguishes_same_vocab_different_tokenization():
    from qwen3_omni_pretrain.training import trainer_thinker

    whitespace = EvidenceTokenizer(
        backend_payload={
            "model": {
                "type": "WordLevel",
                "vocab": {"first": 0, "second": 1},
            },
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
        }
    )
    byte_level = EvidenceTokenizer(
        backend_payload={
            "model": {
                "type": "WordLevel",
                "vocab": {"first": 0, "second": 1},
            },
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "ByteLevel"},
        }
    )

    assert trainer_thinker._tokenizer_identity_sha256(
        whitespace
    ) != trainer_thinker._tokenizer_identity_sha256(byte_level)


def test_metadata_counts_zero3_placeholder_parameters_with_ds_numel():
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
    model = Qwen3OmniMoeThinkerTextModel(config)
    expected = trainer_thinker._build_checkpoint_metadata(
        model,
        EvidenceTokenizer(),
        implementation_commit="d" * 40,
    ).architecture
    for parameter in model.parameters():
        parameter.ds_numel = parameter.numel()
        parameter.data = torch.empty(
            0,
            dtype=parameter.dtype,
            device=parameter.device,
        )

    actual = trainer_thinker._build_checkpoint_metadata(
        model,
        EvidenceTokenizer(),
        implementation_commit="d" * 40,
    ).architecture

    assert actual == expected
