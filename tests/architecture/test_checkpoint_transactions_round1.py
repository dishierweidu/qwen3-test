from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from qwen3_omni_pretrain.architecture.checkpoint_metadata import (
    load_checkpoint_metadata,
    write_checkpoint_metadata,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from tests.architecture.test_checkpoint_metadata import (
    FakeAccelerator,
    FakeModel,
    checkpoint_metadata,
    toy_architecture_summary,
)


class _FailingSaveAccelerator(FakeAccelerator):
    def save_state(self, directory) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "new-payload").write_text("partial", encoding="utf-8")
        raise OSError("injected payload failure")


def _seed_old_generation(destination: Path) -> None:
    destination.mkdir(parents=True)
    (destination / "old-payload").write_text("stable", encoding="utf-8")
    write_checkpoint_metadata(destination, checkpoint_metadata())


def _fail_final_publication_rename(
    monkeypatch: pytest.MonkeyPatch,
    module,
    destination: Path,
) -> None:
    real_rename = module.os.rename
    temporary = Path(f"{destination}.tmp")

    def rename(source, target):
        if Path(source) == temporary and Path(target) == destination:
            raise OSError("injected final rename failure")
        return real_rename(source, target)

    monkeypatch.setattr(module.os, "rename", rename)


def test_accelerator_overwrite_failure_keeps_old_generation_intact(
    tmp_path: Path,
):
    from qwen3_omni_pretrain.training import accelerator_utils

    destination = tmp_path / "checkpoint"
    _seed_old_generation(destination)

    with pytest.raises(RuntimeError, match="payload"):
        accelerator_utils.save_accelerator_checkpoint(
            _FailingSaveAccelerator(),
            str(destination),
            epoch=1,
            global_step=2,
            best_val_loss=3.0,
            metadata=checkpoint_metadata(),
        )

    assert (destination / "old-payload").read_text(
        encoding="utf-8"
    ) == "stable"
    assert not (destination / "new-payload").exists()
    assert load_checkpoint_metadata(destination) == checkpoint_metadata()
    assert not Path(f"{destination}.tmp").exists()


@pytest.mark.parametrize(
    "save_kind",
    ["standard", "accelerator", "model-only", "tp"],
)
def test_public_savers_preserve_only_backup_on_final_rename_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save_kind: str,
):
    from qwen3_omni_pretrain.training import accelerator_utils, checkpoint

    destination = tmp_path / "checkpoint"
    backup = Path(f"{destination}.backup")
    _seed_old_generation(backup)
    _fail_final_publication_rename(
        monkeypatch,
        checkpoint,
        destination,
    )
    monkeypatch.setattr(
        accelerator_utils,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        accelerator_utils,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    model = FakeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="rename|publication"):
        if save_kind == "standard":
            checkpoint.atomic_save_checkpoint(
                checkpoint_dir=str(destination),
                model=model,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                epoch=1,
                global_step=2,
                best_val_loss=3.0,
                metadata=checkpoint_metadata(),
            )
        elif save_kind == "accelerator":
            accelerator_utils.save_accelerator_checkpoint(
                FakeAccelerator(),
                str(destination),
                epoch=1,
                global_step=2,
                best_val_loss=3.0,
                metadata=checkpoint_metadata(),
            )
        elif save_kind == "model-only":
            checkpoint.save_model_only_accelerator(
                FakeAccelerator(),
                model,
                str(destination),
                metadata=checkpoint_metadata(),
            )
        else:
            accelerator_utils.save_tp_sharded_checkpoint(
                FakeAccelerator(),
                model,
                optimizer=optimizer,
                scheduler=None,
                checkpoint_dir=str(destination),
                epoch=1,
                global_step=2,
                best_val_loss=3.0,
                metadata=checkpoint_metadata(),
            )

    assert (destination / "old-payload").read_text(
        encoding="utf-8"
    ) == "stable"
    assert load_checkpoint_metadata(destination) == checkpoint_metadata()
    assert not Path(f"{destination}.tmp").exists()


def test_model_only_publication_failure_rolls_back_old_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from qwen3_omni_pretrain.training import checkpoint

    destination = tmp_path / "model"
    _seed_old_generation(destination)
    old_sidecar = (destination / "architecture.json").read_bytes()
    monkeypatch.setattr(
        checkpoint,
        "write_checkpoint_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("injected sidecar failure")
        ),
    )

    with pytest.raises(RuntimeError, match="publication"):
        checkpoint.save_model_only_accelerator(
            FakeAccelerator(),
            FakeModel(),
            str(destination),
            metadata=checkpoint_metadata(),
        )

    assert (destination / "old-payload").read_text(
        encoding="utf-8"
    ) == "stable"
    assert not (destination / "pytorch_model.bin").exists()
    assert (destination / "architecture.json").read_bytes() == old_sidecar


def test_deepspeed_publication_preserves_only_backup_on_final_rename_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from qwen3_omni_pretrain.training import trainer_thinker

    destination = tmp_path / "checkpoint"
    backup = Path(f"{destination}.backup")
    _seed_old_generation(backup)
    _fail_final_publication_rename(
        monkeypatch,
        trainer_thinker,
        destination,
    )

    class _Engine:
        def save_checkpoint(self, directory, client_state):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "new-payload").write_text(
                "new",
                encoding="utf-8",
            )

    class _Scheduler:
        def state_dict(self):
            return {"step": 2}

    class _Tokenizer:
        def save_pretrained(self, directory):
            (Path(directory) / "tokenizer.json").write_text(
                "{}",
                encoding="utf-8",
            )

    with pytest.raises(RuntimeError, match="publication"):
        trainer_thinker._save_deepspeed_checkpoint_atomic(
            model_engine=_Engine(),
            checkpoint_dir=str(destination),
            client_state={"step": 2},
            scheduler=_Scheduler(),
            tokenizer=_Tokenizer(),
            metadata=checkpoint_metadata(),
            is_main=True,
        )

    assert (destination / "old-payload").read_text(
        encoding="utf-8"
    ) == "stable"
    assert load_checkpoint_metadata(destination) == checkpoint_metadata()
    assert not Path(f"{destination}.tmp").exists()


def test_accelerator_utils_loader_recovers_valid_backup(
    tmp_path: Path,
):
    from qwen3_omni_pretrain.training.accelerator_utils import (
        load_accelerator_checkpoint,
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
    accelerator.load_state = lambda directory: loaded_paths.append(
        str(directory)
    )

    with pytest.warns(UserWarning, match="Trying backup"):
        restored = load_accelerator_checkpoint(
            accelerator,
            str(primary),
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            expected_compatibility=CompatibilityLevel.LEGACY_PROTOTYPE,
            expected_architecture=toy_architecture_summary(),
            expected_tokenizer_sha256="a" * 64,
        )

    assert restored == (7, 11, 0.5)
    assert loaded_paths == [str(backup)]


def test_standard_loader_recovers_backup_after_primary_tensor_eof(
    tmp_path: Path,
):
    from qwen3_omni_pretrain.training import checkpoint

    primary = tmp_path / "checkpoint"
    backup = tmp_path / "checkpoint.backup"
    primary.mkdir()
    backup.mkdir()
    write_checkpoint_metadata(primary, checkpoint_metadata())
    write_checkpoint_metadata(backup, checkpoint_metadata())
    (primary / "pytorch_model.bin").write_bytes(b"")
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


def test_accelerator_loader_recovers_backup_after_primary_state_eof(
    tmp_path: Path,
):
    from qwen3_omni_pretrain.training.accelerator_utils import (
        load_accelerator_checkpoint,
    )

    primary = tmp_path / "checkpoint"
    backup = tmp_path / "checkpoint.backup"
    primary.mkdir()
    backup.mkdir()
    write_checkpoint_metadata(primary, checkpoint_metadata())
    write_checkpoint_metadata(backup, checkpoint_metadata())
    (primary / "trainer_state.pt").write_bytes(b"")
    torch.save(
        {"epoch": 7, "global_step": 11, "best_val_loss": 0.5},
        backup / "trainer_state.pt",
    )
    accelerator = FakeAccelerator()
    loaded_paths: list[str] = []
    accelerator.load_state = lambda directory: loaded_paths.append(
        str(directory)
    )

    with pytest.warns(UserWarning, match="Trying backup"):
        restored = load_accelerator_checkpoint(
            accelerator,
            str(primary),
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            expected_compatibility=CompatibilityLevel.LEGACY_PROTOTYPE,
            expected_architecture=toy_architecture_summary(),
            expected_tokenizer_sha256="a" * 64,
        )

    assert restored == (7, 11, 0.5)
    assert loaded_paths == [str(primary), str(backup)]


def test_checkpoint_resolution_uses_backup_before_deepspeed_tensor_load(
    tmp_path: Path,
):
    from qwen3_omni_pretrain.training import checkpoint

    primary = tmp_path / "checkpoint"
    backup = tmp_path / "checkpoint.backup"
    primary.mkdir()
    backup.mkdir()
    (primary / "architecture.json").write_text("{}", encoding="utf-8")
    write_checkpoint_metadata(backup, checkpoint_metadata())

    with pytest.warns(UserWarning, match="Trying backup"):
        resolved = checkpoint.resolve_checkpoint_directory(
            str(primary),
            expected_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
            expected_compatibility=CompatibilityLevel.LEGACY_PROTOTYPE,
            expected_architecture=toy_architecture_summary(),
            expected_tokenizer_sha256="a" * 64,
        )

    assert resolved == str(backup)


class _DistributedAccelerator:
    def __init__(self, rank: int, world_size: int) -> None:
        self.is_main_process = rank == 0
        self.num_processes = world_size
        self.device = torch.device("cpu")
        self.loaded = False

    def wait_for_everyone(self) -> None:
        dist.barrier()

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        assert reduction == "sum"
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    def unwrap_model(self, model):
        return model

    def load_state(self, directory) -> None:
        self.loaded = True


def _tp_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_dir: str,
    action: str,
    expected_error: str | None = None,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        from qwen3_omni_pretrain.training import accelerator_utils

        accelerator_utils.get_tensor_model_parallel_world_size = (
            lambda: world_size
        )
        accelerator_utils.get_tensor_model_parallel_rank = lambda: rank
        accelerator = _DistributedAccelerator(rank, world_size)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(rank + 1)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        model.weight.grad = torch.full_like(
            model.weight,
            float(rank + 1),
        )
        optimizer.step()
        expected_weight = float(model.weight.item())
        expected_momentum = float(
            optimizer.state[model.weight]["momentum_buffer"].item()
        )
        metadata = checkpoint_metadata()

        if action == "save":
            accelerator_utils.save_tp_sharded_checkpoint(
                accelerator,
                model,
                optimizer=optimizer,
                scheduler=None,
                checkpoint_dir=checkpoint_dir,
                epoch=3,
                global_step=9,
                best_val_loss=0.25,
                metadata=metadata,
            )
            return

        if action == "roundtrip":
            accelerator_utils.save_tp_sharded_checkpoint(
                accelerator,
                model,
                optimizer=optimizer,
                scheduler=None,
                checkpoint_dir=checkpoint_dir,
                epoch=3,
                global_step=9,
                best_val_loss=0.25,
                metadata=metadata,
            )
            with torch.no_grad():
                model.weight.zero_()
            optimizer.state.clear()
            restored = accelerator_utils.load_accelerator_checkpoint(
                accelerator,
                checkpoint_dir,
                model=model,
                optimizer=optimizer,
                expected_profile=metadata.manifest.architecture_profile,
                expected_compatibility=(
                    metadata.manifest.compatibility_level
                ),
                expected_architecture=metadata.architecture,
                expected_tokenizer_sha256=metadata.tokenizer_sha256,
            )
            assert restored == (3, 9, 0.25)
            assert model.weight.item() == pytest.approx(expected_weight)
            assert optimizer.state[model.weight][
                "momentum_buffer"
            ].item() == pytest.approx(expected_momentum)
            return

        if action == "load-error":
            with torch.no_grad():
                model.weight.fill_(-10.0 - rank)
            untouched_weight = float(model.weight.item())
            with pytest.raises(ValueError, match=expected_error or ""):
                accelerator_utils.load_accelerator_checkpoint(
                    accelerator,
                    checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    expected_profile=metadata.manifest.architecture_profile,
                    expected_compatibility=(
                        metadata.manifest.compatibility_level
                    ),
                    expected_architecture=metadata.architecture,
                    expected_tokenizer_sha256=metadata.tokenizer_sha256,
                )
            assert model.weight.item() == pytest.approx(untouched_weight)
            return

        if action == "load-payload-error":
            with torch.no_grad():
                model.weight.fill_(-10.0 - rank)
            untouched_weight = float(model.weight.item())
            with pytest.raises(
                RuntimeError,
                match=expected_error or "",
            ):
                accelerator_utils.load_accelerator_checkpoint(
                    accelerator,
                    checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    expected_profile=(
                        metadata.manifest.architecture_profile
                    ),
                    expected_compatibility=(
                        metadata.manifest.compatibility_level
                    ),
                    expected_architecture=metadata.architecture,
                    expected_tokenizer_sha256=(
                        metadata.tokenizer_sha256
                    ),
                )
            assert model.weight.item() == pytest.approx(untouched_weight)
            return

        raise AssertionError(f"unknown action: {action}")
    finally:
        dist.destroy_process_group()


def _spawn_tp(
    tmp_path: Path,
    checkpoint_dir: Path,
    action: str,
    expected_error: str | None = None,
) -> None:
    rendezvous = tmp_path / f"dist-{action}-{checkpoint_dir.name}"
    mp.spawn(
        _tp_worker,
        args=(
            2,
            f"file://{rendezvous}",
            str(checkpoint_dir),
            action,
            expected_error,
        ),
        nprocs=2,
        join=True,
    )


def test_tp_checkpoint_round_trip_preserves_distinct_rank_local_shards(
    tmp_path: Path,
):
    checkpoint_dir = tmp_path / "checkpoint"

    _spawn_tp(tmp_path, checkpoint_dir, "roundtrip")

    manifest = json.loads(
        (checkpoint_dir / "tp_shards.json").read_text(encoding="utf-8")
    )
    assert manifest["tp_world_size"] == 2
    assert manifest["world_size"] == 2
    assert [entry["global_rank"] for entry in manifest["shards"]] == [0, 1]
    assert [entry["tp_rank"] for entry in manifest["shards"]] == [0, 1]
    assert {
        path.name
        for path in checkpoint_dir.glob("tp-shard-rank-*.pt")
    } == {
        "tp-shard-rank-00000.pt",
        "tp-shard-rank-00001.pt",
    }
    assert not (checkpoint_dir / "train.pt").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw, path: raw.update({"tp_world_size": 4}),
            "TP degree",
        ),
        (
            lambda raw, path: raw["shards"].pop(),
            "complete shard set",
        ),
        (
            lambda raw, path: raw["shards"][1].update({"tp_rank": 0}),
            "TP rank",
        ),
        (
            lambda raw, path: [
                entry.update({"model_schema_sha256": "0" * 64})
                for entry in raw["shards"]
            ],
            "parameter schema",
        ),
    ],
)
def test_tp_loader_rejects_topology_or_schema_before_local_restore(
    tmp_path: Path,
    mutate: Callable[[dict[str, object], Path], object],
    message: str,
):
    checkpoint_dir = tmp_path / "checkpoint"
    _spawn_tp(tmp_path, checkpoint_dir, "save")
    manifest_path = checkpoint_dir / "tp_shards.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(raw, checkpoint_dir)
    manifest_path.write_text(
        json.dumps(raw, sort_keys=True),
        encoding="utf-8",
    )

    _spawn_tp(tmp_path, checkpoint_dir, "load-error", message)


def test_tp_payload_schema_mismatch_is_collective_and_precedes_apply(
    tmp_path: Path,
):
    checkpoint_dir = tmp_path / "checkpoint"
    _spawn_tp(tmp_path, checkpoint_dir, "save")
    shard = checkpoint_dir / "tp-shard-rank-00001.pt"
    payload = torch.load(shard, map_location="cpu")
    payload["model"].pop("weight")
    torch.save(payload, shard)
    manifest_path = checkpoint_dir / "tp_shards.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][1]["sha256"] = hashlib.sha256(
        shard.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )

    _spawn_tp(
        tmp_path,
        checkpoint_dir,
        "load-payload-error",
        "payload validation",
    )


def test_tp_overwrite_failure_rolls_back_complete_old_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from qwen3_omni_pretrain.training import accelerator_utils

    destination = tmp_path / "checkpoint"
    _seed_old_generation(destination)
    monkeypatch.setattr(
        accelerator_utils,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        accelerator_utils,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        accelerator_utils,
        "write_checkpoint_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("injected publication failure")
        ),
    )

    with pytest.raises(RuntimeError, match="publication"):
        model = FakeModel()
        accelerator_utils.save_tp_sharded_checkpoint(
            FakeAccelerator(),
            model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            scheduler=None,
            checkpoint_dir=str(destination),
            epoch=1,
            global_step=2,
            best_val_loss=3.0,
            metadata=checkpoint_metadata(),
        )

    assert (destination / "old-payload").read_text(
        encoding="utf-8"
    ) == "stable"
    assert not (destination / "train.pt").exists()
    assert load_checkpoint_metadata(destination) == checkpoint_metadata()
