from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import textwrap
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from qwen3_omni_pretrain.training import trainer_thinker


def _tp_interrupt_poll_worker(
    rank: int,
    world_size: int,
    init_method: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        from qwen3_omni_pretrain.training import loop_accelerator

        loop_accelerator.get_tensor_model_parallel_world_size = (
            lambda: world_size
        )
        loop_accelerator.get_tensor_model_parallel_group = (
            lambda: dist.group.WORLD
        )
        loop_accelerator.get_tensor_model_parallel_rank = lambda: rank

        class _Accelerator:
            device = torch.device("cpu")
            is_main_process = rank == 0
            num_processes = world_size

            def reduce(
                self,
                value: torch.Tensor,
                reduction: str,
            ) -> torch.Tensor:
                assert reduction == "max"
                reduced = value.clone()
                dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
                return reduced

        accelerator = _Accelerator()
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def globally_polled_stop() -> bool:
            local = torch.tensor(
                [int(rank == 1)],
                dtype=torch.int32,
            )
            return bool(
                accelerator.reduce(local, reduction="max").item()
            )

        _, updates = loop_accelerator.train_one_epoch_accelerator(
            accelerator=accelerator,
            model=model,
            dataloader=[],
            optimizer=optimizer,
            scheduler=None,
            cfg=object(),
            should_stop_fn=globally_polled_stop,
        )
        assert updates == 0
    finally:
        dist.destroy_process_group()


def _join_spawn_without_hanging(context, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if context.join(timeout=0.5):
            return True
    for process in context.processes:
        process.terminate()
    for process in context.processes:
        process.join(timeout=2)
    return False


def _deepspeed_backup_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint: str,
    mode: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        def validate_metadata(candidate, **kwargs):
            if (
                mode == "rank-local-metadata-failure"
                and rank == 1
                and candidate == checkpoint
            ):
                raise ValueError("rank1 cannot validate primary metadata")

        trainer_thinker.load_checkpoint_metadata = validate_metadata
        engine_calls: list[str] = []
        scheduler_calls: list[str] = []

        class _Engine:
            def load_checkpoint(self, candidate: str):
                generation = Path(candidate).name
                engine_calls.append(generation)
                if (
                    mode == "rank-local-engine-failure"
                    and rank == 1
                    and candidate == checkpoint
                ):
                    raise OSError("rank1 primary shard is corrupt")
                if (
                    mode == "corrupt-primary-client-state"
                    and candidate == checkpoint
                ):
                    return candidate, {"step": "not-an-integer"}
                if (
                    mode == "divergent-primary-client-state"
                    and candidate == checkpoint
                ):
                    return candidate, {
                        "step": rank + 1,
                        "epoch": 2,
                    }
                return candidate, {"step": 9, "epoch": 2}

        class _Scheduler:
            def load_state_dict(self, state):
                scheduler_calls.append(str(state["generation"]))

        loaded_root, load_path, client_state = (
            trainer_thinker._load_deepspeed_resume_collectively(
                model_engine=_Engine(),
                scheduler=_Scheduler(),
                resume_path=checkpoint,
                resume_identity={},
            )
        )
        Path(f"{checkpoint}.rank{rank}.json").write_text(
            json.dumps(
                {
                    "engine_calls": engine_calls,
                    "scheduler_calls": scheduler_calls,
                    "loaded_root": loaded_root,
                    "load_path": load_path,
                    "step": client_state["step"],
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def _accelerator_backup_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        from qwen3_omni_pretrain.training import accelerator_utils

        metadata_calls: list[str] = []
        load_calls: list[str] = []

        def validate_metadata(candidate, **kwargs):
            metadata_calls.append(Path(candidate).name)
            if rank == 1 and candidate == checkpoint:
                raise ValueError("rank1 cannot validate primary metadata")

        accelerator_utils.load_checkpoint_metadata = validate_metadata

        class _Accelerator:
            is_main_process = rank == 0
            num_processes = world_size
            device = torch.device("cpu")

            def wait_for_everyone(self):
                dist.barrier()

            def reduce(self, value, reduction):
                assert reduction == "sum"
                reduced = value.clone()
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                return reduced

            def load_state(self, candidate):
                load_calls.append(Path(candidate).name)

        restored = accelerator_utils.load_accelerator_checkpoint(
            _Accelerator(),
            checkpoint,
        )
        Path(f"{checkpoint}.accelerator-rank{rank}.json").write_text(
            json.dumps(
                {
                    "metadata_calls": metadata_calls,
                    "load_calls": load_calls,
                    "restored": restored,
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def _collective_save_event_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_prefix: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        events: list[str] = []
        real_all_reduce = dist.all_reduce
        decision_phase = ["no-save"]

        def recorded_all_reduce(*args, **kwargs):
            events.append(f"decision:{decision_phase[0]}")
            return real_all_reduce(*args, **kwargs)

        trainer_thinker.dist.all_reduce = recorded_all_reduce

        best = trainer_thinker._save_deepspeed_best_collectively(
            tag="best-no-save",
            local_val_loss=4.0,
            best_val_loss=3.0,
            device=torch.device("cpu"),
            save_fn=lambda tag: events.append(f"save:{tag}"),
        )
        assert best == 3.0

        decision_phase[0] = "save"
        best = trainer_thinker._save_deepspeed_best_collectively(
            tag="best-step-2",
            local_val_loss=float(rank + 1),
            best_val_loss=best,
            device=torch.device("cpu"),
            save_fn=lambda tag: events.append(f"save:{tag}"),
        )
        assert best == pytest.approx(1.5)
        trainer_thinker._save_deepspeed_terminal_collectively(
            tag="latest",
            save_fn=lambda tag: events.append(f"save:{tag}"),
        )

        Path(f"{result_prefix}.best-rank{rank}.json").write_text(
            json.dumps(events),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def _interrupt_save_event_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_prefix: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        events: list[str] = []
        original_synchronize = (
            trainer_thinker._synchronize_interrupt_request
        )

        def recorded_synchronize(*args, **kwargs):
            events.append("decision")
            return original_synchronize(*args, **kwargs)

        trainer_thinker._synchronize_interrupt_request = (
            recorded_synchronize
        )
        saved = (
            trainer_thinker._save_interrupt_checkpoint_if_requested(
                local_requested=rank == 1,
                device=torch.device("cpu"),
                global_step=7,
                save_fn=lambda tag: events.append(f"save:{tag}"),
            )
        )
        assert saved is True
        Path(f"{result_prefix}.interrupt-rank{rank}.json").write_text(
            json.dumps(events),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tag", ["best_step_4", "best_epoch0"])
def test_deepspeed_best_checkpoint_caller_reduces_before_every_rank_saves(
    tag: str,
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []

    def all_reduce(value: torch.Tensor, op) -> None:
        assert op is trainer_thinker.dist.ReduceOp.SUM
        events.append("decision")
        value.fill_(6.0)

    monkeypatch.setattr(
        trainer_thinker.dist,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        trainer_thinker.dist,
        "is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        trainer_thinker.dist,
        "all_reduce",
        all_reduce,
    )
    monkeypatch.setattr(
        trainer_thinker.dist,
        "get_world_size",
        lambda: 2,
    )

    new_best = trainer_thinker._save_deepspeed_best_collectively(
        tag=tag,
        local_val_loss=10.0,
        best_val_loss=5.0,
        device=torch.device("cpu"),
        save_fn=lambda saved_tag: events.append(f"save:{saved_tag}"),
    )

    assert new_best == 3.0
    assert events == ["decision", f"save:{tag}"]


@pytest.mark.parametrize(
    "tag",
    ["latest", "interrupted_step_9"],
)
def test_deepspeed_terminal_checkpoint_caller_invokes_collective_saver_on_peer(
    tag: str,
):
    events: list[str] = []

    trainer_thinker._save_deepspeed_terminal_collectively(
        tag=tag,
        save_fn=lambda saved_tag: events.append(saved_tag),
    )

    assert events == [tag]


def test_accelerator_keyboard_interrupt_caller_saves_on_non_main_rank():
    events: list[str] = []

    class PeerAccelerator:
        is_main_process = False

    trainer_thinker._save_accelerator_interrupt_collectively(
        accelerator=PeerAccelerator(),
        global_step=9,
        save_fn=lambda saved_tag: events.append(saved_tag),
    )

    assert events == ["interrupted_step_9"]


def test_interrupt_poll_shares_peer_request_before_collective_save():
    events: list[str] = []

    class PeerRequestedAccelerator:
        def reduce(self, value, reduction):
            assert reduction == "max"
            events.append("shared-decision")
            return value.fill_(1)

    assert trainer_thinker._synchronize_interrupt_request(
        False,
        device=torch.device("cpu"),
        accelerator=PeerRequestedAccelerator(),
    )
    assert events == ["shared-decision"]


def test_tp_interrupt_poll_is_entered_by_every_rank_without_deadlock(
    tmp_path: Path,
):
    context = mp.spawn(
        _tp_interrupt_poll_worker,
        args=(
            2,
            f"file://{tmp_path / 'tp-interrupt-rendezvous'}",
        ),
        nprocs=2,
        join=False,
    )

    assert _join_spawn_without_hanging(context, timeout=30), (
        "TP ranks deadlocked because not every rank entered the "
        "global interrupt poll"
    )


@pytest.mark.parametrize(
    "mode",
    [
        "rank-local-metadata-failure",
        "rank-local-engine-failure",
        "corrupt-primary-client-state",
        "divergent-primary-client-state",
        "corrupt-primary-scheduler",
    ],
)
def test_deepspeed_resume_retries_backup_collectively_for_whole_generation(
    tmp_path: Path,
    mode: str,
):
    checkpoint = tmp_path / "checkpoint"
    backup = tmp_path / "checkpoint.backup"
    checkpoint.mkdir()
    backup.mkdir()
    if mode == "corrupt-primary-scheduler":
        (checkpoint / "scheduler.pt").write_bytes(b"")
    else:
        torch.save(
            {"generation": "primary"},
            checkpoint / "scheduler.pt",
        )
    torch.save(
        {"generation": "backup"},
        backup / "scheduler.pt",
    )
    context = mp.spawn(
        _deepspeed_backup_worker,
        args=(
            2,
            f"file://{tmp_path / f'ds-{mode}-rendezvous'}",
            str(checkpoint),
            mode,
        ),
        nprocs=2,
        join=False,
    )

    assert _join_spawn_without_hanging(context, timeout=30), (
        "DeepSpeed ranks diverged while selecting a checkpoint generation"
    )
    for rank in range(2):
        result = json.loads(
            Path(f"{checkpoint}.rank{rank}.json").read_text(
                encoding="utf-8"
            )
        )
        expected_engine_calls = ["checkpoint.backup"]
        if mode != "rank-local-metadata-failure":
            expected_engine_calls.insert(0, "checkpoint")
        assert result["engine_calls"] == expected_engine_calls
        assert result["scheduler_calls"][-1] == "backup"
        assert result["loaded_root"] == str(backup)
        assert result["load_path"] == str(backup)
        assert result["step"] == 9


def test_accelerator_resume_selects_backup_collectively_on_asymmetric_metadata(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    backup = tmp_path / "checkpoint.backup"
    checkpoint.mkdir()
    backup.mkdir()
    torch.save(
        {"epoch": 1, "global_step": 3, "best_val_loss": 2.0},
        checkpoint / "trainer_state.pt",
    )
    torch.save(
        {"epoch": 2, "global_step": 7, "best_val_loss": 1.0},
        backup / "trainer_state.pt",
    )
    context = mp.spawn(
        _accelerator_backup_worker,
        args=(
            2,
            f"file://{tmp_path / 'accelerator-resume-rendezvous'}",
            str(checkpoint),
        ),
        nprocs=2,
        join=False,
    )

    assert _join_spawn_without_hanging(context, timeout=30), (
        "Accelerator ranks diverged while selecting a restore generation"
    )
    for rank in range(2):
        result = json.loads(
            Path(
                f"{checkpoint}.accelerator-rank{rank}.json"
            ).read_text(encoding="utf-8")
        )
        assert result["metadata_calls"] == [
            "checkpoint",
            "checkpoint.backup",
        ]
        assert result["load_calls"] == ["checkpoint.backup"]
        assert result["restored"] == [2, 7, 1.0]


def test_deepspeed_best_and_terminal_callers_save_on_every_rank_in_order(
    tmp_path: Path,
):
    result_prefix = str(tmp_path / "deepspeed-events")
    context = mp.spawn(
        _collective_save_event_worker,
        args=(
            2,
            f"file://{tmp_path / 'deepspeed-event-rendezvous'}",
            result_prefix,
        ),
        nprocs=2,
        join=False,
    )

    assert _join_spawn_without_hanging(context, timeout=30)
    for rank in range(2):
        events = json.loads(
            Path(f"{result_prefix}.best-rank{rank}.json").read_text(
                encoding="utf-8"
            )
        )
        assert events == [
            "decision:no-save",
            "decision:save",
            "save:best-step-2",
            "save:latest",
        ]


def test_coordinated_interrupt_decision_precedes_save_on_every_rank(
    tmp_path: Path,
):
    result_prefix = str(tmp_path / "interrupt-events")
    context = mp.spawn(
        _interrupt_save_event_worker,
        args=(
            2,
            f"file://{tmp_path / 'interrupt-event-rendezvous'}",
            result_prefix,
        ),
        nprocs=2,
        join=False,
    )

    assert _join_spawn_without_hanging(context, timeout=30)
    for rank in range(2):
        events = json.loads(
            Path(
                f"{result_prefix}.interrupt-rank{rank}.json"
            ).read_text(encoding="utf-8")
        )
        assert events == ["decision", "save:interrupted_step_7"]


def test_distributed_keyboard_interrupt_callers_fail_closed():
    accelerator_source = inspect.getsource(
        trainer_thinker._train_with_accelerator
    )
    stage1_source = inspect.getsource(
        trainer_thinker.train_thinker_stage1
    )

    assert "Unilateral KeyboardInterrupt" in accelerator_source
    assert 'getattr(accelerator, "num_processes", 1)) <= 1' in (
        accelerator_source
    )
    assert "Unilateral KeyboardInterrupt" in stage1_source
    assert "if distributed_world <= 1" in stage1_source


@pytest.mark.parametrize(
    ("entrypoint", "helper", "minimum_calls"),
    [
        (
            trainer_thinker.train_thinker_stage1,
            "_save_deepspeed_best_collectively",
            2,
        ),
        (
            trainer_thinker.train_thinker_stage1,
            "_save_deepspeed_terminal_collectively",
            1,
        ),
        (
            trainer_thinker._train_with_accelerator,
            "_save_accelerator_interrupt_collectively",
            1,
        ),
    ],
)
def test_real_training_callers_keep_collective_saves_outside_main_only_if(
    entrypoint,
    helper: str,
    minimum_calls: int,
):
    tree = ast.parse(textwrap.dedent(inspect.getsource(entrypoint)))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper
    ]

    assert len(calls) >= minimum_calls
    for call in calls:
        ancestor = parents.get(call)
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                condition = ast.unparse(ancestor.test)
                assert "is_main_process" not in condition
            ancestor = parents.get(ancestor)
