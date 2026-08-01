from dataclasses import replace

import pytest
import torch

from qwen3_omni_pretrain.training import trainer_thinker
from qwen3_omni_pretrain.training.stage2_config import Stage2RuntimeConfig


def runtime_config():
    return Stage2RuntimeConfig(
        experiment_name="stage2-test",
        stage1_init_ckpt="outputs/stage1",
        model_config_path="model.yaml",
        train_corpus_path="train.jsonl",
        val_corpus_path="val.jsonl",
        image_root="images",
        audio_root="audio",
        output_dir="outputs/stage2",
        resume_from_checkpoint="resume",
        num_epochs=2,
        max_steps=12,
        batch_size=7,
        max_seq_length=123,
        learning_rate=1e-5,
        weight_decay=0.02,
        warmup_ratio=0.05,
        gradient_accumulation_steps=3,
        logging_steps=4,
        eval_steps=5,
        save_steps=6,
        num_workers=0,
        shuffle=False,
        skip_bad_media=True,
        seed=777,
        ddp=False,
        fp16=False,
        bf16=True,
        fp8=False,
        int8_optimizer=False,
    )


def test_epoch_driver_applies_eval_save_and_max_steps(monkeypatch):
    runtime = replace(
        runtime_config(),
        max_steps=5,
        eval_steps=2,
        save_steps=3,
    )
    events = []
    model = torch.nn.Linear(1, 1)

    def fake_train_one_epoch(**kwargs):
        for step in range(1, 10):
            if kwargs["should_stop_fn"]():
                break
            kwargs["after_step_fn"](
                step=step,
                loss=1.0,
                batch_idx=step - 1,
                ce_loss=None,
                aux_loss=None,
                lr=1e-4,
                model=model,
            )
            if kwargs["should_stop_fn"]():
                break
        return 1.0

    losses = iter((0.5, 0.6))

    def evaluate_fn(step):
        events.append(("eval", step))
        model.eval()
        return next(losses)

    def checkpoint_fn(tag, epoch, step, best):
        events.append(("save", tag, epoch, step, best))

    monkeypatch.setattr(
        trainer_thinker, "train_one_epoch", fake_train_one_epoch
    )
    result = trainer_thinker._run_stage2_epoch(
        runtime=runtime,
        epoch=0,
        starting_global_step=0,
        best_val_loss=1.0,
        model=model,
        train_loader=object(),
        optimizer=object(),
        scheduler=object(),
        device=torch.device("cpu"),
        autocast_dtype=None,
        grad_scaler=None,
        evaluate_fn=evaluate_fn,
        checkpoint_fn=checkpoint_fn,
    )

    assert result.global_step == 5
    assert result.should_stop is True
    assert result.best_val_loss == 0.5
    assert ("eval", 2) in events
    assert ("eval", 4) in events
    assert any(event[1] == "best_step_2" for event in events if event[0] == "save")
    assert any(event[1] == "step_3" for event in events if event[0] == "save")
    assert model.training is True


def test_epoch_driver_resumes_global_step_and_stops_after_one_update(
    monkeypatch,
):
    runtime = replace(
        runtime_config(),
        max_steps=5,
        eval_steps=0,
        save_steps=0,
    )
    local_steps = []
    logged_steps = []

    def fake_train_one_epoch(**kwargs):
        for step in range(1, 4):
            if kwargs["should_stop_fn"]():
                break
            local_steps.append(step)
            kwargs["log_step_fn"](
                step=step,
                loss=1.0,
                batch_idx=step - 1,
                ce_loss=None,
                aux_loss=None,
                lr=1e-4,
            )
            kwargs["after_step_fn"](
                step=step,
                loss=1.0,
                batch_idx=step - 1,
                ce_loss=None,
                aux_loss=None,
                lr=1e-4,
                model=kwargs["model"],
            )
            if kwargs["should_stop_fn"]():
                break
        return 1.0

    monkeypatch.setattr(
        trainer_thinker, "train_one_epoch", fake_train_one_epoch
    )
    result = trainer_thinker._run_stage2_epoch(
        runtime=runtime,
        epoch=1,
        starting_global_step=4,
        best_val_loss=1.0,
        model=torch.nn.Linear(1, 1),
        train_loader=object(),
        optimizer=object(),
        scheduler=object(),
        device=torch.device("cpu"),
        autocast_dtype=None,
        grad_scaler=None,
        evaluate_fn=lambda step: pytest.fail("evaluation must be disabled"),
        checkpoint_fn=lambda *args: pytest.fail(
            "periodic checkpoints must be disabled"
        ),
        log_step_fn=lambda **event: logged_steps.append(event["step"]),
    )

    assert local_steps == [1]
    assert logged_steps == [5]
    assert result.global_step == 5
    assert result.should_stop is True


def test_schedule_shape_uses_ceil_and_max_steps():
    runtime = replace(
        runtime_config(),
        num_epochs=2,
        max_steps=5,
        gradient_accumulation_steps=2,
    )
    assert trainer_thinker._stage2_schedule_shape(
        runtime, num_batches=5
    ) == (3, 5, 2)
