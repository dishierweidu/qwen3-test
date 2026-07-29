from dataclasses import asdict

import pytest

from conftest import TinyTokenizer
from qwen3_omni_pretrain.training import stage2_config, trainer_thinker


def complete_nested_config():
    return {
        "experiment_name": "stage2-test",
        "stage1_init_ckpt": "outputs/stage1",
        "model": {"model_config_path": "model.yaml"},
        "data": {
            "train_corpus_path": "train.jsonl",
            "val_corpus_path": "val.jsonl",
            "image_root": "images",
            "audio_root": "audio",
            "batch_size": 7,
            "max_seq_length": 123,
            "num_workers": 0,
            "shuffle": False,
            "skip_bad_media": True,
        },
        "train": {
            "output_dir": "outputs/stage2",
            "resume_from_checkpoint": "resume",
            "num_epochs": 2,
            "max_steps": 12,
            "learning_rate": 1e-5,
            "weight_decay": 0.02,
            "warmup_ratio": 0.05,
            "gradient_accumulation_steps": 3,
            "logging_steps": 4,
            "eval_steps": 5,
            "save_steps": 6,
            "seed": 777,
            "ddp": False,
            "fp16": False,
            "bf16": True,
            "fp8": False,
            "int8_optimizer": False,
        },
    }


EXPECTED_CANONICAL = {
    "experiment_name": "stage2-test",
    "stage1_init_ckpt": "outputs/stage1",
    "model_config_path": "model.yaml",
    "train_corpus_path": "train.jsonl",
    "val_corpus_path": "val.jsonl",
    "image_root": "images",
    "audio_root": "audio",
    "output_dir": "outputs/stage2",
    "resume_from_checkpoint": "resume",
    "num_epochs": 2,
    "max_steps": 12,
    "batch_size": 7,
    "max_seq_length": 123,
    "learning_rate": 1e-5,
    "weight_decay": 0.02,
    "warmup_ratio": 0.05,
    "gradient_accumulation_steps": 3,
    "logging_steps": 4,
    "eval_steps": 5,
    "save_steps": 6,
    "num_workers": 0,
    "shuffle": False,
    "skip_bad_media": True,
    "seed": 777,
    "ddp": False,
    "fp16": False,
    "bf16": True,
    "fp8": False,
    "int8_optimizer": False,
}


def test_legacy_stage2_config_constructor_contract():
    cfg = trainer_thinker.Stage2TrainConfig(
        stage1_init_ckpt="outputs/stage1",
        model_config_path="model.yaml",
        train_corpus_path="train.jsonl",
        val_corpus_path="val.jsonl",
        image_root="images",
        audio_root="audio",
        output_dir="outputs/stage2",
    )
    assert cfg.batch_size == 2
    assert cfg.max_seq_length == 1024
    assert cfg.learning_rate == 1e-4
    assert cfg.eval_steps == 200
    assert cfg.seed == 42


def test_complete_nested_mapping():
    result = stage2_config.normalize_stage2_config(
        complete_nested_config()
    )
    assert asdict(result) == EXPECTED_CANONICAL


def test_cli_wins_but_legacy_still_warns():
    raw = complete_nested_config()
    raw["batch_size"] = 11
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"batch_size.*legacy=11.*canonical=7.*"
            r"cli=13.*selected=cli"
        ),
    ):
        result = stage2_config.normalize_stage2_config(
            raw, overrides={"batch_size": 13}
        )
    assert result.batch_size == 13


def test_legacy_wins_over_canonical_and_warns():
    raw = complete_nested_config()
    raw["batch_size"] = 11
    with pytest.warns(
        DeprecationWarning,
        match=r"batch_size.*legacy=11.*canonical=7.*selected=legacy",
    ):
        result = stage2_config.normalize_stage2_config(raw)
    assert result.batch_size == 11


def test_false_and_zero_cli_overrides_are_not_dropped():
    result = stage2_config.normalize_stage2_config(
        complete_nested_config(),
        overrides={"shuffle": False, "num_workers": 0},
    )
    assert result.shuffle is False
    assert result.num_workers == 0


def test_documented_defaults_are_used():
    raw = complete_nested_config()
    for key in (
        "num_epochs",
        "max_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "seed",
        "ddp",
        "fp16",
        "bf16",
        "fp8",
        "int8_optimizer",
    ):
        raw["train"].pop(key)
    for key in (
        "batch_size",
        "max_seq_length",
        "num_workers",
        "shuffle",
        "skip_bad_media",
    ):
        raw["data"].pop(key)
    raw["train"].pop("resume_from_checkpoint")

    result = stage2_config.normalize_stage2_config(raw)

    assert result.resume_from_checkpoint is None
    assert result.num_epochs == 1
    assert result.max_steps == -1
    assert result.batch_size == 2
    assert result.max_seq_length == 1024
    assert result.learning_rate == 1e-4
    assert result.weight_decay == 0.01
    assert result.warmup_ratio == 0.03
    assert result.gradient_accumulation_steps == 1
    assert result.logging_steps == 10
    assert result.eval_steps == 200
    assert result.save_steps == 200
    assert result.num_workers == 4
    assert result.shuffle is True
    assert result.skip_bad_media is False
    assert result.seed == 42
    assert result.ddp is False
    assert result.fp16 is False
    assert result.bf16 is False
    assert result.fp8 is False
    assert result.int8_optimizer is False


def test_legacy_flat_dataclass_is_supported_with_warning():
    legacy = stage2_config.Stage2TrainConfig(
        stage1_init_ckpt="outputs/stage1",
        model_config_path="model.yaml",
        train_corpus_path="train.jsonl",
        val_corpus_path="val.jsonl",
        image_root="images",
        audio_root="audio",
        output_dir="outputs/stage2",
        batch_size=9,
    )
    with pytest.warns(DeprecationWarning, match="legacy top-level"):
        result = stage2_config.normalize_stage2_config(legacy)
    assert result.batch_size == 9
    assert result.experiment_name == "thinker_stage2"


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("train", "num_epochs", 0, "num_epochs must be positive"),
        ("train", "max_steps", 0, "max_steps must be -1 or positive"),
        ("train", "max_steps", -2, "max_steps must be -1 or positive"),
        ("data", "batch_size", 0, "batch_size must be positive"),
        ("data", "max_seq_length", 0, "max_seq_length must be positive"),
        ("data", "num_workers", -1, "num_workers must be non-negative"),
        ("train", "learning_rate", 0, "learning_rate must be positive"),
        ("train", "weight_decay", -0.1, "weight_decay must be non-negative"),
        ("train", "warmup_ratio", 1.1, "warmup_ratio must be between 0 and 1"),
        (
            "train",
            "gradient_accumulation_steps",
            0,
            "gradient_accumulation_steps must be positive",
        ),
        ("train", "logging_steps", 0, "logging_steps must be positive"),
        ("train", "eval_steps", -1, "eval_steps must be non-negative"),
        ("train", "save_steps", -1, "save_steps must be non-negative"),
    ],
)
def test_invalid_ranges_fail(section, key, value, message):
    raw = complete_nested_config()
    raw[section][key] = value
    with pytest.raises(ValueError, match=message):
        stage2_config.normalize_stage2_config(raw)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("data", "shuffle"),
        ("data", "skip_bad_media"),
        ("train", "ddp"),
        ("train", "fp16"),
        ("train", "bf16"),
        ("train", "fp8"),
        ("train", "int8_optimizer"),
    ],
)
def test_boolean_fields_reject_string_values(section, key):
    raw = complete_nested_config()
    raw[section][key] = "false"
    with pytest.raises(TypeError, match=rf"{key} must be a boolean"):
        stage2_config.normalize_stage2_config(raw)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "stage1_init_ckpt"),
        ("model", "model_config_path"),
        ("data", "train_corpus_path"),
        ("data", "val_corpus_path"),
        ("data", "image_root"),
        ("data", "audio_root"),
        ("train", "output_dir"),
    ],
)
def test_required_paths_reject_empty_strings(section, key):
    raw = complete_nested_config()
    target = raw if section is None else raw[section]
    target[key] = ""
    with pytest.raises(ValueError, match=rf"{key} must be a non-empty string"):
        stage2_config.normalize_stage2_config(raw)


def test_optional_resume_path_rejects_non_string_values():
    raw = complete_nested_config()
    raw["train"]["resume_from_checkpoint"] = 7
    with pytest.raises(
        TypeError,
        match="resume_from_checkpoint must be a string or None",
    ):
        stage2_config.normalize_stage2_config(raw)


def test_invalid_precision_combination_fails():
    raw = complete_nested_config()
    raw["train"]["fp16"] = True
    with pytest.raises(ValueError, match="Only one of fp16/bf16/fp8"):
        stage2_config.normalize_stage2_config(raw)


def test_cli_resume_wins_but_legacy_resume_still_warns(monkeypatch):
    from qwen3_omni_pretrain.training import trainer_thinker

    raw = complete_nested_config()
    raw["resume_from_checkpoint"] = "legacy"
    monkeypatch.setattr(trainer_thinker, "set_seed", lambda seed: None)
    with pytest.warns(
        DeprecationWarning,
        match=r"resume_from_checkpoint.*selected=cli",
    ):
        runtime = trainer_thinker._prepare_stage2_runtime(
            raw, "cli-checkpoint"
        )
    assert runtime.resume_from_checkpoint == "cli-checkpoint"


def test_normalized_runtime_is_an_idempotent_boundary_value():
    runtime = stage2_config.normalize_stage2_config(
        complete_nested_config()
    )
    assert stage2_config.normalize_stage2_config(runtime) is runtime


def test_stage2_ddp_fails_before_seed_or_tokenizer(monkeypatch):
    from qwen3_omni_pretrain.training import trainer_thinker

    raw = complete_nested_config()
    raw["train"]["ddp"] = True
    monkeypatch.setattr(
        trainer_thinker,
        "set_seed",
        lambda seed: pytest.fail("seed ran before config validation"),
    )
    monkeypatch.setattr(
        trainer_thinker.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail(
            "tokenizer allocated before config validation"
        ),
    )
    with pytest.raises(NotImplementedError, match="Stage2 ddp=true"):
        trainer_thinker.train_thinker_stage2(
            raw, tokenizer_name_or_path="unused"
        )


def test_checked_in_yaml_reaches_seed_collator_and_dataloaders(monkeypatch):
    from qwen3_omni_pretrain.training import trainer_thinker
    from qwen3_omni_pretrain.utils.config_utils import load_yaml

    raw = load_yaml("configs/train/stage2_omni_vision_audio.yaml")
    calls = []
    seeds = []

    class SpyDataLoader:
        def __init__(self, dataset, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(trainer_thinker, "DataLoader", SpyDataLoader)
    monkeypatch.setattr(trainer_thinker, "set_seed", seeds.append)
    runtime = trainer_thinker._prepare_stage2_runtime(raw)
    collator = trainer_thinker._build_stage2_collator(
        runtime, TinyTokenizer()
    )
    trainer_thinker._build_stage2_dataloaders(
        runtime, object(), object(), collator
    )

    assert seeds == [42]
    assert runtime.max_steps == -1
    assert runtime.gradient_accumulation_steps == 8
    assert runtime.eval_steps == 200
    assert runtime.save_steps == 200
    assert runtime.ddp is False
    assert collator.max_seq_length == 1024
    assert calls[0]["batch_size"] == 2
    assert calls[0]["num_workers"] == 4
    assert calls[0]["shuffle"] is False
    assert calls[1]["shuffle"] is False
