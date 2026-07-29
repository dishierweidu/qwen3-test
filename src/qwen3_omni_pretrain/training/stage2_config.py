from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Optional
import warnings


@dataclass
class Stage2TrainConfig:
    stage1_init_ckpt: str
    model_config_path: str
    train_corpus_path: str
    val_corpus_path: str
    image_root: str
    audio_root: str
    output_dir: str
    resume_from_checkpoint: Optional[str] = None
    num_epochs: int = 1
    max_steps: int = -1
    batch_size: int = 2
    max_seq_length: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    num_workers: int = 4
    shuffle: bool = True
    skip_bad_media: bool = False
    seed: int = 42
    ddp: bool = False
    fp16: bool = False
    bf16: bool = False
    fp8: bool = False
    int8_optimizer: bool = False


@dataclass(frozen=True)
class Stage2RuntimeConfig:
    experiment_name: str
    stage1_init_ckpt: str
    model_config_path: str
    train_corpus_path: str
    val_corpus_path: str
    image_root: str
    audio_root: str
    output_dir: str
    resume_from_checkpoint: Optional[str]
    num_epochs: int
    max_steps: int
    batch_size: int
    max_seq_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    num_workers: int
    shuffle: bool
    skip_bad_media: bool
    seed: int
    ddp: bool
    fp16: bool
    bf16: bool
    fp8: bool
    int8_optimizer: bool


_MISSING = object()


def _select(raw, section, key, default=_MISSING, overrides=None):
    nested = raw.get(section, {})
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise TypeError(f"{section} must be a mapping")

    has_cli = (
        overrides is not None
        and key in overrides
        and overrides[key] is not None
    )
    has_legacy = key in raw
    canonical = nested.get(key, _MISSING)

    if has_legacy:
        details = [f"legacy={raw[key]!r}"]
        if canonical is not _MISSING:
            details.append(f"canonical={canonical!r}")
        if has_cli:
            details.append(f"cli={overrides[key]!r}")
        selected = "cli" if has_cli else "legacy"
        warnings.warn(
            f"legacy top-level {key} is deprecated "
            f"({', '.join(details)}, selected={selected})",
            DeprecationWarning,
            stacklevel=3,
        )

    if has_cli:
        return overrides[key]
    if has_legacy:
        return raw[key]
    if canonical is not _MISSING:
        return canonical
    if default is _MISSING:
        raise KeyError(
            f"missing required configuration value: {section}.{key}"
        )
    return default


def _require_bool(name, value):
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_stage2_runtime(
    result: Stage2RuntimeConfig,
) -> Stage2RuntimeConfig:
    for name in (
        "stage1_init_ckpt",
        "model_config_path",
        "train_corpus_path",
        "val_corpus_path",
        "image_root",
        "audio_root",
        "output_dir",
    ):
        value = getattr(result, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    if (
        result.resume_from_checkpoint is not None
        and not isinstance(result.resume_from_checkpoint, str)
    ):
        raise TypeError(
            "resume_from_checkpoint must be a string or None"
        )

    for name in (
        "shuffle",
        "skip_bad_media",
        "ddp",
        "fp16",
        "bf16",
        "fp8",
        "int8_optimizer",
    ):
        _require_bool(name, getattr(result, name))

    if result.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if result.max_steps != -1 and result.max_steps <= 0:
        raise ValueError("max_steps must be -1 or positive")
    if result.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if result.max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    if result.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if result.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if result.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= result.warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if result.gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive"
        )
    if result.logging_steps <= 0:
        raise ValueError("logging_steps must be positive")
    if result.eval_steps < 0:
        raise ValueError("eval_steps must be non-negative")
    if result.save_steps < 0:
        raise ValueError("save_steps must be non-negative")

    if sum((result.fp16, result.bf16, result.fp8)) > 1:
        raise ValueError("Only one of fp16/bf16/fp8 can be enabled")
    if result.ddp:
        raise NotImplementedError(
            "Stage2 ddp=true is not implemented in this P0; "
            "set train.ddp=false"
        )
    return result


def normalize_stage2_config(
    raw,
    *,
    overrides=None,
) -> Stage2RuntimeConfig:
    if isinstance(raw, Stage2RuntimeConfig):
        effective = {
            key: value
            for key, value in (overrides or {}).items()
            if value is not None
        }
        if not effective:
            return _validate_stage2_runtime(raw)
        return _validate_stage2_runtime(replace(raw, **effective))

    if is_dataclass(raw) and not isinstance(raw, type):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        raise TypeError("Stage2 configuration must be a mapping or dataclass")

    result = Stage2RuntimeConfig(
        experiment_name=raw.get("experiment_name", "thinker_stage2"),
        stage1_init_ckpt=raw["stage1_init_ckpt"],
        model_config_path=_select(
            raw, "model", "model_config_path", overrides=overrides
        ),
        train_corpus_path=_select(
            raw, "data", "train_corpus_path", overrides=overrides
        ),
        val_corpus_path=_select(
            raw, "data", "val_corpus_path", overrides=overrides
        ),
        image_root=_select(
            raw, "data", "image_root", overrides=overrides
        ),
        audio_root=_select(
            raw, "data", "audio_root", overrides=overrides
        ),
        output_dir=_select(
            raw, "train", "output_dir", overrides=overrides
        ),
        resume_from_checkpoint=_select(
            raw,
            "train",
            "resume_from_checkpoint",
            None,
            overrides,
        ),
        num_epochs=_select(
            raw, "train", "num_epochs", 1, overrides
        ),
        max_steps=_select(raw, "train", "max_steps", -1, overrides),
        batch_size=_select(raw, "data", "batch_size", 2, overrides),
        max_seq_length=_select(
            raw, "data", "max_seq_length", 1024, overrides
        ),
        learning_rate=_select(
            raw, "train", "learning_rate", 1e-4, overrides
        ),
        weight_decay=_select(
            raw, "train", "weight_decay", 0.01, overrides
        ),
        warmup_ratio=_select(
            raw, "train", "warmup_ratio", 0.03, overrides
        ),
        gradient_accumulation_steps=_select(
            raw,
            "train",
            "gradient_accumulation_steps",
            1,
            overrides,
        ),
        logging_steps=_select(
            raw, "train", "logging_steps", 10, overrides
        ),
        eval_steps=_select(
            raw, "train", "eval_steps", 200, overrides
        ),
        save_steps=_select(
            raw, "train", "save_steps", 200, overrides
        ),
        num_workers=_select(
            raw, "data", "num_workers", 4, overrides
        ),
        shuffle=_select(raw, "data", "shuffle", True, overrides),
        skip_bad_media=_select(
            raw, "data", "skip_bad_media", False, overrides
        ),
        seed=_select(raw, "train", "seed", 42, overrides),
        ddp=_select(raw, "train", "ddp", False, overrides),
        fp16=_select(raw, "train", "fp16", False, overrides),
        bf16=_select(raw, "train", "bf16", False, overrides),
        fp8=_select(raw, "train", "fp8", False, overrides),
        int8_optimizer=_select(
            raw, "train", "int8_optimizer", False, overrides
        ),
    )
    return _validate_stage2_runtime(result)
