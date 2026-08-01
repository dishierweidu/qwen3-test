#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import torch
import yaml

from qwen3_omni_pretrain.architecture.summary import summarize_model
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime import (
    GenerationRequest,
    LegacyGreedyPrefillDecodeEngine,
    ModelDecodeInputs,
    ModelPrefillInputs,
    StateOwner,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/model/legacy_full_attention_tiny.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness-gated legacy incremental-decode benchmark",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--output-length", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    return parser.parse_args()


def _strict_positive(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _strict_nonnegative(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _load_model(
    config_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3OmniMoeThinkerTextModel:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("benchmark config must contain a mapping")
    config = Qwen3OmniMoeConfig(**raw)
    model = Qwen3OmniMoeThinkerTextModel(config).to(
        device=device,
        dtype=dtype,
    )
    return model.eval()


def _prefill_inputs(prompt_ids: torch.Tensor) -> ModelPrefillInputs:
    batch_size, sequence_length = prompt_ids.shape
    mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    position_ids = torch.arange(
        sequence_length,
        dtype=torch.long,
        device=prompt_ids.device,
    ).unsqueeze(0).expand(batch_size, -1)
    return ModelPrefillInputs(
        input_ids=prompt_ids,
        inputs_embeds=None,
        key_valid_mask=mask,
        position_batch=PositionBatch(
            position_ids=position_ids.unsqueeze(0),
            rope_deltas=torch.zeros(
                (batch_size, 1),
                dtype=torch.long,
                device=prompt_ids.device,
            ),
            axis_names=("sequence",),
        ),
    )


def _uncached_trace(
    model: Qwen3OmniMoeThinkerTextModel,
    prompt_ids: torch.Tensor,
    output_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generated = prompt_ids
    logits = []
    with torch.inference_mode():
        for _ in range(output_length):
            output = model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
            )
            current = output["logits"][:, -1]
            logits.append(current)
            token = current.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, token), dim=1)
    return torch.cat(logits, dim=0), generated[:, prompt_ids.shape[1] :]


def _cached_trace(
    model: Qwen3OmniMoeThinkerTextModel,
    prompt_ids: torch.Tensor,
    output_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    owner = StateOwner.fresh("benchmark-correctness")
    output = model.prefill(
        inputs=_prefill_inputs(prompt_ids),
        owner=owner,
        use_cache=True,
    )
    assert output.decoder_state is not None
    state = output.decoder_state
    current_logits = output.logits[:, -1]
    logits = [current_logits]
    generated = [current_logits.argmax(dim=-1, keepdim=True)]
    for _ in range(1, output_length):
        cursor = state.position.continuation
        position = PositionBatch(
            position_ids=cursor.next_storage_position.view(1, 1, 1),
            rope_deltas=state.position.cached.rope_deltas,
            axis_names=("sequence",),
        )
        output = model.decode(
            inputs=ModelDecodeInputs(
                token_ids=generated[-1],
                current_key_valid_mask=torch.ones(
                    (1, 1),
                    dtype=torch.bool,
                    device=prompt_ids.device,
                ),
                position_batch=position,
                decoder_state=state,
            ),
            owner=owner,
        )
        assert output.decoder_state is not None
        state = output.decoder_state
        current_logits = output.logits[:, -1]
        logits.append(current_logits)
        generated.append(current_logits.argmax(dim=-1, keepdim=True))
    return torch.cat(logits, dim=0), torch.cat(generated, dim=1)


def _verify_parity(
    cached_logits: torch.Tensor,
    uncached_logits: torch.Tensor,
    cached_tokens: torch.Tensor,
    uncached_tokens: torch.Tensor,
) -> tuple[float, bool]:
    if cached_logits.shape != uncached_logits.shape:
        raise AssertionError("cached/uncached logit shapes differ")
    max_abs = (
        cached_logits.float() - uncached_logits.float()
    ).abs().max().item()
    if max_abs > 1e-5:
        raise AssertionError(f"cached logits parity failed: max_abs={max_abs}")
    token_exact = torch.equal(cached_tokens, uncached_tokens)
    if not token_exact:
        raise AssertionError("cached and uncached greedy tokens differ")
    return max_abs, token_exact


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    operation: Callable[[], object],
    *,
    device: torch.device,
) -> float:
    _synchronize(device)
    started = time.perf_counter()
    operation()
    _synchronize(device)
    return time.perf_counter() - started


def _implementation_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_benchmark(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    prompt_length: int = 8,
    output_length: int = 3,
    warmup: int = 1,
    repetitions: int = 2,
    seed: int = 20260730,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    prompt_length = _strict_positive(prompt_length, "prompt_length")
    output_length = _strict_positive(output_length, "output_length")
    warmup = _strict_nonnegative(warmup, "warmup")
    repetitions = _strict_positive(repetitions, "repetitions")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    resolved_device = torch.device(device)
    if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise ValueError("dtype must be float32, bfloat16 or float16")
    torch.manual_seed(seed)
    model = _load_model(
        config_path,
        device=resolved_device,
        dtype=dtype,
    )
    if prompt_length + output_length - 1 > model.thinker_cfg.max_position_embeddings:
        raise ValueError("prompt/output lengths exceed the model context")
    prompt_ids = torch.randint(
        3,
        model.config.vocab_size,
        (1, prompt_length),
        dtype=torch.long,
        device=resolved_device,
    )

    # Correctness is deliberately completed before any warmup or timing.
    uncached_logits, uncached_tokens = _uncached_trace(
        model,
        prompt_ids,
        output_length,
    )
    cached_logits, cached_tokens = _cached_trace(
        model,
        prompt_ids,
        output_length,
    )
    max_abs, token_exact = _verify_parity(
        cached_logits,
        uncached_logits,
        cached_tokens,
        uncached_tokens,
    )

    engine = LegacyGreedyPrefillDecodeEngine(model)
    generation_request = GenerationRequest(
        display_request_id="benchmark",
        prefill_inputs=_prefill_inputs(prompt_ids),
        max_new_tokens=output_length,
        eos_token_id=None,
    )

    def cached_operation():
        return engine.generate(generation_request)

    def uncached_operation():
        return _uncached_trace(model, prompt_ids, output_length)

    for _ in range(warmup):
        cached_operation()
        uncached_operation()
    cached_samples = [
        _measure(cached_operation, device=resolved_device)
        for _ in range(repetitions)
    ]
    uncached_samples = [
        _measure(uncached_operation, device=resolved_device)
        for _ in range(repetitions)
    ]
    final_result = cached_operation()
    state = final_result.checkpoint.decoder_state

    peak_memory = 0
    memory_method = "unavailable-on-cpu"
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
        cached_operation()
        _synchronize(resolved_device)
        peak_memory = torch.cuda.max_memory_allocated(resolved_device)
        memory_method = "torch.cuda.max_memory_allocated"

    partition_names = (
        "seen_tokens",
        "position",
        "full_attention_kv",
        "swa_kv",
        "gdn_state",
        "talker_state",
        "mtp_state",
        "codec_state",
    )
    logical_by_partition = {
        name: state.logical_tensor_bytes(name)
        for name in partition_names
    }
    allocated_by_partition = {
        name: state.unique_allocated_bytes(name)
        for name in partition_names
    }
    summary = summarize_model(model, model.config.profile_manifest)
    cached_mean = statistics.fmean(cached_samples)
    uncached_mean = statistics.fmean(uncached_samples)
    return {
        "schema_version": 1,
        "architecture_manifest": model.config.profile_manifest.to_dict(),
        "architecture_summary": summary.to_dict(),
        "implementation_commit": _implementation_commit(),
        "seed": seed,
        "prompt_length": prompt_length,
        "output_length": output_length,
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(resolved_device),
        "synchronization": (
            "torch.cuda.synchronize"
            if resolved_device.type == "cuda"
            else "synchronous-cpu"
        ),
        "warmup": warmup,
        "repetitions": repetitions,
        "latency_seconds": {
            "cached_raw": cached_samples,
            "uncached_raw": uncached_samples,
            "cached_mean": cached_mean,
            "cached_median": statistics.median(cached_samples),
            "uncached_mean": uncached_mean,
            "uncached_median": statistics.median(uncached_samples),
        },
        "tokens_per_second": {
            "cached": output_length / cached_mean,
            "uncached": output_length / uncached_mean,
        },
        "peak_memory_bytes": peak_memory,
        "peak_memory_method": memory_method,
        "cache_bytes": {
            "logical_total": state.logical_tensor_bytes(),
            "unique_allocated_total": state.unique_allocated_bytes(),
            "logical_by_partition": logical_by_partition,
            "unique_allocated_by_partition": allocated_by_partition,
        },
        "max_abs": max_abs,
        "token_exact": token_exact,
        "fallback": {
            "used": False,
            "code": None,
        },
    }


def main() -> int:
    args = _parse_args()
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    result = run_benchmark(
        config_path=args.config,
        prompt_length=args.prompt_length,
        output_length=args.output_length,
        warmup=args.warmup,
        repetitions=args.repetitions,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
