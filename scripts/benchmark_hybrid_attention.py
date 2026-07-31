#!/usr/bin/env python3
"""Correctness-gated full/SWA attention benchmark for the tiny profile."""

from __future__ import annotations

import argparse
import hashlib
import statistics

import torch

from qwen3_omni_pretrain.architecture.summary import summarize_model
from qwen3_omni_pretrain.evaluation.benchmarking import (
    load_hybrid_config,
    positive_int,
    positive_int_values,
    require_passing_gates,
    resolve_dtype,
    runtime_environment,
    timing_samples,
    write_report,
)
from qwen3_omni_pretrain.evaluation.contracts import (
    BenchmarkMeasurement,
    CorrectnessGate,
    ExperimentReport,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.attention import (
    HybridSelfAttention,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.modeling_hybrid_swa_moe import (
    HybridSwaMoeForCausalLM,
)
from qwen3_omni_pretrain.utils.profiling import measure_peak_memory


def _parameter_digest(module: torch.nn.Module) -> tuple[str, str]:
    digest = hashlib.sha256()
    inventory = []
    for name, tensor in module.state_dict().items():
        inventory.append(f"{name}:{tuple(tensor.shape)}")
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        raw_bytes = (
            tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
        )
        digest.update(bytes(raw_bytes))
    return digest.hexdigest(), ",".join(inventory)


def _cached_decode(
    module: HybridSelfAttention,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    *,
    prompt_length: int,
) -> tuple[float, object, object]:
    with torch.inference_mode():
        expected, _ = module(hidden, position_ids=positions)
        prefill, prefix_cache = module(
            hidden[:, :prompt_length],
            position_ids=positions[:, :prompt_length],
            use_cache=True,
        )
        cache = prefix_cache
        pieces = [prefill]
        for index in range(prompt_length, hidden.shape[1]):
            decoded, cache = module(
                hidden[:, index : index + 1],
                position_ids=positions[:, index : index + 1],
                cache=cache,
                use_cache=True,
            )
            pieces.append(decoded)
        actual = torch.cat(pieces, dim=1)
    assert prefix_cache is not None and cache is not None
    error = float((actual.float() - expected.float()).abs().max().item())
    return error, prefix_cache, cache


def _parity_tolerance(dtype: torch.dtype) -> float:
    return 1e-5 if dtype is torch.float32 else 5e-3


def run_benchmark(args: argparse.Namespace) -> ExperimentReport:
    config = load_hybrid_config(args.config)
    device = torch.device(getattr(args, "device", "cpu"))
    dtype = resolve_dtype(getattr(args, "dtype", "float32"))
    seed = int(getattr(args, "seed", 20260729))
    repetitions = positive_int(
        int(getattr(args, "repetitions", 3)), name="repetitions"
    )
    warmup = int(getattr(args, "warmup", 1))
    prompt_lengths = positive_int_values(
        getattr(args, "prompt_lengths", (32,)), name="prompt_lengths"
    )
    output_lengths = positive_int_values(
        getattr(args, "output_lengths", (1,)), name="output_lengths"
    )
    torch.manual_seed(seed)

    paired_kv_heads = getattr(args, "paired_kv_heads", None)
    modes = tuple(getattr(args, "attention_modes", ("full", "swa")))
    if len(modes) != 2 or set(modes) != {"full", "swa"}:
        raise ValueError("attention_modes must contain full and swa exactly")
    modules: dict[str, HybridSelfAttention] = {}
    if paired_kv_heads is not None:
        kv_heads = positive_int(paired_kv_heads, name="paired_kv_heads")
        if config.num_attention_heads % kv_heads:
            raise ValueError("paired KV heads must divide query heads")
        full = HybridSelfAttention(
            config,
            layer_index=config.attention_layer_types.index("full"),
            attention_type="full",
            num_key_value_heads=kv_heads,
            rope_theta=config.swa_rope_theta,
        )
        swa = HybridSelfAttention(
            config,
            layer_index=config.attention_layer_types.index("swa"),
            attention_type="swa",
            num_key_value_heads=kv_heads,
            rope_theta=config.swa_rope_theta,
        )
        swa.load_state_dict(full.state_dict(), strict=True)
        modules = {"full": full, "swa": swa}
        copied_digest, inventory = _parameter_digest(full)
        comparison_metadata = {
            "architecture_count": 2,
            "weight_comparable": True,
            "kv_heads": kv_heads,
            "copied_parameter_digest": copied_digest,
            "copied_parameter_inventory": inventory,
            "comparison": "equal-KV-head tied attention pair",
        }
    else:
        modules = {
            "full": HybridSelfAttention(
                config,
                layer_index=config.attention_layer_types.index("full"),
            ),
            "swa": HybridSelfAttention(
                config,
                layer_index=config.attention_layer_types.index("swa"),
            ),
        }
        comparison_metadata = {
            "architecture_count": 2,
            "weight_comparable": False,
            "reason": "full and SWA KV projection shapes differ",
            "primary": "hybrid_full_attention",
            "subject": "hybrid_swa",
        }

    modules = {
        name: module.to(device=device, dtype=dtype).eval()
        for name, module in modules.items()
    }
    cases: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for prompt_length in prompt_lengths:
        for output_length in output_lengths:
            total_length = prompt_length + output_length
            if total_length > config.max_position_embeddings:
                raise ValueError(
                    "prompt_length + output_length exceeds configured context"
                )
            cases[(prompt_length, output_length)] = (
                torch.randn(
                    (1, total_length, config.hidden_size),
                    device=device,
                    dtype=dtype,
                ),
                torch.arange(
                    total_length, device=device, dtype=torch.long
                ).unsqueeze(0),
            )
    gates: list[CorrectnessGate] = []
    prefix_caches: dict[tuple[str, int, int], object] = {}
    final_caches: dict[tuple[str, int, int], object] = {}
    tolerance = _parity_tolerance(dtype)
    for (prompt_length, output_length), (hidden, positions) in cases.items():
        for name, module in modules.items():
            error, prefix_cache, final_cache = _cached_decode(
                module,
                hidden,
                positions,
                prompt_length=prompt_length,
            )
            key = (name, prompt_length, output_length)
            prefix_caches[key] = prefix_cache
            final_caches[key] = final_cache
            gates.append(
                CorrectnessGate(
                    name=(
                        f"{name}_cached_uncached_parity_"
                        f"p{prompt_length}_o{output_length}"
                    ),
                    passed=error <= tolerance,
                    tolerance=tolerance,
                    observed=error,
                )
            )
        total_length = prompt_length + output_length
        if paired_kv_heads is not None and total_length <= config.swa_window_size:
            with torch.inference_mode():
                full_output, _ = modules["full"](
                    hidden, position_ids=positions
                )
                swa_output, _ = modules["swa"](
                    hidden, position_ids=positions
                )
            pair_error = float(
                (full_output.float() - swa_output.float()).abs().max().item()
            )
            gates.append(
                CorrectnessGate(
                    name=(
                        "tied_full_swa_parity_"
                        f"p{prompt_length}_o{output_length}"
                    ),
                    passed=pair_error <= tolerance,
                    tolerance=tolerance,
                    observed=pair_error,
                )
            )
    gate_tuple = tuple(gates)
    require_passing_gates(gate_tuple)

    measurements: list[BenchmarkMeasurement] = []
    for (prompt_length, output_length), (hidden, positions) in cases.items():
        for name, module in modules.items():
            def prefill() -> torch.Tensor:
                with torch.inference_mode():
                    return module(
                        hidden[:, :prompt_length],
                        position_ids=positions[:, :prompt_length],
                    )[0]

            prefix_cache = prefix_caches[(name, prompt_length, output_length)]

            def decode() -> object:
                cache = prefix_cache
                with torch.inference_mode():
                    for index in range(
                        prompt_length, prompt_length + output_length
                    ):
                        _, cache = module(
                            hidden[:, index : index + 1],
                            position_ids=positions[:, index : index + 1],
                            cache=cache,
                            use_cache=True,
                        )
                return cache

            prefill_raw = timing_samples(
                prefill,
                repetitions=repetitions,
                warmup=warmup,
                device=device,
            )
            decode_raw = timing_samples(
                decode,
                repetitions=repetitions,
                warmup=warmup,
                device=device,
            )
            prefill_throughput = tuple(
                prompt_length / elapsed for elapsed in prefill_raw
            )
            decode_throughput = tuple(
                output_length / elapsed for elapsed in decode_raw
            )
            dimensions = {
                "batch_size": 1,
                "prompt_length": prompt_length,
                "output_length": output_length,
                "kv_heads": module.num_key_value_heads,
            }
            measurements.extend(
                (
                    BenchmarkMeasurement(
                        name=f"{name}_prefill_latency_p50",
                        value=0.0,
                        unit="s",
                        dimensions=dimensions,
                        raw_samples=prefill_raw,
                    ),
                    BenchmarkMeasurement(
                        name=f"{name}_prefill_tokens_per_second",
                        value=statistics.mean(prefill_throughput),
                        unit="tokens/s",
                        dimensions=dimensions,
                        raw_samples=prefill_throughput,
                    ),
                    BenchmarkMeasurement(
                        name=f"{name}_decode_latency_p50",
                        value=0.0,
                        unit="s",
                        dimensions=dimensions,
                        raw_samples=decode_raw,
                    ),
                    BenchmarkMeasurement(
                        name=f"{name}_decode_tokens_per_second",
                        value=statistics.mean(decode_throughput),
                        unit="tokens/s",
                        dimensions=dimensions,
                        raw_samples=decode_throughput,
                    ),
                    BenchmarkMeasurement(
                        name=f"{name}_cache_bytes",
                        value=float(
                            final_caches[
                                (name, prompt_length, output_length)
                            ].logical_tensor_bytes()
                        ),
                        unit="bytes",
                        dimensions=dimensions,
                    ),
                )
            )
    peak = measure_peak_memory(device)
    measurements.append(
        BenchmarkMeasurement(
            name="process_peak_memory",
            value=float(peak.bytes),
            unit="bytes",
            dimensions={"source": peak.source, "platform": peak.platform},
        )
    )
    model = HybridSwaMoeForCausalLM(config).to(device="meta")
    architecture = summarize_model(model, config.profile_manifest)
    return ExperimentReport(
        manifest=config.profile_manifest,
        architecture=architecture,
        environment=runtime_environment(
            device=device,
            dtype=dtype,
            seed=seed,
            kernels=("eager-matmul-softmax",),
            fallbacks=("attention sink requires eager attention",),
        ),
        router_aux_loss_weight=config.router_aux_loss_weight,
        mtp_loss_weight=config.mtp_loss_weight,
        correctness_gates=gate_tuple,
        measurements=tuple(measurements),
        comparison_metadata=comparison_metadata,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attention-modes", nargs="+", default=["full", "swa"])
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128])
    parser.add_argument("--output-lengths", nargs="+", type=int, default=[32])
    parser.add_argument("--paired-kv-heads", type=int)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    write_report(run_benchmark(parsed), parsed.output)
