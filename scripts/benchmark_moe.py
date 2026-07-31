#!/usr/bin/env python3
"""Correctness-gated routed SwiGLU versus matched-active dense benchmark."""

from __future__ import annotations

import argparse
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
from qwen3_omni_pretrain.models.hybrid_swa_moe.modeling_hybrid_swa_moe import (
    HybridSwaMoeForCausalLM,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.moe import (
    RoutedSwiGLUMoE,
    SwiGLU,
)
from qwen3_omni_pretrain.utils.model_stats import collect_parameter_stats
from qwen3_omni_pretrain.utils.profiling import measure_peak_memory


def run_benchmark(args: argparse.Namespace) -> ExperimentReport:
    config = load_hybrid_config(args.config)
    device = torch.device(getattr(args, "device", "cpu"))
    dtype = resolve_dtype(getattr(args, "dtype", "float32"))
    seed = int(getattr(args, "seed", 20260729))
    repetitions = positive_int(
        int(getattr(args, "repetitions", 3)), name="repetitions"
    )
    warmup = int(getattr(args, "warmup", 1))
    sequence_lengths = positive_int_values(
        getattr(args, "sequence_lengths", (32,)), name="sequence_lengths"
    )
    experts = positive_int(
        int(getattr(args, "experts", config.num_experts)), name="experts"
    )
    top_k = positive_int(
        int(getattr(args, "top_k", config.num_experts_per_token)),
        name="top_k",
    )
    if top_k >= experts:
        raise ValueError("MoE benchmark requires top_k < experts")
    if getattr(args, "baseline", "dense") != "dense":
        raise ValueError("MoE benchmark baseline must be dense")
    torch.manual_seed(seed)
    routed = RoutedSwiGLUMoE(
        config.hidden_size,
        config.expert_intermediate_size,
        experts,
        top_k,
    ).to(device=device, dtype=dtype)
    # One dense SwiGLU with k-times width matches the selected expert parameter
    # count (router parameters remain separately observable).
    dense = SwiGLU(
        config.hidden_size,
        config.expert_intermediate_size * top_k,
    ).to(device=device, dtype=dtype)
    gates: list[CorrectnessGate] = []
    cases: dict[int, tuple[torch.Tensor, object]] = {}
    for sequence_length in sequence_lengths:
        routed.zero_grad(set_to_none=True)
        dense.zero_grad(set_to_none=True)
        hidden = torch.randn(
            (1, sequence_length, config.hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        routed_output = routed(hidden, collect_stats=True)
        weight_error = float(
            (routed_output.selected_weights.sum(dim=-1) - 1.0)
            .abs()
            .max()
            .item()
        )
        finite_forward = bool(
            torch.isfinite(routed_output.hidden_states).all().item()
            and torch.isfinite(routed_output.aux_loss).item()
        )
        routed_output.hidden_states.float().square().mean().backward()
        selected_experts = set(
            routed_output.selected_expert_ids.flatten().tolist()
        )
        finite_backward = bool(
            hidden.grad is not None
            and torch.isfinite(hidden.grad).all().item()
            and routed.router.weight.grad is not None
            and torch.isfinite(routed.router.weight.grad).all().item()
            and all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for expert_id in selected_experts
                for parameter in routed.experts[expert_id].parameters()
            )
        )
        stats = routed_output.stats
        assert stats is not None
        count_error = abs(
            int(stats.expert_token_counts.sum().item())
            - sequence_length * top_k
        )
        gates.extend(
            (
                CorrectnessGate(
                    f"selected_weight_normalization_t{sequence_length}",
                    weight_error <= 1e-6,
                    1e-6,
                    weight_error,
                ),
                CorrectnessGate(
                    f"finite_forward_backward_t{sequence_length}",
                    finite_forward and finite_backward,
                    None,
                    None,
                ),
                CorrectnessGate(
                    f"route_count_t{sequence_length}",
                    count_error == 0,
                    0.0,
                    float(count_error),
                ),
            )
        )
        cases[sequence_length] = (hidden.detach(), stats)
    gate_tuple = tuple(gates)
    require_passing_gates(gate_tuple)

    routed_parameters = collect_parameter_stats(routed)
    dense_parameters = sum(parameter.numel() for parameter in dense.parameters())
    measurements: list[BenchmarkMeasurement] = [
        BenchmarkMeasurement(
            "routed_total_parameters",
            float(routed_parameters.total_parameters),
            "parameters",
            {"experts": experts},
        ),
        BenchmarkMeasurement(
            "routed_active_parameters",
            float(routed_parameters.estimated_active_parameters_per_token),
            "parameters/token",
            {"top_k": top_k},
        ),
        BenchmarkMeasurement(
            "matched_dense_parameters",
            float(dense_parameters),
            "parameters",
            {"top_k": top_k},
        ),
    ]
    for sequence_length, (benchmark_hidden, stats_value) in cases.items():
        routed_raw = timing_samples(
            lambda hidden=benchmark_hidden: routed(
                hidden, collect_stats=False
            ).hidden_states,
            repetitions=repetitions,
            warmup=warmup,
            device=device,
        )
        dense_raw = timing_samples(
            lambda hidden=benchmark_hidden: dense(hidden),
            repetitions=repetitions,
            warmup=warmup,
            device=device,
        )
        routed_throughput = tuple(
            sequence_length / value for value in routed_raw
        )
        dense_throughput = tuple(
            sequence_length / value for value in dense_raw
        )
        measurements.extend((
        BenchmarkMeasurement(
            "routed_moe_latency_p50",
            0.0,
            "s",
            {"experts": experts, "top_k": top_k, "tokens": sequence_length},
            routed_raw,
        ),
        BenchmarkMeasurement(
            "dense_latency_p50",
            0.0,
            "s",
            {"intermediate_size": config.expert_intermediate_size * top_k, "tokens": sequence_length},
            dense_raw,
        ),
        BenchmarkMeasurement(
            "routed_tokens_per_second",
            statistics.mean(routed_throughput),
            "tokens/s",
            {
                "experts": experts,
                "top_k": top_k,
                "tokens": sequence_length,
            },
            routed_throughput,
        ),
        BenchmarkMeasurement(
            "dense_tokens_per_second",
            statistics.mean(dense_throughput),
            "tokens/s",
            {"top_k": top_k, "tokens": sequence_length},
            dense_throughput,
        ),
        BenchmarkMeasurement(
            "router_entropy",
            float(stats_value.router_entropy.item()),
            "nats",
            {"tokens": sequence_length},
        ),
        BenchmarkMeasurement(
            "max_mean_load",
            float(stats_value.max_mean_load.item()),
            "ratio",
            {"tokens": sequence_length},
        ),
        ))
        for expert_id, count in enumerate(
            stats_value.expert_token_counts.tolist()
        ):
            measurements.append(
                BenchmarkMeasurement(
                    "expert_token_count",
                    float(count),
                    "routes",
                    {"expert_id": expert_id, "tokens": sequence_length},
                )
            )
    peak = measure_peak_memory(device)
    measurements.append(
        BenchmarkMeasurement(
            "process_peak_memory",
            float(peak.bytes),
            "bytes",
            {"source": peak.source, "platform": peak.platform},
        )
    )
    with torch.device("meta"):
        summary_model = HybridSwaMoeForCausalLM(config)
    return ExperimentReport(
        manifest=config.profile_manifest,
        architecture=summarize_model(summary_model, config.profile_manifest),
        environment=runtime_environment(
            device=device,
            dtype=dtype,
            seed=seed,
            kernels=("eager-topk-index-add",),
            fallbacks=("Python expert loop",),
        ),
        router_aux_loss_weight=config.router_aux_loss_weight,
        mtp_loss_weight=config.mtp_loss_weight,
        correctness_gates=gate_tuple,
        measurements=tuple(measurements),
        comparison_metadata={
            "architecture_count": 2,
            "weight_comparable": False,
            "reason": "dense and routed FFN parameterizations differ",
            "primary": "routed_swiglu_moe",
            "subject": "matched_active_dense_swiglu",
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[128])
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
