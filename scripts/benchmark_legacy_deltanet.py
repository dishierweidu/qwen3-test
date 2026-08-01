#!/usr/bin/env python3
"""Independent, non-weight-comparable legacy DeltaNet engineering baseline."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
import yaml

from qwen3_omni_pretrain.architecture.summary import summarize_model
from qwen3_omni_pretrain.evaluation.benchmarking import (
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
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.profiles.legacy_prototype.config_adapter import (
    adapt_legacy_config_dict,
)
from qwen3_omni_pretrain.utils.profiling import measure_peak_memory


def _load_config(path_value: str) -> Qwen3OmniMoeConfig:
    with Path(path_value).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("legacy DeltaNet config must contain a mapping")
    return Qwen3OmniMoeConfig(**adapt_legacy_config_dict(raw))


def run_benchmark(args: argparse.Namespace) -> ExperimentReport:
    config = _load_config(args.config)
    device = torch.device(getattr(args, "device", "cpu"))
    dtype = resolve_dtype(getattr(args, "dtype", "float32"))
    seed = int(getattr(args, "seed", 20260729))
    repetitions = positive_int(
        int(getattr(args, "repetitions", 2)), name="repetitions"
    )
    warmup = int(getattr(args, "warmup", 1))
    lengths = positive_int_values(
        getattr(args, "sequence_lengths", (16,)), name="sequence_lengths"
    )
    if max(lengths) > config.max_position_embeddings:
        raise ValueError("sequence length exceeds legacy context")
    torch.manual_seed(seed)
    model = Qwen3OmniMoeThinkerTextModel(config).to(
        device=device, dtype=dtype
    )
    inputs = {
        sequence_length: torch.randint(
            0, config.vocab_size, (1, sequence_length), device=device
        )
        for sequence_length in lengths
    }
    gates: list[CorrectnessGate] = []
    tolerance = 1e-5 if dtype is torch.float32 else 5e-3
    for sequence_length, input_ids in inputs.items():
        model.zero_grad(set_to_none=True)
        model.train()
        training = model(input_ids=input_ids, labels=input_ids)
        finite_forward = bool(
            torch.isfinite(training["logits"]).all().item()
            and training["loss"] is not None
            and torch.isfinite(training["loss"]).item()
        )
        training["loss"].backward()
        finite_backward = all(
            parameter.grad is None
            or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        model.eval()
        prefix_length = max(1, sequence_length // 2)
        with torch.inference_mode():
            full_logits = model(input_ids=input_ids)["logits"][
                :, :prefix_length
            ]
            prefix_logits = model(
                input_ids=input_ids[:, :prefix_length]
            )["logits"]
            repeated_logits = model(input_ids=input_ids)["logits"]
            repeated_again = model(input_ids=input_ids)["logits"]
        prefix_error = float(
            (full_logits.float() - prefix_logits.float()).abs().max().item()
        )
        repeat_error = float(
            (repeated_logits.float() - repeated_again.float()).abs().max().item()
        )
        gates.extend(
            (
                CorrectnessGate(
                    f"finite_forward_backward_t{sequence_length}",
                    finite_forward and finite_backward,
                    None,
                    None,
                ),
                CorrectnessGate(
                    f"causal_prefix_invariance_t{sequence_length}",
                    prefix_error <= tolerance,
                    tolerance,
                    prefix_error,
                ),
                CorrectnessGate(
                    f"seeded_repeatability_t{sequence_length}",
                    repeat_error == 0.0,
                    0.0,
                    repeat_error,
                ),
            )
        )
    gate_tuple = tuple(gates)
    require_passing_gates(gate_tuple)

    peak = measure_peak_memory(device)
    summary = summarize_model(model, config.profile_manifest)
    measurements: list[BenchmarkMeasurement] = [
        BenchmarkMeasurement(
            "total_parameters",
            float(summary.total_parameters),
            "parameters",
            {"model_type": summary.model_type},
        ),
        BenchmarkMeasurement(
            "active_parameters",
            float(summary.active_parameters_per_token),
            "parameters/token",
            {"model_type": summary.model_type},
        ),
    ]
    for sequence_length, input_ids in inputs.items():
        def forward(ids: torch.Tensor = input_ids) -> torch.Tensor:
            with torch.inference_mode():
                return model(input_ids=ids)["logits"]

        raw = timing_samples(
            forward,
            repetitions=repetitions,
            warmup=warmup,
            device=device,
        )
        throughput = tuple(sequence_length / elapsed for elapsed in raw)
        measurements.extend((
        BenchmarkMeasurement(
            "legacy_deltanet_latency_p50",
            0.0,
            "s",
            {"sequence_length": sequence_length, "layers": len(model.layers)},
            raw,
        ),
        BenchmarkMeasurement(
            "legacy_deltanet_tokens_per_second",
            statistics.mean(throughput),
            "tokens/s",
            {"sequence_length": sequence_length},
            throughput,
        ),
        ))
    measurements.append(
        BenchmarkMeasurement(
            "process_peak_memory",
            float(peak.bytes),
            "bytes",
            {"source": peak.source, "platform": peak.platform},
        )
    )
    return ExperimentReport(
        manifest=config.profile_manifest,
        architecture=summary,
        environment=runtime_environment(
            device=device,
            dtype=dtype,
            seed=seed,
            kernels=("legacy-eager-deltanet-recurrence",),
            fallbacks=("different class and independently initialized weights",),
        ),
        router_aux_loss_weight=0.0,
        mtp_loss_weight=0.0,
        correctness_gates=gate_tuple,
        measurements=tuple(measurements),
        comparison_metadata={
            "architecture_count": 2,
            "primary": "mimo_v25_experimental",
            "subject": "legacy_deltanet",
            "weight_comparable": False,
            "reason": "different model class and independently initialized weights",
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    write_report(run_benchmark(parsed), parsed.output)
