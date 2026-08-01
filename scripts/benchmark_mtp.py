#!/usr/bin/env python3
"""Correctness-gated MTP training-head and verification microbenchmark."""

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
from qwen3_omni_pretrain.generation.speculative import (
    DraftProposal,
    SpeculativeStepOutput,
    verify_greedy_proposal,
    verify_sampled_proposal,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.modeling_hybrid_swa_moe import (
    HybridSwaMoeForCausalLM,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.mtp import MultiTokenPredictor
from qwen3_omni_pretrain.runtime.state import DecoderState, StateOwner
from qwen3_omni_pretrain.utils.profiling import measure_peak_memory


def _one_hot(token: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    value = torch.zeros(vocab_size, device=device)
    value[token] = 1.0
    return value


def _replay(*, state: DecoderState, token_ids: torch.Tensor) -> DecoderState:
    return state.advance_seen_tokens(
        torch.tensor(
            [token_ids.numel()], dtype=torch.long, device=state.device
        )
    )


def _greedy_case(
    *,
    natural: bool,
    vocab_size: int,
    state: DecoderState,
    draft_length: int,
) -> SpeculativeStepOutput:
    target_tokens = tuple(
        (3 + index) % vocab_size for index in range(draft_length + 1)
    )
    draft_tokens = list(target_tokens[:-1])
    if not natural:
        draft_tokens[0] = (draft_tokens[0] + 1) % vocab_size
    proposal = DraftProposal(
        token_ids=torch.tensor(
            draft_tokens, dtype=torch.long, device=state.device
        ),
        probabilities=tuple(
            _one_hot(token, vocab_size, state.device)
            for token in draft_tokens
        ),
        state=state.advance_seen_tokens(
            torch.tensor(
                [draft_length], dtype=torch.long, device=state.device
            )
        ),
    )
    return verify_greedy_proposal(
        proposal=proposal,
        target_probabilities=tuple(
            _one_hot(token, vocab_size, state.device)
            for token in target_tokens
        ),
        committed_state=state,
        replay=_replay,
    )


def _sampled_distribution_error(*, seed: int, trials: int = 4096) -> float:
    """Measure the corrected first-token marginal against a fixed target."""

    generator = torch.Generator().manual_seed(seed)
    target = torch.tensor([0.25, 0.75], dtype=torch.float32)
    draft = torch.tensor([0.50, 0.50], dtype=torch.float32)
    state = DecoderState.empty(
        StateOwner.fresh("mtp-distribution-gate"),
        batch_size=1,
        device="cpu",
    )
    counts = torch.zeros(2, dtype=torch.long)
    for _ in range(trials):
        token = int(torch.multinomial(draft, 1, generator=generator).item())
        proposal = DraftProposal(
            token_ids=torch.tensor([token], dtype=torch.long),
            probabilities=(draft,),
            state=state.advance_seen_tokens(torch.tensor([1])),
        )
        output = verify_sampled_proposal(
            proposal=proposal,
            target_probabilities=(target, target),
            committed_state=state,
            replay=_replay,
            generator=generator,
        )
        counts[int(output.committed_token_ids[0].item())] += 1
    observed = counts.to(torch.float64) / trials
    return float((observed - target.to(torch.float64)).abs().max().item())


def run_benchmark(args: argparse.Namespace) -> ExperimentReport:
    config = load_hybrid_config(args.config)
    if config.mtp_num_predictors != 1:
        raise ValueError("D6 MTP benchmark keeps exactly one reviewed predictor")
    device = torch.device(getattr(args, "device", "cpu"))
    dtype = resolve_dtype(getattr(args, "dtype", "float32"))
    seed = int(getattr(args, "seed", 20260729))
    repetitions = positive_int(
        int(getattr(args, "repetitions", 3)), name="repetitions"
    )
    warmup = int(getattr(args, "warmup", 1))
    sequence_length = positive_int(
        int(getattr(args, "sequence_length", 16)), name="sequence_length"
    )
    if sequence_length <= 2:
        raise ValueError("sequence_length must exceed the next-2 target offset")
    batch_sizes = positive_int_values(
        getattr(args, "batch_sizes", (1,)), name="batch_sizes"
    )
    draft_lengths = positive_int_values(
        getattr(args, "draft_lengths", (1,)), name="draft_lengths"
    )
    prompt_sources = tuple(getattr(args, "prompt_sources", ("natural", "random")))
    if len(prompt_sources) != 2 or set(prompt_sources) != {"natural", "random"}:
        raise ValueError("prompt_sources must contain natural and random")
    torch.manual_seed(seed)
    predictor = MultiTokenPredictor(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_predictors=config.mtp_num_predictors,
        rms_norm_eps=config.rms_norm_eps,
    ).to(device=device, dtype=dtype)
    gates: list[CorrectnessGate] = []
    predictor_cases: dict[int, torch.Tensor] = {}
    for batch_size in batch_sizes:
        predictor.zero_grad(set_to_none=True)
        hidden = torch.randn(
            (batch_size, sequence_length, config.hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        labels = torch.randint(
            0,
            config.vocab_size,
            (batch_size, sequence_length),
            device=device,
        )
        output = predictor(hidden, labels=labels)
        offset_error = abs(output.target_offsets[0] - 2)
        expected_count = batch_size * (sequence_length - 2)
        count_error = abs(output.valid_token_counts[0] - expected_count)
        finite = output.loss is not None and bool(
            torch.isfinite(output.loss).item()
        )
        if output.loss is not None:
            output.loss.backward()
        finite_gradients = bool(
            hidden.grad is not None
            and torch.isfinite(hidden.grad).all().item()
            and all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in predictor.parameters()
            )
        )
        gates.extend(
            (
                CorrectnessGate(
                    f"mtp_offset_j_plus_two_b{batch_size}",
                    offset_error == 0,
                    0.0,
                    float(offset_error),
                ),
                CorrectnessGate(
                    f"mtp_valid_token_count_b{batch_size}",
                    count_error == 0,
                    0.0,
                    float(count_error),
                ),
                CorrectnessGate(
                    f"mtp_finite_forward_backward_b{batch_size}",
                    finite and finite_gradients,
                    None,
                    None,
                ),
            )
        )
        predictor_cases[batch_size] = hidden.detach()

    owner = StateOwner.fresh("mtp-benchmark")
    state = DecoderState.empty(owner, batch_size=1, device=device)
    verification_cases: dict[tuple[str, int], SpeculativeStepOutput] = {}
    for source in prompt_sources:
        for draft_length in draft_lengths:
            result = _greedy_case(
                natural=source == "natural",
                vocab_size=config.vocab_size,
                state=state,
                draft_length=draft_length,
            )
            expected = (
                result.accepted == draft_length
                and result.rejected == 0
                and result.committed_token_ids.numel() == draft_length + 1
                if source == "natural"
                else result.accepted == 0
                and result.rejected == 1
                and result.committed_token_ids.numel() == 1
            )
            expected_seen = result.committed_token_ids.numel()
            state_repaired = result.decoder_state.seen_tokens.tolist() == [
                expected_seen
            ]
            gates.append(
                CorrectnessGate(
                    f"greedy_target_parity_{source}_d{draft_length}",
                    expected and state_repaired,
                    0.0,
                    0.0 if expected and state_repaired else 1.0,
                )
            )
            verification_cases[(source, draft_length)] = result

    distribution_error = _sampled_distribution_error(seed=seed)
    gates.append(
        CorrectnessGate(
            "sampled_distribution_target_parity",
            distribution_error <= 0.03,
            0.03,
            distribution_error,
        )
    )
    gate_tuple = tuple(gates)
    require_passing_gates(gate_tuple)

    measurements: list[BenchmarkMeasurement] = [
        BenchmarkMeasurement(
            "sampled_distribution_max_error",
            distribution_error,
            "probability",
            {"trials": 4096, "target_token_1_probability": 0.75},
        )
    ]
    for batch_size, benchmark_hidden in predictor_cases.items():
        predictor_raw = timing_samples(
            lambda hidden=benchmark_hidden: predictor(hidden, labels=None),
            repetitions=repetitions,
            warmup=warmup,
            device=device,
        )
        predictor_throughput = tuple(
            (batch_size * sequence_length) / value
            for value in predictor_raw
        )
        measurements.extend(
            (
                BenchmarkMeasurement(
                    "mtp_predictor_latency_p50",
                    0.0,
                    "s",
                    {
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "predictors": 1,
                    },
                    predictor_raw,
                ),
                BenchmarkMeasurement(
                    "mtp_predictor_tokens_per_second",
                    statistics.mean(predictor_throughput),
                    "tokens/s",
                    {
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                    },
                    predictor_throughput,
                ),
            )
        )

    for batch_size in batch_sizes:
        for source in prompt_sources:
            for draft_length in draft_lengths:
                result = verification_cases[(source, draft_length)]
                ordinary_token = torch.tensor(
                    [3], dtype=torch.long, device=device
                )

                def ordinary(
                    *,
                    rows: int = batch_size,
                    committed: int = result.committed_token_ids.numel(),
                ) -> DecoderState:
                    current = state
                    for _ in range(rows * committed):
                        current = _replay(
                            state=current, token_ids=ordinary_token
                        )
                    return current

                def speculative(
                    *,
                    rows: int = batch_size,
                    natural: bool = source == "natural",
                    length: int = draft_length,
                ) -> SpeculativeStepOutput:
                    output = result
                    for _ in range(rows):
                        output = _greedy_case(
                            natural=natural,
                            vocab_size=config.vocab_size,
                            state=state,
                            draft_length=length,
                        )
                    return output

                ordinary_raw = timing_samples(
                    ordinary,
                    repetitions=repetitions,
                    warmup=warmup,
                    device=device,
                )
                speculative_raw = timing_samples(
                    speculative,
                    repetitions=repetitions,
                    warmup=warmup,
                    device=device,
                )
                speedup_samples = tuple(
                    ordinary_value / speculative_value
                    for ordinary_value, speculative_value in zip(
                        ordinary_raw, speculative_raw
                    )
                )
                dimensions = {
                    "scope": "verification-only-synthetic",
                    "prompt_source": source,
                    "draft_length": draft_length,
                    "batch_size": batch_size,
                    "real_end_to_end_claim": False,
                }
                measurements.extend(
                    (
                        BenchmarkMeasurement(
                            "verification_latency_p50",
                            0.0,
                            "s",
                            dimensions,
                            speculative_raw,
                        ),
                        BenchmarkMeasurement(
                            "end_to_end_speedup",
                            statistics.mean(speedup_samples),
                            "ratio",
                            dimensions,
                            speedup_samples,
                        ),
                        BenchmarkMeasurement(
                            "proposed_tokens",
                            float(result.proposed),
                            "tokens/request",
                            dimensions,
                        ),
                        BenchmarkMeasurement(
                            "accepted_tokens",
                            float(result.accepted),
                            "tokens/request",
                            dimensions,
                        ),
                        BenchmarkMeasurement(
                            "rejected_tokens",
                            float(result.rejected),
                            "tokens/request",
                            dimensions,
                        ),
                        BenchmarkMeasurement(
                            "acceptance_length",
                            float(result.accepted),
                            "tokens/request",
                            dimensions,
                        ),
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
            kernels=("teacher-forced-independent-head", "corrected-speculative-verifier"),
            fallbacks=("verification-only synthetic timing; no quality claim",),
        ),
        router_aux_loss_weight=config.router_aux_loss_weight,
        mtp_loss_weight=config.mtp_loss_weight,
        correctness_gates=gate_tuple,
        measurements=tuple(measurements),
        comparison_metadata={
            "architecture_count": 1,
            "weight_comparable": True,
            "implemented_predictors": config.mtp_num_predictors,
            "prompt_sources": ",".join(prompt_sources),
            "draft_lengths": ",".join(map(str, draft_lengths)),
            "batch_sizes": ",".join(map(str, batch_sizes)),
            "scope": "mechanism-correctness-microbenchmark",
            "real_end_to_end_speedup_measured": False,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-sources", nargs="+", default=["natural", "random"])
    parser.add_argument("--draft-lengths", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--sequence-length", type=int, default=128)
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
