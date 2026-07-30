from __future__ import annotations

import math
from numbers import Real

import torch


def quantize_timestamps_half_up(
    timestamps: torch.Tensor,
    seconds_per_bucket: float,
) -> torch.LongTensor:
    if not isinstance(timestamps, torch.Tensor):
        raise TypeError("timestamps must be a torch.Tensor")
    if not timestamps.is_floating_point():
        raise TypeError(
            "timestamps must have a floating non-complex dtype"
        )
    if isinstance(seconds_per_bucket, bool) or not isinstance(
        seconds_per_bucket,
        Real,
    ):
        raise TypeError("seconds_per_bucket must be a real number")
    step_value = float(seconds_per_bucket)
    if not math.isfinite(step_value) or step_value <= 0:
        raise ValueError(
            "seconds_per_bucket must be finite and positive"
        )
    if not bool(torch.isfinite(timestamps).all().item()):
        raise ValueError("timestamps must contain finite values")
    if bool((timestamps < 0).any().item()):
        raise ValueError("timestamps must be non-negative")

    values = timestamps.to(dtype=torch.float32)
    step = torch.tensor(
        step_value,
        dtype=torch.float32,
        device=timestamps.device,
    )
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(
            "timestamps must remain finite after float32 conversion"
        )
    if not bool(torch.isfinite(step).item()) or float(step.item()) <= 0:
        raise ValueError(
            "seconds_per_bucket must remain finite and positive "
            "after float32 conversion"
        )
    half = torch.tensor(
        0.5,
        dtype=torch.float32,
        device=timestamps.device,
    )
    bucket_values = torch.floor(values / step + half)
    int64_upper_bound = torch.tensor(
        2**63,
        dtype=torch.float32,
        device=timestamps.device,
    )
    if (
        not bool(torch.isfinite(bucket_values).all().item())
        or bool((bucket_values >= int64_upper_bound).any().item())
    ):
        raise ValueError(
            "quantized bucket values must fit in non-negative int64"
        )
    return bucket_values.to(dtype=torch.long)


__all__ = ["quantize_timestamps_half_up"]
