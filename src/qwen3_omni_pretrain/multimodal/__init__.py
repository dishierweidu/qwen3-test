from typing import TYPE_CHECKING, Any

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.types import (
    AssembledSequence,
    MediaGrid,
    MediaSequence,
    MediaSource,
    PositionBatch,
    SequenceSpan,
    SequenceSpanKind,
)


if TYPE_CHECKING:
    from qwen3_omni_pretrain.multimodal.prefill import (
        MultimodalPrefillOutput,
        MultimodalPrefillPipeline,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "MultimodalPrefillOutput",
        "MultimodalPrefillPipeline",
    }:
        from qwen3_omni_pretrain.multimodal.prefill import (
            MultimodalPrefillOutput,
            MultimodalPrefillPipeline,
        )

        exports = {
            "MultimodalPrefillOutput": MultimodalPrefillOutput,
            "MultimodalPrefillPipeline": MultimodalPrefillPipeline,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AssembledSequence",
    "MediaGrid",
    "MediaModality",
    "MediaSequence",
    "MediaSource",
    "MultimodalPrefillOutput",
    "MultimodalPrefillPipeline",
    "PositionBatch",
    "SequenceSpan",
    "SequenceSpanKind",
]
