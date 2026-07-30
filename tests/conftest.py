from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import shutil
import sys
import types

import pytest
import torch


try:
    import transformers  # noqa: F401
except ImportError:
    transformers = types.ModuleType("transformers")

    class PretrainedConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self):
            return dict(self.__dict__)

    class PreTrainedModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            return None

    transformers.PretrainedConfig = PretrainedConfig
    transformers.PreTrainedModel = PreTrainedModel
    sys.modules["transformers"] = transformers


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [3 + (ord(ch) % 50) for ch in text]

    def __call__(
        self,
        texts,
        *,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=True,
    ):
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        encoded = [self._encode(text) for text in text_list]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        if return_tensors == "pt":
            width = max((len(ids) for ids in encoded), default=0) if padding else max_length
            if width is None:
                width = max((len(ids) for ids in encoded), default=0)
            input_ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
            for i, ids in enumerate(encoded):
                n = min(len(ids), width)
                if n:
                    input_ids[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
                    attention_mask[i, :n] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        result_ids = encoded[0] if single else encoded
        return {"input_ids": result_ids}


REFERENCE_DISTRIBUTION_VERSIONS = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
    "transformers": "5.2.0",
    "qwen-omni-utils": "0.0.9",
}
TORCH_DISTRIBUTIONS = frozenset({"torch", "torchvision", "torchaudio"})


def pytest_addoption(parser):
    parser.addoption(
        "--run-large-model-tests",
        action="store_true",
        default=False,
        help="run tests that require locally cached large model weights",
    )


def _reference_environment_issue() -> str | None:
    for distribution, expected in REFERENCE_DISTRIBUTION_VERSIONS.items():
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            return f"{distribution} is not installed"
        comparable = (
            actual.partition("+")[0]
            if distribution in TORCH_DISTRIBUTIONS
            else actual
        )
        if comparable != expected:
            return (
                f"{distribution}=={actual}; reference tests require "
                f"{distribution}=={expected}"
            )
    if shutil.which("ffmpeg") is None:
        return "ffmpeg is not available in PATH"
    return None


def pytest_collection_modifyitems(config, items):
    reference_issue = _reference_environment_issue()
    run_large = config.getoption("--run-large-model-tests")
    for item in items:
        if item.get_closest_marker("reference") and reference_issue is not None:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"official reference environment unavailable: "
                    f"{reference_issue}"
                )
            )
        if item.get_closest_marker("large_model") and not run_large:
            item.add_marker(
                pytest.mark.skip(
                    reason="requires explicit --run-large-model-tests"
                )
            )
