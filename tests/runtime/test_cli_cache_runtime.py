from __future__ import annotations

import json

import pytest
import torch

from qwen3_omni_pretrain import cli_infer_thinker as cli
from qwen3_omni_pretrain.data.collators import Stage2MediaLoader
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_vision_audio import (
    Qwen3OmniMoeThinkerVisionAudioModel,
)
from qwen3_omni_pretrain.runtime import CacheCapabilityError, CacheErrorCode


class Tokenizer:
    eos_token_id = None

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        return {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded"


def config(*, hybrid: bool = False) -> Qwen3OmniMoeConfig:
    return Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4 if hybrid else 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32,
            "use_moe": False,
            "use_flash_attention": False,
            "use_deltanet": hybrid,
            "deltanet_layer_indices": None,
            "deltanet_num_heads": 4,
        },
    )


class CountingCacheModel(torch.nn.Module):
    def __init__(self, base) -> None:
        super().__init__()
        self.base = base
        self.config = base.config
        self.prefill_calls = 0
        self.decode_calls = 0
        if hasattr(base, "thinker"):
            self.thinker = base.thinker
        else:
            self.layers = base.layers

    def prefill(self, **kwargs):
        self.prefill_calls += 1
        return self.base.prefill(**kwargs)

    def decode(self, **kwargs):
        self.decode_calls += 1
        return self.base.decode(**kwargs)


def test_stage1_cli_uses_one_prefill_then_pending_token_decode():
    wrapped = CountingCacheModel(
        Qwen3OmniMoeThinkerTextModel(config()).eval()
    ).eval()
    result = cli.greedy_decode_stage1(
        model=wrapped,
        tokenizer=Tokenizer(),
        device=torch.device("cpu"),
        prompt="hello",
        max_new_tokens=3,
    )
    assert wrapped.prefill_calls == 1
    assert wrapped.decode_calls == 2
    assert result["stats"]["output_tokens"] == 3


def test_stage2_cli_passes_media_only_to_prefill_and_uses_cache():
    wrapped = CountingCacheModel(
        Qwen3OmniMoeThinkerVisionAudioModel(config()).eval()
    ).eval()
    result = cli.greedy_decode_stage2(
        model=wrapped,
        tokenizer=Tokenizer(),
        media_loader=Stage2MediaLoader(image_size=2, max_audio_len=4),
        device=torch.device("cpu"),
        sample={"id": "stage2-cache", "input_text": "hello"},
        max_new_tokens=3,
        image_root="",
        audio_root="",
        max_seq_length=8,
    )
    assert wrapped.prefill_calls == 1
    assert wrapped.decode_calls == 2
    assert result["stats"]["output_tokens"] == 3
    assert result["media_errors"] == []


def test_hybrid_requires_explicit_fallback_before_tokenizer():
    model = Qwen3OmniMoeThinkerTextModel(config(hybrid=True)).eval()
    tokenizer = Tokenizer()
    with pytest.raises(CacheCapabilityError) as captured:
        cli.greedy_decode_stage1(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            prompt="delta",
            max_new_tokens=1,
        )
    assert captured.value.code is CacheErrorCode.DELTA_NET_UNSUPPORTED
    assert tokenizer.calls == 0


def test_explicit_hybrid_fallback_emits_exactly_one_structured_warning(
    capsys,
):
    model = Qwen3OmniMoeThinkerTextModel(config(hybrid=True)).eval()
    result = cli.greedy_decode_stage1(
        model=model,
        tokenizer=Tokenizer(),
        device=torch.device("cpu"),
        prompt="delta",
        max_new_tokens=2,
        allow_uncached_fallback=True,
    )
    warnings = [
        json.loads(line)["cache_warning"]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"cache_warning"')
    ]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "DELTA_NET_UNSUPPORTED"
    assert warnings[0]["profile"] == "legacy_prototype"
    assert warnings[0]["display_request_id"] == "delta"
    assert warnings[0]["semantic_change"] is False
    assert result["stats"]["output_tokens"] == 2


def test_beam_rejection_precedes_tokenizer_and_model():
    model = Qwen3OmniMoeThinkerTextModel(config()).eval()
    tokenizer = Tokenizer()
    with pytest.raises(CacheCapabilityError) as captured:
        cli.greedy_decode_stage1(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            prompt="beam",
            max_new_tokens=1,
            num_beams=2,
        )
    assert captured.value.code is CacheErrorCode.BEAM_UNSUPPORTED
    assert tokenizer.calls == 0


@pytest.mark.parametrize("runner", [cli.run_stage1, cli.run_stage2])
def test_cli_entrypoint_rejects_beam_before_bootstrap(monkeypatch, runner):
    bootstrap_calls = []
    monkeypatch.setattr(
        cli,
        "_ensure_tokenizer",
        lambda *_args, **_kwargs: bootstrap_calls.append("tokenizer"),
    )
    with pytest.raises(CacheCapabilityError) as captured:
        runner(type("Args", (), {"num_beams": 2})())
    assert captured.value.code is CacheErrorCode.BEAM_UNSUPPORTED
    assert bootstrap_calls == []
