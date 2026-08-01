from __future__ import annotations

import json
from pathlib import Path


def test_pinned_public_qwen35_config_contract():
    contract = json.loads(
        Path("tests/fixtures/qwen35/public_config_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["revision"] == (
        "59d61f3ce65a6d9863b86d2e96597125219dc754"
    )
    assert contract["vocab_size"] == 248_320
    assert contract["max_position_embeddings"] == 262_144
    assert contract["num_hidden_layers"] == 40
    assert contract["full_attention_interval"] == 4
    assert contract["num_experts"] == 256
    assert contract["num_experts_per_tok"] == 8
    assert contract["qk_norm"] is True
    assert contract["attn_output_gate"] is True
    assert contract["hidden_act"] == "silu"


def test_public_layer_pattern_is_three_linear_then_one_full():
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    ]
    assert all(
        layer_types[start : start + 4]
        == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        for start in range(0, 40, 4)
    )
