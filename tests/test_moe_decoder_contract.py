from pathlib import Path


def test_decoder_layer_keeps_only_one_shared_dense_path():
    source = Path(
        "src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py"
    ).read_text(encoding="utf-8")

    assert "self.shared_mlp = MLP" in source
    assert "mlp_out = shared_out + moe_out" in source

    moe_source = Path(
        "src/qwen3_omni_pretrain/models/qwen3_omni_moe/modules/moe.py"
    ).read_text(encoding="utf-8")
    assert "self.shared_expert = None" in moe_source
    assert "y_flat = y_flat + shared_out" not in moe_source
