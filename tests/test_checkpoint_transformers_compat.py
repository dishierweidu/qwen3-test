import importlib


def test_checkpoint_helpers_import_with_supported_transformers():
    checkpoint = importlib.import_module(
        "qwen3_omni_pretrain.training.checkpoint"
    )
    assert callable(checkpoint.load_sharded_checkpoint)
