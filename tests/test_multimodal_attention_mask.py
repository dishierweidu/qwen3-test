import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe import modeling_thinker_vision_audio as module


def test_multimodal_mask_preserves_text_padding_and_modality_presence():
    text_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
    has_image = torch.tensor([1, 0], dtype=torch.long)
    has_audio = torch.tensor([0, 1], dtype=torch.long)

    result = module._build_multimodal_attention_mask(
        text_mask, has_image, has_audio
    )

    assert result.tolist() == [
        [1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0],
    ]


def test_nonfinite_modality_features_raise_instead_of_being_sanitized():
    values = torch.tensor([[[float("nan")]]])
    try:
        module._require_finite_tensor(values, "vision features")
    except FloatingPointError as exc:
        assert "vision features" in str(exc)
    else:
        raise AssertionError("expected FloatingPointError")


def test_wrapper_passes_composed_attention_and_labels_to_thinker(monkeypatch):
    class CapturingThinker(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, 4)
            self.received = None

        def forward(self, **kwargs):
            self.received = kwargs
            return kwargs

    monkeypatch.setattr(
        module, "Qwen3OmniMoeThinkerTextModel", CapturingThinker
    )
    config = module.Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": False,
            "moe_shared_expert": False,
        },
    )
    model = module.Qwen3OmniMoeThinkerVisionAudioModel(config)

    output = model(
        input_ids=torch.tensor([[3, 4, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.long),
        labels=torch.tensor([[-100, 4, -100]], dtype=torch.long),
        pixel_values=torch.zeros(1, 3, 8, 8),
        audio_values=torch.zeros(1, 32000),
        has_image=torch.tensor([0], dtype=torch.long),
        has_audio=torch.tensor([1], dtype=torch.long),
    )

    assert output["attention_mask"].tolist() == [[0, 1, 1, 1, 0]]
    assert output["labels"].tolist() == [[-100, -100, -100, 4, -100]]
    assert output["inputs_embeds"].shape == (1, 5, 4)
