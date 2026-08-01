from __future__ import annotations

import torch

from qwen3_omni_pretrain.models.hybrid_swa_moe import (
    HybridSwaMoeConfig,
    HybridSwaMoeForCausalLM,
    RoutedSwiGLUMoE,
    SwiGLU,
)
from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime.protocols import (
    CacheCapableModel,
    ModelDecodeInputs,
    ModelPrefillInputs,
)
from qwen3_omni_pretrain.runtime.state import StateOwner
from tests.hybrid_swa_moe.test_attention import attention_config


def _config(
    *,
    router_weight: float = 0.01,
    mtp_weight: float = 0.1,
) -> HybridSwaMoeConfig:
    raw = attention_config(window_size=4).to_dict()
    raw["router_aux_loss_weight"] = router_weight
    raw["mtp_loss_weight"] = mtp_weight
    return HybridSwaMoeConfig(**raw)


def _positions(values: torch.Tensor) -> PositionBatch:
    return PositionBatch(
        position_ids=values.unsqueeze(0),
        rope_deltas=torch.zeros(
            (values.shape[0], 1), dtype=torch.long, device=values.device
        ),
        axis_names=("sequence",),
    )


def test_model_tree_has_dense_layer_zero_and_routed_only_later_layers():
    model = HybridSwaMoeForCausalLM(_config())
    assert isinstance(model.layers[0].ffn, SwiGLU)
    for layer in model.layers[1:]:
        assert isinstance(layer.ffn, RoutedSwiGLUMoE)
        assert not hasattr(layer, "shared_mlp")
        assert not any("shared" in name for name, _ in layer.ffn.named_modules())


def test_complete_forward_backward_returns_reserved_keys_and_owned_cache():
    torch.manual_seed(37)
    model = HybridSwaMoeForCausalLM(_config())
    input_ids = torch.randint(0, model.vocab_size, (2, 7))
    labels = input_ids.clone()
    output = model(
        input_ids,
        labels=labels,
        request_id="train-request",
        use_cache=True,
        collect_router_stats=True,
    )

    assert set(output) == {
        "logits",
        "loss",
        "ce_loss",
        "aux_loss",
        "mtp_loss",
        "router_stats",
        "decoder_state",
    }
    assert output["logits"].shape == (2, 7, model.vocab_size)
    for name in ("loss", "ce_loss", "aux_loss", "mtp_loss"):
        assert output[name] is not None and torch.isfinite(output[name])
    assert len(output["router_stats"]) == 5
    state = output["decoder_state"]
    assert state.owner.display_request_id == "train-request"
    assert tuple(state.full_attention_kv) == (5,)
    assert tuple(state.swa_kv) == (0, 1, 2, 3, 4)
    assert all(cache.sequence_length <= 4 for cache in state.swa_kv.values())

    output["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.embed_tokens.weight.grad is not None


def test_zero_loss_weights_are_exact_omission_branches():
    torch.manual_seed(41)
    model = HybridSwaMoeForCausalLM(
        _config(router_weight=0.0, mtp_weight=0.0)
    )
    input_ids = torch.randint(0, model.vocab_size, (1, 6))
    output = model(input_ids, labels=input_ids)

    assert output["loss"] is output["ce_loss"]
    assert output["aux_loss"] is not None
    assert output["mtp_loss"] is not None


def test_each_weighted_loss_term_matches_exact_equation():
    torch.manual_seed(43)
    model = HybridSwaMoeForCausalLM(
        _config(router_weight=0.02, mtp_weight=0.3)
    )
    input_ids = torch.randint(0, model.vocab_size, (1, 6))
    output = model(input_ids, labels=input_ids)
    expected = (
        output["ce_loss"]
        + 0.02 * output["aux_loss"]
        + 0.3 * output["mtp_loss"]
    )
    torch.testing.assert_close(output["loss"], expected, rtol=0, atol=0)


def test_typed_prefill_decode_matches_one_shot_and_preserves_owner():
    torch.manual_seed(47)
    model = HybridSwaMoeForCausalLM(_config()).eval()
    assert isinstance(model, CacheCapableModel)
    owner = StateOwner.fresh("typed-request")
    all_ids = torch.randint(0, model.vocab_size, (2, 7))
    all_positions = torch.arange(7).expand(2, -1)
    mask = torch.ones((2, 7), dtype=torch.bool)
    expected = model(
        all_ids,
        attention_mask=mask,
        position_ids=all_positions,
    )["logits"]

    prefill = model.prefill(
        inputs=ModelPrefillInputs(
            input_ids=all_ids[:, :5],
            inputs_embeds=None,
            key_valid_mask=mask[:, :5],
            position_batch=_positions(all_positions[:, :5]),
        ),
        owner=owner,
        use_cache=True,
    )
    assert prefill.decoder_state is not None
    decoded = model.decode(
        inputs=ModelDecodeInputs(
            token_ids=all_ids[:, 5:],
            current_key_valid_mask=mask[:, 5:],
            position_batch=_positions(all_positions[:, 5:]),
            decoder_state=prefill.decoder_state,
        ),
        owner=owner,
    )

    torch.testing.assert_close(
        decoded.logits, expected[:, 5:], atol=1e-5, rtol=1e-5
    )
    assert decoded.decoder_state is not None
    decoded.decoder_state.assert_owner(owner)
    assert decoded.decoder_state.seen_tokens.tolist() == [7, 7]


def test_base_logits_do_not_depend_on_labels_or_mtp_loss_weight():
    torch.manual_seed(53)
    first = HybridSwaMoeForCausalLM(_config(mtp_weight=0.0))
    second = HybridSwaMoeForCausalLM(_config(mtp_weight=0.7))
    second.load_state_dict(first.state_dict())
    input_ids = torch.randint(0, first.vocab_size, (1, 6))

    first_output = first(input_ids, labels=input_ids)
    second_output = second(input_ids, labels=input_ids)
    no_labels = first(input_ids)

    torch.testing.assert_close(first_output["logits"], second_output["logits"])
    torch.testing.assert_close(first_output["logits"], no_labels["logits"])
