from __future__ import annotations

import pytest
import torch

from qwen3_omni_pretrain.models.hybrid_swa_moe.mtp import MultiTokenPredictor


def tiny_mtp(*, num_predictors: int = 2, vocab_size: int = 16):
    return MultiTokenPredictor(
        hidden_size=8,
        vocab_size=vocab_size,
        num_predictors=num_predictors,
    )


def test_predictor_j_targets_offset_j_plus_two():
    predictor = tiny_mtp(num_predictors=2, vocab_size=16)
    hidden = torch.randn(1, 6, 8)
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = predictor(hidden, labels=labels)

    assert output.target_offsets == (2, 3)
    assert output.valid_token_counts == (4, 3)
    assert [item.shape[1] for item in output.logits] == [4, 3]
    assert len(output.losses) == 2
    assert output.loss is not None and torch.isfinite(output.loss)


@pytest.mark.parametrize("num_predictors", [1, 2, 3])
def test_predictors_have_independent_parameters_and_finite_gradients(
    num_predictors,
):
    predictor = tiny_mtp(num_predictors=num_predictors)
    hidden = torch.randn(2, 8, 8, requires_grad=True)
    labels = torch.randint(0, 16, (2, 8))
    output = predictor(hidden, labels=labels)
    assert output.loss is not None
    output.loss.backward()

    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    parameter_ids = [
        {id(parameter) for parameter in head.parameters()}
        for head in predictor.predictors
    ]
    assert all(
        parameter_ids[left].isdisjoint(parameter_ids[right])
        for left in range(num_predictors)
        for right in range(left + 1, num_predictors)
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in predictor.parameters()
    )


def test_short_and_all_ignored_sequences_have_no_loss():
    predictor = tiny_mtp(num_predictors=3)
    short = predictor(
        torch.randn(1, 2, 8), labels=torch.tensor([[1, 2]])
    )
    assert short.loss is None
    assert short.valid_token_counts == (0, 0, 0)

    ignored = predictor(
        torch.randn(1, 6, 8),
        labels=torch.full((1, 6), -100, dtype=torch.long),
    )
    assert ignored.loss is None
    assert ignored.valid_token_counts == (0, 0, 0)


def test_padding_is_normalized_by_each_predictors_valid_count():
    predictor = tiny_mtp(num_predictors=2)
    hidden = torch.randn(1, 7, 8)
    labels = torch.tensor([[1, 2, 3, 4, -100, -100, -100]])
    output = predictor(hidden, labels=labels)
    assert output.valid_token_counts == (2, 1)
    assert len(output.losses) == 2
