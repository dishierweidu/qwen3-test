from __future__ import annotations

from unittest.mock import patch

import torch

from qwen3_omni_pretrain.models.hybrid_swa_moe import HybridSelfAttention
from qwen3_omni_pretrain.runtime.state import AttentionKV, SlidingWindowKV
from tests.hybrid_swa_moe.test_attention import attention_config


def _chunked_forward(
    module: HybridSelfAttention,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    chunks: tuple[int, ...],
):
    outputs = []
    cache = None
    start = 0
    for length in chunks:
        end = start + length
        output, cache = module(
            hidden[:, start:end],
            position_ids=positions[:, start:end],
            current_key_valid_mask=mask[:, start:end],
            cache=cache,
            use_cache=True,
        )
        outputs.append(output)
        if isinstance(cache, SlidingWindowKV):
            assert cache.sequence_length <= module.window_size
            assert bool(
                (cache.key_valid_mask.sum(dim=1) <= module.window_size)
                .all()
                .item()
            )
        start = end
    assert start == hidden.shape[1]
    return torch.cat(outputs, dim=1), cache


def test_irregular_chunked_swa_matches_one_shot_including_chunk_larger_than_window():
    torch.manual_seed(17)
    module = HybridSelfAttention(attention_config(window_size=4), layer_index=0)
    hidden = torch.randn(2, 13, 32)
    positions = torch.arange(13).expand(2, -1)
    mask = torch.ones((2, 13), dtype=torch.bool)

    one_shot, one_cache = module(
        hidden,
        position_ids=positions,
        current_key_valid_mask=mask,
        use_cache=True,
    )
    chunked, chunk_cache = _chunked_forward(
        module, hidden, positions, mask, (2, 7, 1, 3)
    )

    torch.testing.assert_close(chunked, one_shot, atol=1e-5, rtol=1e-5)
    assert isinstance(one_cache, SlidingWindowKV)
    assert isinstance(chunk_cache, SlidingWindowKV)
    torch.testing.assert_close(chunk_cache.key, one_cache.key)
    torch.testing.assert_close(chunk_cache.value, one_cache.value)
    assert torch.equal(chunk_cache.key_valid_mask, one_cache.key_valid_mask)
    assert torch.equal(
        chunk_cache.position.position_ids,
        one_cache.position.position_ids,
    )


def test_padded_rows_compact_independently_across_chunks():
    torch.manual_seed(19)
    module = HybridSelfAttention(attention_config(window_size=4), layer_index=0)
    hidden = torch.randn(2, 9, 32)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    positions = torch.tensor(
        [
            list(range(9)),
            [0, 1, 0, 2, 0, 3, 4, 0, 5],
        ],
        dtype=torch.long,
    )

    expected, _ = module(
        hidden,
        position_ids=positions,
        current_key_valid_mask=mask,
    )
    actual, cache = _chunked_forward(
        module, hidden, positions, mask, (3, 2, 4)
    )

    torch.testing.assert_close(
        actual.masked_select(mask.unsqueeze(-1)),
        expected.masked_select(mask.unsqueeze(-1)),
        atol=1e-5,
        rtol=1e-5,
    )
    assert isinstance(cache, SlidingWindowKV)
    assert cache.key_valid_mask.sum(dim=1).tolist() == [4, 4]
    assert cache.key_valid_mask.tolist() == [[True] * 4, [True] * 4]
    assert cache.position.position_ids[0, 1].tolist() == [2, 3, 4, 5]


def test_swa_cache_stays_bounded_through_four_windows_of_decode():
    torch.manual_seed(23)
    module = HybridSelfAttention(attention_config(window_size=4), layer_index=0)
    cache = None
    for position in range(17):
        _, cache = module(
            torch.randn(1, 1, 32),
            position_ids=torch.tensor([[position]]),
            cache=cache,
            use_cache=True,
        )
        assert isinstance(cache, SlidingWindowKV)
        assert cache.sequence_length <= 4
    assert cache.sequence_length == 4
    assert cache.position.position_ids.flatten().tolist() == [13, 14, 15, 16]
    assert cache.logical_tensor_bytes() > 0


def test_full_attention_cache_grows_and_cached_output_matches_one_shot():
    torch.manual_seed(29)
    module = HybridSelfAttention(attention_config(), layer_index=5)
    hidden = torch.randn(2, 9, 32)
    positions = torch.arange(9).expand(2, -1)
    mask = torch.ones((2, 9), dtype=torch.bool)

    expected, _ = module(hidden, position_ids=positions)
    actual, cache = _chunked_forward(
        module, hidden, positions, mask, (4, 2, 3)
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert isinstance(cache, AttentionKV)
    assert cache.sequence_length == 9


def test_one_token_cached_decode_never_allocates_square_attention_scores():
    torch.manual_seed(31)
    module = HybridSelfAttention(attention_config(window_size=8), layer_index=0)
    _, cache = module(
        torch.randn(1, 8, 32),
        position_ids=torch.arange(8).unsqueeze(0),
        use_cache=True,
    )
    matmul_shapes: list[tuple[torch.Size, torch.Size]] = []
    original_matmul = torch.matmul

    def recording_matmul(left, right):
        matmul_shapes.append((left.shape, right.shape))
        return original_matmul(left, right)

    with patch("torch.matmul", side_effect=recording_matmul):
        module(
            torch.randn(1, 1, 32),
            position_ids=torch.tensor([[8]]),
            cache=cache,
            use_cache=True,
        )

    score_products = [
        shapes for shapes in matmul_shapes if len(shapes[0]) == 4
    ]
    assert score_products
    assert all(left[-2] == 1 for left, _ in score_products)
