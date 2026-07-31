from __future__ import annotations

import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    MultiHeadSelfAttention,
    RotaryEmbedding,
    _build_rectangular_causal_bias,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text_tp import (
    TensorParallelMultiHeadSelfAttention,
)


def make_standard(*, flash: bool = False) -> tuple[
    MultiHeadSelfAttention,
    RotaryEmbedding,
]:
    torch.manual_seed(7)
    attention = MultiHeadSelfAttention(
        hidden_size=8,
        num_heads=4,
        num_kv_heads=2,
        head_dim=2,
        use_flash_attention=flash,
    ).eval()
    rope = RotaryEmbedding(2, max_position_embeddings=32)
    return attention, rope


def run_attention(
    attention,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    current_mask: torch.Tensor,
    *,
    past=None,
    rope: RotaryEmbedding,
    use_cache: bool,
):
    past_mask = None if past is None else past.key_valid_mask
    bias = _build_rectangular_causal_bias(
        past_key_valid_mask=past_mask,
        current_key_valid_mask=current_mask,
        dtype=hidden.dtype,
    )
    return attention(
        hidden,
        attention_bias=bias,
        position_ids=positions,
        rotary_emb=rope,
        past_key_value=past,
        current_key_valid_mask=current_mask,
        use_cache=use_cache,
    )


def test_rectangular_bias_uses_p_plus_q_rule_and_padding_keys():
    past = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    current = torch.tensor([[1, 1]], dtype=torch.bool)
    bias = _build_rectangular_causal_bias(
        past_key_valid_mask=past,
        current_key_valid_mask=current,
        dtype=torch.float32,
    )
    assert bias.shape == (1, 1, 2, 5)
    assert (bias[0, 0, 0] == 0).tolist() == [True, False, True, True, False]
    assert (bias[0, 0, 1] == 0).tolist() == [True, False, True, True, True]


def test_rectangular_bias_lets_a_valid_current_query_attend_to_itself():
    bias = _build_rectangular_causal_bias(
        past_key_valid_mask=torch.zeros(1, 2, dtype=torch.bool),
        current_key_valid_mask=torch.tensor([[1]], dtype=torch.bool),
        dtype=torch.float32,
    )
    assert (bias == 0).tolist() == [[[[False, False, True]]]]


@pytest.mark.parametrize("flash", [False, True])
def test_standard_attention_cached_chunks_match_full_sequence(flash):
    attention, rope = make_standard(flash=flash)
    hidden = torch.randn(2, 6, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1]],
        dtype=torch.bool,
    )
    positions = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 1, 0, 3, 4, 5]],
        dtype=torch.long,
    )
    full, no_present = run_attention(
        attention,
        hidden,
        positions,
        mask,
        rope=rope,
        use_cache=False,
    )
    assert no_present is None

    outputs = []
    present = None
    start = 0
    for width in (2, 1, 3):
        stop = start + width
        current, present = run_attention(
            attention,
            hidden[:, start:stop],
            positions[:, start:stop],
            mask[:, start:stop],
            past=present,
            rope=rope,
            use_cache=True,
        )
        outputs.append(current)
        start = stop
    cached = torch.cat(outputs, dim=1)
    assert present is not None
    assert present.key.shape == (2, 2, 6, 2)
    assert present.value.shape == (2, 2, 6, 2)
    assert torch.equal(present.key_valid_mask, mask)
    assert (cached.float() - full.float()).abs().max().item() <= 1e-5
    assert torch.equal(cached.argmax(dim=-1), full.argmax(dim=-1))


def test_attention_cache_is_rope_applied_local_unrepeated_kv():
    attention, rope = make_standard()
    hidden = torch.randn(1, 2, 8)
    positions = torch.tensor([[3, 5]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    _, present = run_attention(
        attention,
        hidden,
        positions,
        mask,
        rope=rope,
        use_cache=True,
    )
    projected = attention.k_proj(hidden).view(1, 2, 2, 2).transpose(1, 2)
    expected = rope(projected, positions)
    assert present is not None
    assert present.key.shape[1] == attention.num_kv_heads
    assert torch.allclose(present.key, expected)


def test_bfloat16_cache_preserves_dtype_and_matches_full_attention():
    attention, rope = make_standard()
    attention = attention.to(dtype=torch.bfloat16)
    hidden = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    mask = torch.ones(1, 3, dtype=torch.bool)
    full, _ = run_attention(
        attention,
        hidden,
        torch.tensor([[0, 1, 2]]),
        mask,
        rope=rope,
        use_cache=False,
    )
    first, cache = run_attention(
        attention,
        hidden[:, :2],
        torch.tensor([[0, 1]]),
        mask[:, :2],
        rope=rope,
        use_cache=True,
    )
    last, cache = run_attention(
        attention,
        hidden[:, 2:],
        torch.tensor([[2]]),
        mask[:, 2:],
        past=cache,
        rope=rope,
        use_cache=True,
    )
    assert cache.key.dtype is torch.bfloat16
    assert torch.equal(torch.cat((first, last), dim=1), full)


def test_invalid_current_query_is_zero_after_output_projection():
    attention, rope = make_standard()
    hidden = torch.randn(1, 2, 8)
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    output, present = run_attention(
        attention,
        hidden,
        torch.tensor([[0, 0]]),
        mask,
        rope=rope,
        use_cache=True,
    )
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))
    assert present is not None
    assert present.key_valid_mask.tolist() == [[True, False]]


def test_failed_append_does_not_mutate_prior_cache():
    attention, rope = make_standard()
    hidden = torch.randn(1, 1, 8)
    _, present = run_attention(
        attention,
        hidden,
        torch.tensor([[0]]),
        torch.ones(1, 1, dtype=torch.bool),
        rope=rope,
        use_cache=True,
    )
    fingerprint = present.key.clone()
    pointer = present.key.data_ptr()
    with pytest.raises(ValueError):
        run_attention(
            attention,
            torch.randn(2, 1, 8),
            torch.tensor([[1], [1]]),
            torch.ones(2, 1, dtype=torch.bool),
            past=present,
            rope=rope,
            use_cache=True,
        )
    assert present.key.data_ptr() == pointer
    assert torch.equal(present.key, fingerprint)


def test_tp_world_size_one_attention_uses_same_cache_contract(monkeypatch):
    import qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text_tp as tp

    monkeypatch.setattr(tp, "get_tensor_model_parallel_world_size", lambda: 1)
    torch.manual_seed(11)
    attention = TensorParallelMultiHeadSelfAttention(
        hidden_size=8,
        num_heads=4,
        num_kv_heads=2,
        head_dim=2,
    ).eval()
    rope = RotaryEmbedding(2, max_position_embeddings=16)
    hidden = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    full, _ = run_attention(
        attention,
        hidden,
        torch.tensor([[0, 1, 2]]),
        mask,
        rope=rope,
        use_cache=False,
    )
    first, cache = run_attention(
        attention,
        hidden[:, :2],
        torch.tensor([[0, 1]]),
        mask[:, :2],
        rope=rope,
        use_cache=True,
    )
    last, cache = run_attention(
        attention,
        hidden[:, 2:],
        torch.tensor([[2]]),
        mask[:, 2:],
        past=cache,
        rope=rope,
        use_cache=True,
    )
    assert cache.key.shape == (1, 2, 3, 2)
    assert torch.allclose(torch.cat((first, last), dim=1), full, atol=1e-5)


def _copy_standard_weights_to_tp(standard, parallel, rank: int) -> None:
    world_size = 2
    with torch.no_grad():
        vocab_chunk = standard.embed_tokens.weight.shape[0] // world_size
        parallel.embed_tokens.weight.copy_(
            standard.embed_tokens.weight[
                rank * vocab_chunk : (rank + 1) * vocab_chunk
            ]
        )
        parallel.norm.weight.copy_(standard.norm.weight)
        for source, target in zip(standard.layers, parallel.layers):
            target.attn_norm.weight.copy_(source.attn_norm.weight)
            target.mlp_norm.weight.copy_(source.mlp_norm.weight)
            for name in ("q_proj", "k_proj", "v_proj"):
                source_weight = getattr(source.self_attn, name).weight
                target_weight = getattr(target.self_attn, name).weight
                rows = target_weight.shape[0]
                target_weight.copy_(
                    source_weight[rank * rows : (rank + 1) * rows]
                )
            source_o = source.self_attn.o_proj.weight
            target_o = target.self_attn.o_proj.weight
            columns = target_o.shape[1]
            target_o.copy_(
                source_o[:, rank * columns : (rank + 1) * columns]
            )
            source_fc1 = source.shared_mlp.fc1.weight
            target_fc1 = target.shared_mlp.fc1.weight
            rows = target_fc1.shape[0]
            target_fc1.copy_(
                source_fc1[rank * rows : (rank + 1) * rows]
            )
            source_fc2 = source.shared_mlp.fc2.weight
            target_fc2 = target.shared_mlp.fc2.weight
            columns = target_fc2.shape[1]
            target_fc2.copy_(
                source_fc2[:, rank * columns : (rank + 1) * columns]
            )


def _tp_two_rank_cache_worker(
    rank: int,
    world_size: int,
    init_method: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeConfig,
        )
        from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
            Qwen3OmniMoeThinkerTextModel,
        )
        from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text_tp import (
            Qwen3OmniMoeThinkerTextModelTP,
        )
        from qwen3_omni_pretrain.multimodal.types import PositionBatch
        from qwen3_omni_pretrain.parallel import (
            destroy_model_parallel,
            initialize_model_parallel,
        )
        from qwen3_omni_pretrain.runtime import (
            ModelDecodeInputs,
            ModelPrefillInputs,
            StateOwner,
        )

        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            backend="gloo",
        )
        config = Qwen3OmniMoeConfig(
            vocab_size=32,
            thinker_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 16,
                "use_moe": False,
                "use_flash_attention": False,
                "use_tensor_parallel": True,
                "tensor_parallel_size": 2,
            },
        )
        torch.manual_seed(41)
        standard = Qwen3OmniMoeThinkerTextModel(config).eval()
        parallel = Qwen3OmniMoeThinkerTextModelTP(config).eval()
        _copy_standard_weights_to_tp(standard, parallel, rank)
        token_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]])
        mask = torch.ones_like(token_ids, dtype=torch.bool)
        all_positions = torch.arange(7).unsqueeze(0)
        reference = standard(
            input_ids=token_ids,
            attention_mask=mask,
            position_ids=all_positions,
        )["logits"]

        def position(values: torch.Tensor) -> PositionBatch:
            return PositionBatch(
                position_ids=values.unsqueeze(0),
                rope_deltas=torch.zeros(1, 1, dtype=torch.long),
                axis_names=("sequence",),
            )

        # Independent rank-local nonces are not a valid distributed request.
        # Every rank must reject before entering the embedding collective.
        embedding_calls = 0
        original_embedding = parallel.embed_tokens.forward

        def counted_mismatched_embedding(*args, **kwargs):
            nonlocal embedding_calls
            embedding_calls += 1
            return original_embedding(*args, **kwargs)

        parallel.embed_tokens.forward = counted_mismatched_embedding
        rejected = False
        try:
            parallel.prefill(
                inputs=ModelPrefillInputs(
                    input_ids=token_ids[:, :3],
                    inputs_embeds=None,
                    key_valid_mask=mask[:, :3],
                    position_batch=position(all_positions[:, :3]),
                ),
                owner=StateOwner.fresh("rank-local-owner"),
                use_cache=True,
            )
        except RuntimeError:
            rejected = True
        proof = torch.tensor(
            [int(rejected), embedding_calls],
            dtype=torch.int32,
        )
        dist.all_reduce(proof)
        assert proof.tolist() == [2, 0]
        parallel.embed_tokens.forward = original_embedding

        owner = parallel.create_state_owner("tp-parity")
        gathered_owner_nonces = [
            torch.empty(16, dtype=torch.int32)
            for _ in range(world_size)
        ]
        dist.all_gather(
            gathered_owner_nonces,
            torch.tensor(tuple(owner.nonce), dtype=torch.int32),
        )
        assert all(
            torch.equal(gathered_owner_nonces[0], candidate)
            for candidate in gathered_owner_nonces[1:]
        )

        first = parallel.prefill(
            inputs=ModelPrefillInputs(
                input_ids=token_ids[:, :3],
                inputs_embeds=None,
                key_valid_mask=mask[:, :3],
                position_batch=position(all_positions[:, :3]),
            ),
            owner=owner,
            use_cache=True,
        )
        assert first.decoder_state is not None
        assert all(
            cache.key.shape[1] == 1
            for cache in first.decoder_state.full_attention_kv.values()
        )
        second = parallel.decode(
            inputs=ModelDecodeInputs(
                token_ids=token_ids[:, 3:4],
                current_key_valid_mask=mask[:, 3:4],
                position_batch=position(all_positions[:, 3:4]),
                decoder_state=first.decoder_state,
            ),
            owner=owner,
        )
        assert second.decoder_state is not None
        third = parallel.decode(
            inputs=ModelDecodeInputs(
                token_ids=token_ids[:, 4:],
                current_key_valid_mask=mask[:, 4:],
                position_batch=position(all_positions[:, 4:]),
                decoder_state=second.decoder_state,
            ),
            owner=owner,
        )
        cached = torch.cat((first.logits, second.logits, third.logits), dim=1)
        assert (cached.float() - reference.float()).abs().max().item() <= 1e-5
        assert torch.equal(cached.argmax(dim=-1), reference.argmax(dim=-1))

        # A rank-local owner error must be shared before the peer enters the
        # vocabulary-parallel embedding collective.
        embedding_calls = 0
        original_embedding = parallel.embed_tokens.forward

        def counted_embedding(*args, **kwargs):
            nonlocal embedding_calls
            embedding_calls += 1
            return original_embedding(*args, **kwargs)

        parallel.embed_tokens.forward = counted_embedding
        rejected = False
        try:
            parallel.decode(
                inputs=ModelDecodeInputs(
                    token_ids=torch.tensor([[10]]),
                    current_key_valid_mask=torch.ones(1, 1, dtype=torch.bool),
                    position_batch=position(torch.tensor([[7]])),
                    decoder_state=third.decoder_state,
                ),
                owner=(
                    StateOwner.fresh("wrong-owner")
                    if rank == 0
                    else owner
                ),
            )
        except (RuntimeError, ValueError):
            rejected = True
        proof = torch.tensor(
            [int(rejected), embedding_calls],
            dtype=torch.int32,
        )
        dist.all_reduce(proof)
        assert proof.tolist() == [2, 0]
        destroy_model_parallel()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _join_without_hanging(context, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if context.join(timeout=0.5):
            return True
    for process in context.processes:
        process.terminate()
    for process in context.processes:
        process.join(timeout=2)
    return False


def test_tp_two_rank_local_shard_cache_matches_standard_model(tmp_path):
    if not dist.is_available():
        pytest.fail("torch.distributed is required for the TP cache gate")
    context = mp.spawn(
        _tp_two_rank_cache_worker,
        args=(2, f"file://{tmp_path / 'tp-cache-rendezvous'}"),
        nprocs=2,
        join=False,
    )
    assert _join_without_hanging(context, timeout=30), (
        "2-rank TP cache validation deadlocked"
    )
