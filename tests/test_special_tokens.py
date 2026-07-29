from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.multimodal.tokenization import (
    special_tokens as subject,
)


TOKEN_IDS = {
    "<|image_pad|>": 1,
    "<|video_pad|>": 2,
    "<|audio_pad|>": 3,
    "<|audio_start|>": 4,
    "<|audio_end|>": 5,
}
EXPECTED_IDS = {
    "image_token_id": 1,
    "video_token_id": 2,
    "audio_token_id": 3,
    "audio_start_token_id": 4,
    "audio_end_token_id": 5,
}


class FakeTokenizer:
    def __init__(self, vocab=None, length=6):
        self._vocab = dict(TOKEN_IDS if vocab is None else vocab)
        self._length = int(length)

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self):
        return self._length


def tiny_config(**kwargs):
    return Qwen3OmniMoeConfig(
        vocab_size=16,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        **kwargs,
    )


def test_happy_path_uses_tokenizer_ids_without_resizing_vocab():
    config = tiny_config()
    before_vocab_size = config.vocab_size

    resolved = subject.reconcile_multimodal_token_ids(
        config, FakeTokenizer()
    )

    assert resolved == EXPECTED_IDS
    assert {
        field: getattr(config, field) for field in EXPECTED_IDS
    } == EXPECTED_IDS
    assert config.vocab_size == before_vocab_size


def test_reconciled_ids_survive_config_round_trip(tmp_path):
    config = tiny_config()
    subject.reconcile_multimodal_token_ids(config, FakeTokenizer())

    config.save_pretrained(tmp_path)
    reloaded = Qwen3OmniMoeConfig.from_pretrained(
        tmp_path, local_files_only=True
    )

    assert {
        field: getattr(reloaded, field) for field in EXPECTED_IDS
    } == EXPECTED_IDS
    assert reloaded.vocab_size == 16


def test_missing_token_is_rejected_without_partial_mutation():
    config = tiny_config(image_token_id=9)
    tokenizer = FakeTokenizer({"<|image_pad|>": 1})

    with pytest.raises(ValueError, match="missing required multimodal token"):
        subject.reconcile_multimodal_token_ids(config, tokenizer)

    assert config.image_token_id == 9


def test_duplicate_ids_are_rejected():
    tokenizer = FakeTokenizer(
        {token: 1 for token in TOKEN_IDS},
        length=2,
    )
    with pytest.raises(ValueError, match="distinct IDs"):
        subject.reconcile_multimodal_token_ids(tiny_config(), tokenizer)


def test_out_of_range_id_is_rejected():
    vocab = dict(TOKEN_IDS)
    vocab["<|audio_end|>"] = 16
    with pytest.raises(ValueError, match="outside model vocab_size"):
        subject.reconcile_multimodal_token_ids(
            tiny_config(), FakeTokenizer(vocab, length=6)
        )


def test_tokenizer_larger_than_embedding_is_rejected():
    with pytest.raises(ValueError, match="exceeds model vocab_size"):
        subject.reconcile_multimodal_token_ids(
            tiny_config(), FakeTokenizer(length=17)
        )


def test_legacy_mismatch_warns_and_tokenizer_wins():
    config = tiny_config(image_token_id=9)
    with pytest.warns(RuntimeWarning, match="image_token_id"):
        resolved = subject.reconcile_multimodal_token_ids(
            config, FakeTokenizer()
        )
    assert config.image_token_id == resolved["image_token_id"] == 1


def test_padded_embedding_vocab_remains_valid():
    config = tiny_config()
    subject.reconcile_multimodal_token_ids(
        config, FakeTokenizer(length=6)
    )
    assert config.vocab_size == 16
