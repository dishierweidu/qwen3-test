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
