from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from qwen3_omni_pretrain.architecture import checkpoint_metadata as metadata_module
from qwen3_omni_pretrain.architecture.checkpoint_metadata import (
    CheckpointMetadata,
    load_checkpoint_metadata,
    write_checkpoint_metadata,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_vision_audio import (
    Qwen3OmniMoeThinkerVisionAudioModel,
    SimpleVisionEncoder,
)
from qwen3_omni_pretrain.training import checkpoint, trainer_thinker


class _SerializedBackend:
    def to_str(self) -> str:
        return json.dumps(
            {
                "model": {
                    "type": "WordLevel",
                    "vocab": {str(index): index for index in range(5)},
                },
                "pre_tokenizer": {"type": "Whitespace"},
            }
        )


class _Tokenizer:
    backend_tokenizer = _SerializedBackend()
    additional_special_tokens_ids: list[int] = []

    def __len__(self) -> int:
        return 5


def _config() -> Qwen3OmniMoeConfig:
    return Qwen3OmniMoeConfig(
        vocab_size=8,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 8,
            "use_moe": False,
        },
    )


def _metadata(
    model: torch.nn.Module,
    artifact_kind: str,
) -> CheckpointMetadata:
    return trainer_thinker._build_checkpoint_metadata(
        model,
        _Tokenizer(),
        artifact_kind=artifact_kind,
        implementation_commit="d" * 40,
    )


def test_stage2_metadata_has_distinct_kind_and_complete_wrapper_topology():
    config = _config()
    stage1 = _metadata(
        Qwen3OmniMoeThinkerTextModel(config),
        "stage1-training",
    )
    stage2 = _metadata(
        Qwen3OmniMoeThinkerVisionAudioModel(config),
        "stage2-training",
    )

    assert stage1.artifact_kind.value == "stage1-training"
    assert stage2.artifact_kind.value == "stage2-training"
    assert stage1.architecture == stage2.architecture
    assert stage1.topology != stage2.topology
    assert stage2.topology.model_class.endswith(
        "Qwen3OmniMoeThinkerVisionAudioModel"
    )
    assert stage2.topology.component_types["vision_encoder"].endswith(
        "SimpleVisionEncoder"
    )
    assert stage2.topology.component_types["audio_encoder"].endswith(
        "SimpleAudioEncoder"
    )


def test_metadata_round_trip_preserves_artifact_and_topology(tmp_path: Path):
    expected = _metadata(
        Qwen3OmniMoeThinkerVisionAudioModel(_config()),
        "stage2-training",
    )

    write_checkpoint_metadata(tmp_path, expected)

    assert load_checkpoint_metadata(tmp_path) == expected
    raw = json.loads(
        (tmp_path / "architecture.json").read_text(encoding="utf-8")
    )
    assert raw["schema_version"] == 2
    assert raw["artifact_kind"] == "stage2-training"
    assert raw["topology"]["state_schema_sha256"] == (
        expected.topology.state_schema_sha256
    )


def test_stage2_identity_rejects_stage1_before_tensor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    stage1_model = Qwen3OmniMoeThinkerTextModel(_config())
    stage1 = _metadata(stage1_model, "stage1-training")
    write_checkpoint_metadata(tmp_path, stage1)
    torch.save(stage1_model.state_dict(), tmp_path / "pytorch_model.bin")
    stage2 = _metadata(
        Qwen3OmniMoeThinkerVisionAudioModel(_config()),
        "stage2-training",
    )
    monkeypatch.setattr(
        checkpoint.torch,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "stage mismatch must fail before tensor reads"
        ),
    )

    with pytest.raises(ValueError, match="artifact kind"):
        checkpoint.load_checkpoint(
            str(tmp_path),
            Qwen3OmniMoeThinkerVisionAudioModel(_config()),
            expected_profile=stage2.manifest.architecture_profile,
            expected_compatibility=stage2.manifest.compatibility_level,
            expected_architecture=stage2.architecture,
            expected_artifact_kind=stage2.artifact_kind,
            expected_topology=stage2.topology,
            expected_tokenizer_sha256=stage2.tokenizer_sha256,
        )


def test_stage2_identity_rejects_altered_vision_topology(tmp_path: Path):
    saved_model = Qwen3OmniMoeThinkerVisionAudioModel(_config())
    saved = _metadata(saved_model, "stage2-training")
    write_checkpoint_metadata(tmp_path, saved)

    expected_model = Qwen3OmniMoeThinkerVisionAudioModel(_config())
    expected_model.vision_encoder = SimpleVisionEncoder(
        hidden_size=8,
        image_size=112,
    )
    expected = _metadata(expected_model, "stage2-training")

    with pytest.raises(ValueError, match="topology"):
        load_checkpoint_metadata(
            tmp_path,
            expected_profile=expected.manifest.architecture_profile,
            expected_compatibility=expected.manifest.compatibility_level,
            expected_architecture=expected.architecture,
            expected_artifact_kind=expected.artifact_kind,
            expected_topology=expected.topology,
            expected_tokenizer_sha256=expected.tokenizer_sha256,
        )


def test_same_stage_model_restore_is_strict(tmp_path: Path):
    model = Qwen3OmniMoeThinkerTextModel(_config())
    metadata = _metadata(model, "stage1-training")
    write_checkpoint_metadata(tmp_path, metadata)
    incomplete = dict(model.state_dict())
    incomplete.pop(next(iter(incomplete)))
    torch.save(incomplete, tmp_path / "pytorch_model.bin")

    with pytest.raises(RuntimeError, match="Missing key"):
        checkpoint.load_checkpoint(
            str(tmp_path),
            Qwen3OmniMoeThinkerTextModel(_config()),
            expected_profile=metadata.manifest.architecture_profile,
            expected_compatibility=metadata.manifest.compatibility_level,
            expected_architecture=metadata.architecture,
            expected_artifact_kind=metadata.artifact_kind,
            expected_topology=metadata.topology,
            expected_tokenizer_sha256=metadata.tokenizer_sha256,
        )


def test_metadata_uses_actual_tokenizer_length_not_embedding_capacity():
    metadata = _metadata(
        Qwen3OmniMoeThinkerTextModel(_config()),
        "stage1-training",
    )

    assert metadata.architecture.tokenizer_vocab_size == 5
    assert metadata.architecture.embedding_vocab_size == 8


def test_topology_descriptor_is_allocation_free_on_meta_device():
    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerVisionAudioModel(_config())

    topology = metadata_module.describe_model_topology(model)

    assert topology.state_schema_sha256
    assert all(parameter.is_meta for parameter in model.parameters())


def test_topology_identity_is_independent_of_runtime_precision():
    full_precision = Qwen3OmniMoeThinkerTextModel(_config())
    half_precision = Qwen3OmniMoeThinkerTextModel(_config()).half()

    assert metadata_module.describe_model_topology(
        full_precision
    ) == metadata_module.describe_model_topology(half_precision)


def test_profile_manifest_supports_transformers_deepcopy_and_safe_round_trip(
    tmp_path: Path,
):
    config = _config()

    copied = deepcopy(config)
    model = Qwen3OmniMoeThinkerTextModel(config)
    model.save_pretrained(tmp_path, safe_serialization=True)
    _, loading_info = Qwen3OmniMoeThinkerTextModel.from_pretrained(
        tmp_path,
        config=config,
        output_loading_info=True,
    )

    assert copied.profile_manifest == config.profile_manifest
    assert not {
        key: value
        for key, value in loading_info.items()
        if value
    }
