from pathlib import Path


def _pins(path: str) -> dict[str, str]:
    result = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        result[name.lower()] = version
    return result


def test_prototype_profile_has_matching_torch_family():
    pins = _pins("constraints/prototype-py310.txt")
    assert pins["torch"] == "2.10.0"
    assert pins["torchvision"] == "0.25.0"
    assert pins["torchaudio"] == "2.10.0"
    assert pins["transformers"] == "4.57.6"


def test_reference_profile_is_separate_and_qwen_capable():
    pins = _pins("constraints/qwen3-omni-reference-py310.txt")
    assert pins["torch"] == "2.10.0"
    assert pins["torchvision"] == "0.25.0"
    assert pins["torchaudio"] == "2.10.0"
    assert pins["transformers"] == "5.2.0"
    assert pins["qwen-omni-utils"] == "0.0.9"
