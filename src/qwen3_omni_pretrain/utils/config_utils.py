# src/qwen3_omni_pretrain/utils/config_utils.py

import os
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return a dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dicts.
    Values in `override` take precedence over `base`.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out
