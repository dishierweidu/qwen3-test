# src/qwen3_omni_pretrain/data/datasets/omni_webdataset.py

import json
import os
from typing import List, Dict, Optional

from torch.utils.data import Dataset


class OmniJsonlDataset(Dataset):
    """
    Stage2: 多模态 JSONL 数据集。
    一行一个样本，例如：
    {
      "id": "imgcap_0001",
      "task": "image_caption",
      "image": "images/cat.jpg",
      "input_text": "",
      "target_text": "一只趴在沙发上的橘猫。"
    }
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: Optional[str] = None,
        audio_root: Optional[str] = None,
    ):
        super().__init__()
        self.samples: List[Dict] = []
        self.image_root = image_root
        self.audio_root = audio_root

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # 统一处理路径为绝对路径
                image_rel = obj.get("image")
                if image_rel and self.image_root is not None:
                    obj["image_path"] = os.path.join(self.image_root, image_rel)
                else:
                    obj["image_path"] = None

                audio_rel = obj.get("audio")
                if audio_rel and self.audio_root is not None:
                    obj["audio_path"] = os.path.join(self.audio_root, audio_rel)
                else:
                    obj["audio_path"] = None

                self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
