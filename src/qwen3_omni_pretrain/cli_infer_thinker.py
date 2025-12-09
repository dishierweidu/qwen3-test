import argparse
import json
import os
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer

from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_vision_audio import (
    Qwen3OmniMoeThinkerVisionAudioModel,
)
from qwen3_omni_pretrain.data.collators import OmniStage2Collator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick inference to sanity-check Stage1/Stage2 Thinker checkpoints",
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2"],
        required=True,
        help="Which Thinker variant to run",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved checkpoint directory (the folder that has config.json / pytorch_model*).",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="Tokenizer name or path. Defaults to the checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Ad-hoc prompt for Stage1 (text) or Stage2 (input_text). If omitted, will sample from --jsonl.",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Optional jsonl for batch-ish sampling. Stage1 expects {\"text\"}. Stage2 expects fields similar to stage2_omni_*.jsonl.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Max samples to preview from jsonl when --prompt is not set.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max tokens to autoregressively generate.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Optional dtype hint when loading the model.",
    )
    parser.add_argument(
        "--image_root",
        default="data",
        help="Base dir for Stage2 image files (used when jsonl has relative paths).",
    )
    parser.add_argument(
        "--audio_root",
        default="data",
        help="Base dir for Stage2 audio files (used when jsonl has relative paths).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Tokenization max length for prompts.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enable interactive chat mode (stdin loop).",
    )
    return parser.parse_args()


def _resolve_path(maybe_path: Optional[str], root: str) -> Optional[str]:
    if not maybe_path:
        return None
    if os.path.isabs(maybe_path):
        return maybe_path
    candidate = os.path.join(root, maybe_path)
    return candidate


def _load_jsonl(path: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path:
        return rows
    if not os.path.exists(path):
        raise FileNotFoundError(f"jsonl not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
                if len(rows) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    return rows


def _get_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str)


def _ensure_tokenizer(tokenizer_name_or_path: Optional[str], checkpoint: str):
    path = tokenizer_name_or_path or checkpoint
    # fix_mistral_regex helps avoid known tokenizer regex bug for some checkpoints
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def greedy_decode_stage1(
    model: Qwen3OmniMoeThinkerTextModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    model.eval()
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

    generated = input_ids
    input_len = generated.size(1)
    eos_id = tokenizer.eos_token_id

    import time
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=generated,
                attention_mask=attention_mask,
                labels=None,
            )
            logits = out["logits"]
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.ones_like(generated, device=device)
            if eos_id is not None and next_token.item() == eos_id:
                break

    elapsed = time.perf_counter() - t_start
    new_tokens = generated[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    stats = {
        "input_tokens": input_len,
        "output_tokens": new_tokens.numel(),
        "total_tokens": input_len + new_tokens.numel(),
        "latency_sec": elapsed,
        "output_toks_per_sec": (new_tokens.numel() / elapsed) if elapsed > 0 else float("inf"),
        "total_toks_per_sec": ((input_len + new_tokens.numel()) / elapsed) if elapsed > 0 else float("inf"),
    }
    return {"text": completion, "stats": stats}


def greedy_decode_stage2(
    model: Qwen3OmniMoeThinkerVisionAudioModel,
    tokenizer: AutoTokenizer,
    collator: OmniStage2Collator,
    device: torch.device,
    sample: Dict[str, Any],
    max_new_tokens: int,
    image_root: str,
    audio_root: str,
    max_seq_length: int,
) -> Dict[str, Any]:
    model.eval()
    prompt_text = sample.get("input_text") or sample.get("text") or ""

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # multimodal pieces
    image_path = _resolve_path(sample.get("image"), image_root)
    audio_path = _resolve_path(sample.get("audio"), audio_root)

    pixel = torch.zeros(3, collator.image_size, collator.image_size)
    has_image = torch.tensor([0], dtype=torch.long)
    if image_path and os.path.exists(image_path):
        try:
            pixel = collator._load_image(image_path)
            has_image = torch.tensor([1], dtype=torch.long)
        except Exception:
            pixel = torch.zeros(3, collator.image_size, collator.image_size)
            has_image = torch.tensor([0], dtype=torch.long)
    pixel_values = pixel.unsqueeze(0).to(device)

    audio = torch.zeros(collator.max_audio_len)
    has_audio = torch.tensor([0], dtype=torch.long)
    if audio_path and os.path.exists(audio_path):
        try:
            audio = collator._load_audio(audio_path)
            has_audio = torch.tensor([1], dtype=torch.long)
        except Exception:
            audio = torch.zeros(collator.max_audio_len)
            has_audio = torch.tensor([0], dtype=torch.long)
    audio_values = audio.unsqueeze(0).to(device)

    has_image = has_image.to(device)
    has_audio = has_audio.to(device)

    generated = input_ids
    input_len = generated.size(1)
    eos_id = tokenizer.eos_token_id

    import time
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=generated,
                attention_mask=attention_mask,
                labels=None,
                pixel_values=pixel_values,
                audio_values=audio_values,
                has_image=has_image,
                has_audio=has_audio,
            )
            logits = out["logits"]
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.ones_like(generated, device=device)
            if eos_id is not None and next_token.item() == eos_id:
                break

    elapsed = time.perf_counter() - t_start
    new_tokens = generated[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    stats = {
        "input_tokens": input_len,
        "output_tokens": new_tokens.numel(),
        "total_tokens": input_len + new_tokens.numel(),
        "latency_sec": elapsed,
        "output_toks_per_sec": (new_tokens.numel() / elapsed) if elapsed > 0 else float("inf"),
        "total_toks_per_sec": ((input_len + new_tokens.numel()) / elapsed) if elapsed > 0 else float("inf"),
    }
    return {"text": completion, "stats": stats}


def _print_stats(prefix: str, stats: Dict[str, Any]):
    print(
        f"{prefix} tokens: in={stats['input_tokens']} out={stats['output_tokens']} total={stats['total_tokens']} | "
        f"latency={stats['latency_sec']:.3f}s | out_toks/s={stats['output_toks_per_sec']:.2f} "
        f"total_toks/s={stats['total_toks_per_sec']:.2f}"
    )


def run_stage1(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _get_dtype(args.dtype)
    tokenizer = _ensure_tokenizer(args.tokenizer_name_or_path, args.checkpoint)

    load_kwargs: Dict[str, Any] = {}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    try:
        model = Qwen3OmniMoeThinkerTextModel.from_pretrained(args.checkpoint, **load_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "torch.load" in msg or "CVE-2025-32434" in msg:
            raise SystemExit(
                "Model load blocked by torch.load safety check. "
                "Please upgrade torch to >=2.6 or convert the checkpoint to safetensors and retry."
            ) from exc
        raise
    model.to(device)

    def _run_once(prompt: str, idx: int = 1):
        result = greedy_decode_stage1(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print("=" * 40)
        print(f"[Stage1 Sample {idx}] Prompt: {prompt}")
        print(f"[Stage1 Sample {idx}] Completion: {result['text']}")
        _print_stats(f"[Stage1 Sample {idx}]", result["stats"])

    if args.chat:
        print("Chat mode (stage1). Press Ctrl+C or empty line to exit.")
        idx = 1
        try:
            while True:
                user_in = input("> ").strip()
                if not user_in:
                    break
                _run_once(user_in, idx)
                idx += 1
        except KeyboardInterrupt:
            print("\nChat stopped.")
        return

    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    else:
        jsonl_path = args.jsonl or "data/corpus/val_text.jsonl"
        rows = _load_jsonl(jsonl_path, args.num_samples)
        prompts = [r.get("text", "") for r in rows if r.get("text")]
        if not prompts:
            prompts = ["你好，介绍一下你自己。"]

    for idx, prompt in enumerate(prompts, 1):
        _run_once(prompt, idx)


def run_stage2(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _get_dtype(args.dtype)
    tokenizer = _ensure_tokenizer(args.tokenizer_name_or_path, args.checkpoint)
    collator = OmniStage2Collator(tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    load_kwargs: Dict[str, Any] = {}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    try:
        model = Qwen3OmniMoeThinkerVisionAudioModel.from_pretrained(args.checkpoint, **load_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "torch.load" in msg or "CVE-2025-32434" in msg:
            raise SystemExit(
                "Model load blocked by torch.load safety check. "
                "Please upgrade torch to >=2.6 or convert the checkpoint to safetensors and retry."
            ) from exc
        raise
    model.to(device)

    def _run_once(sample: Dict[str, Any], idx: int = 1):
        result = greedy_decode_stage2(
            model=model,
            tokenizer=tokenizer,
            collator=collator,
            device=device,
            sample=sample,
            max_new_tokens=args.max_new_tokens,
            image_root=args.image_root,
            audio_root=args.audio_root,
            max_seq_length=args.max_seq_length,
        )
        target = sample.get("target_text") or ""
        print("=" * 40)
        print(f"[Stage2 Sample {idx}] input_text: {sample.get('input_text') or sample.get('text')}")
        if sample.get("image"):
            print(f"[Stage2 Sample {idx}] image: {sample.get('image')}")
        if sample.get("audio"):
            print(f"[Stage2 Sample {idx}] audio: {sample.get('audio')}")
        if target:
            print(f"[Stage2 Sample {idx}] target_text: {target}")
        print(f"[Stage2 Sample {idx}] completion: {result['text']}")
        _print_stats(f"[Stage2 Sample {idx}]", result["stats"])

    if args.chat:
        print("Chat mode (stage2, text-only input). Press Ctrl+C or empty line to exit.")
        idx = 1
        try:
            while True:
                user_in = input("> ").strip()
                if not user_in:
                    break
                sample = {"input_text": user_in, "image": None, "audio": None}
                _run_once(sample, idx)
                idx += 1
        except KeyboardInterrupt:
            print("\nChat stopped.")
        return

    samples: List[Dict[str, Any]] = []
    if args.prompt:
        samples.append({"input_text": args.prompt})
    else:
        jsonl_path = args.jsonl or "data/corpus/stage2_omni_val.jsonl"
        samples = _load_jsonl(jsonl_path, args.num_samples)
        if not samples:
            samples = [{"input_text": "这张图里有什么？", "image": None, "audio": None}]

    for idx, sample in enumerate(samples, 1):
        _run_once(sample, idx)


def main():
    args = parse_args()
    if args.stage == "stage1":
        run_stage1(args)
    else:
        run_stage2(args)


if __name__ == "__main__":
    main()
