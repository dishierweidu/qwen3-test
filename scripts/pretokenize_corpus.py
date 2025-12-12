# scripts/pretokenize_corpus.py
import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer


def iter_text_from_jsonl(paths):
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if text:
                    yield text


def main():
    parser = argparse.ArgumentParser(
        description="Offline pre-tokenization for jsonl corpus ({'text': ...})."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more jsonl files, each line: {'text': '...'}",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix, will write <prefix>.bin and <prefix>.meta.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="HF tokenizer name or path",
    )
    parser.add_argument(
        "--append-eos",
        action="store_true",
        help="Append eos_token_id after each sample (recommended for continuous stream).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="(Optional) max samples to process, for debugging.",
    )
    args = parser.parse_args()

    print(f"[pretokenize] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        # 兜底：没有 pad，就用 eos，当做 pad
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    tokens = []
    n_samples = 0

    print(f"[pretokenize] reading from: {args.input}")
    for text in iter_text_from_jsonl(args.input):
        n_samples += 1
        if args.max_samples > 0 and n_samples > args.max_samples:
            break

        out = tokenizer(
            text,
            add_special_tokens=False,
        )
        ids = out["input_ids"]
        if not ids:
            continue

        if args.append_eos and eos_id is not None:
            ids = ids + [int(eos_id)]

        tokens.extend(ids)

        if n_samples % 1000 == 0:
            print(f"[pretokenize] processed {n_samples} samples, total tokens={len(tokens)}")

    tokens = np.asarray(tokens, dtype=np.int32)
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    bin_path = args.output_prefix + ".bin"
    meta_path = args.output_prefix + ".meta.json"

    print(f"[pretokenize] writing {tokens.shape[0]} tokens to {bin_path}")
    tokens.tofile(bin_path)

    meta = {
        "token_count": int(tokens.shape[0]),
        "dtype": "int32",
        "eos_token_id": int(eos_id) if eos_id is not None else None,
        "pad_token_id": int(tokenizer.pad_token_id),
        "vocab_size": int(tokenizer.vocab_size),
        # 只是记录推荐 seq_length，真正用多少由训练配置决定
        "recommended_seq_length": 2048,
        "source_files": args.input,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[pretokenize] done. bin={bin_path}, meta={meta_path}")


if __name__ == "__main__":
    main()
