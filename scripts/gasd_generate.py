#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GASD synthetic data generator.

Usage:
    python scripts/gasd_generate.py --config configs/config.yml
"""

import argparse
import csv
import os
import random
from typing import List

import yaml
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- GASD components: SDA + SEG (simplified, faithful to the paper’s idea) ---

def swap_words(tokens: List[str], n_swaps: int) -> List[str]:
    tokens = tokens[:]
    length = len(tokens)
    for _ in range(n_swaps):
        if length < 2:
            break
        i, j = random.sample(range(length), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens


AR_CHAR_NORMALIZATION = {
    "ة": "ه",
    "ى": "ي",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ؤ": "و",
    "ئ": "ي",
}


def seg_char_level(word: str, min_len: int = 2) -> str:
    """Very small SEG-style character noise: insert/delete/substitute."""
    if len(word) < min_len:
        return word

    op = random.choice(["insert", "delete", "normalize"])
    chars = list(word)

    if op == "insert":
        pos = random.randrange(0, len(chars) + 1)
        ch = random.choice(chars)
        chars.insert(pos, ch)
    elif op == "delete" and len(chars) > 1:
        pos = random.randrange(0, len(chars))
        del chars[pos]
    elif op == "normalize":
        for i, ch in enumerate(chars):
            if ch in AR_CHAR_NORMALIZATION:
                chars[i] = AR_CHAR_NORMALIZATION[ch]
                break

    return "".join(chars)


def apply_gasd(
    sentence: str,
    alpha: float,
    sda_swap_prob: float,
    seg_char_error_prob: float,
    min_len_char_ops: int,
) -> str:

    tokens = sentence.strip().split()
    if not tokens:
        return sentence

    length = len(tokens)

    # --- SDA: word swapping ---
    if length >= 4 and random.random() < sda_swap_prob:
        n_swaps = max(1, int(alpha * length))
        tokens = swap_words(tokens, n_swaps)

    # --- SEG: character-level perturbations ---
    for i in range(length):
        if random.random() < seg_char_error_prob:
            tokens[i] = seg_char_level(tokens[i], min_len=min_len_char_ops)

    return " ".join(tokens)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configs/config.yml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    general = cfg.get("general", {})
    data_cfg = cfg.get("data", {})
    gasd_cfg = cfg.get("gasd", {})

    set_seed(general.get("seed", 42))

    raw_path = data_cfg["raw_mono"]
    output_path = gasd_cfg.get("output_path", data_cfg["train_csv"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[GASD] Reading monolingual corpus from: {raw_path}")
    sentences = read_lines(raw_path)

    alpha = float(gasd_cfg.get("alpha", 0.1))
    sda_swap_prob = float(gasd_cfg.get("sda_swap_prob", 0.2))
    seg_char_error_prob = float(gasd_cfg.get("seg_char_error_prob", 0.15))
    min_len_char_ops = int(gasd_cfg.get("min_len_char_ops", 2))

    print(f"[GASD] Generating noisy pairs → {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["src", "trg"])  # header

        for sent in tqdm(sentences, desc="GASD", unit="sent"):
            trg = sent
            src = apply_gasd(
                sent,
                alpha=alpha,
                sda_swap_prob=sda_swap_prob,
                seg_char_error_prob=seg_char_error_prob,
                min_len_char_ops=min_len_char_ops,
            )
            writer.writerow([src, trg])

    print("[GASD] Done.")


if __name__ == "__main__":
    main()
