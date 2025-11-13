#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bidirectional training script (Transformer + Capsule-like aggregation).

Usage:
    python scripts/train.py --config configs/config.yml
"""

import argparse
import math
import os
import random
from typing import List, Tuple, Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm


# -------------------- Utils -------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------- Vocab -------------------- #

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = range(4)


def build_or_load_vocab(vocab_path: str, sentences: List[str]) -> Dict[str, int]:
    if os.path.exists(vocab_path):
        # one token per line
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokens = [ln.strip() for ln in f if ln.strip()]
        stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + tokens)}
        return stoi

    print(f"[Vocab] Building vocab from training data → {vocab_path}")
    freq = {}
    for sent in sentences:
        for tok in sent.split():
            freq[tok] = freq.get(tok, 0) + 1

    # keep all tokens (or add frequency filtering here)
    sorted_tokens = sorted(freq.keys(), key=lambda t: -freq[t])

    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok in sorted_tokens:
            f.write(tok + "\n")

    stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + sorted_tokens)}
    return stoi


def numericalize(sentence: str, stoi: Dict[str, int], max_len: int) -> List[int]:
    tokens = sentence.strip().split()
    tokens = tokens[: max_len - 2]  # reserve for BOS/EOS
    ids = [BOS_IDX]
    for t in tokens:
        ids.append(stoi.get(t, UNK_IDX))
    ids.append(EOS_IDX)
    return ids


# -------------------- Dataset -------------------- #

class GECDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stoi: Dict[str, int], max_len: int):
        self.src = df["src"].tolist()
        self.trg = df["trg"].tolist()
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]


def collate_batch(
    batch,
    stoi: Dict[str, int],
    max_len: int,
    direction: str = "L2R",
):
    """
    direction:
        "L2R": normal target
        "R2L": reverse target tokens between BOS and EOS
    """
    src_ids = []
    trg_ids = []

    for src, trg in batch:
        src_seq = numericalize(src, stoi, max_len)
        trg_seq = numericalize(trg, stoi, max_len)

        if direction == "R2L":
            # reverse tokens except BOS/EOS
            core = trg_seq[1:-1]
            core = core[::-1]
            trg_seq = [BOS_IDX] + core + [EOS_IDX]

        src_ids.append(torch.tensor(src_seq, dtype=torch.long))
        trg_ids.append(torch.tensor(trg_seq, dtype=torch.long))

    src_padded = nn.utils.rnn.pad_sequence(
        src_ids, batch_first=True, padding_value=PAD_IDX
    )
    trg_padded = nn.utils.rnn.pad_sequence(
        trg_ids, batch_first=True, padding_value=PAD_IDX
    )

    return src_padded, trg_padded


# -------------------- Model -------------------- #

class CapsuleAggregation(nn.Module):
    """
    Very lightweight capsule-like aggregation over encoder outputs.
    Replace with full CapsNet + EM routing if desired.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        returns: [batch, seq_len, d_model] (reweighted)
        """
        # compute a soft gate per token
        gate = torch.sigmoid(self.gate(x))  # [b, s, 1]
        out = self.proj(x) * gate
        return out


class Seq2SeqTransformerWithCaps(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.trg_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)

        self.pos_encoder = nn.Embedding(512, d_model)  # simple learned PE
        self.pos_decoder = nn.Embedding(512, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.caps_agg = CapsuleAggregation(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def _add_positional(self, x, pos_embedding):
        b, s, _ = x.size()
        positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        return x + pos_embedding(positions)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
    ) -> torch.Tensor:
        """
        src: [b, src_len]
        trg: [b, trg_len]
        returns logits: [b, trg_len, vocab]
        """
        src_mask = (src == PAD_IDX)
        trg_mask = (trg == PAD_IDX)

        # subsequent mask for auto-regressive decoding
        tgt_seq_len = trg.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_seq_len, tgt_seq_len, device=trg.device), diagonal=1
        ).bool()

        src_emb = self._add_positional(self.src_embed(src), self.pos_encoder)
        trg_emb = self._add_positional(self.trg_embed(trg), self.pos_decoder)

        enc_out = self.encoder(src_emb, src_key_padding_mask=src_mask)

        # capsule-like aggregation over encoder outputs
        enc_out = self.caps_agg(enc_out)

        dec_out = self.decoder(
            trg_emb,
            enc_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=trg_mask,
            memory_key_padding_mask=src_mask,
        )

        logits = self.output_layer(dec_out)
        return logits


# -------------------- Training -------------------- #

def compute_ce_loss(
    logits: torch.Tensor,
    trg: torch.Tensor,
) -> torch.Tensor:
    """
    logits: [b, T, V]
    trg: [b, T]
    """
    # shift targets to ignore BOS, predict next token
    logits = logits[:, :-1].contiguous()
    gold = trg[:, 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    return loss_fn(logits.view(-1, logits.size(-1)), gold.view(-1))


def symmetric_kl(p_log_probs, q_log_probs) -> torch.Tensor:
    """
    p_log_probs, q_log_probs: [b, T, V] log_softmax outputs
    symmetric KL: D_KL(p||q) + D_KL(q||p)
    """
    p_probs = p_log_probs.exp()
    q_probs = q_log_probs.exp()
    kl_pq = (p_probs * (p_log_probs - q_log_probs)).sum(-1)
    kl_qp = (q_probs * (q_log_probs - p_log_probs)).sum(-1)
    return (kl_pq + kl_qp).mean()


def train_bidirectional(cfg: dict):
    general = cfg["general"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    set_seed(general.get("seed", 42))

    device_str = general.get("device", "cuda")
    device = torch.device(
        "cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[Train] Using device: {device}")

    save_dir = general.get("save_dir", "checkpoints/")
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(data_cfg["train_csv"])
    assert "src" in df.columns and "trg" in df.columns

    all_sentences = df["src"].tolist() + df["trg"].tolist()
    stoi = build_or_load_vocab(data_cfg["vocab"], all_sentences)
    vocab_size = len(stoi)
    print(f"[Vocab] Size: {vocab_size}")

    max_len = int(train_cfg.get("max_seq_len", 128))

    # create two datasets that share the same df but direction differs in collate
    dataset = GECDataset(df, stoi, max_len=max_len)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_epochs = int(train_cfg.get("num_epochs", 20))
    lr = float(train_cfg.get("lr", 2e-4))
    kl_weight = float(train_cfg.get("kl_regularization_weight", 0.8))
    use_bidirectional = bool(train_cfg.get("use_bidirectional", True))

    # --- Dataloaders with separate collate_fns ---
    def collate_l2r(batch):
        return collate_batch(batch, stoi, max_len=max_len, direction="L2R")

    def collate_r2l(batch):
        return collate_batch(batch, stoi, max_len=max_len, direction="R2L")

    loader_l2r = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_l2r,
    )

    loader_r2l = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_r2l,
    )

    # --- Models ---
    d_model = int(train_cfg.get("embedding_dim", 256))
    hidden_dim = int(train_cfg.get("hidden_dim", 256))
    num_layers = int(train_cfg.get("num_layers", 4))
    num_heads = int(train_cfg.get("num_heads", 8))
    dropout = float(train_cfg.get("dropout", 0.15))

    model_l2r = Seq2SeqTransformerWithCaps(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=hidden_dim * 4,
        dropout=dropout,
    ).to(device)

    model_r2l = Seq2SeqTransformerWithCaps(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=hidden_dim * 4,
        dropout=dropout,
    ).to(device)

    params = list(model_l2r.parameters()) + list(model_r2l.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # --- Training loop ---
    for epoch in range(1, num_epochs + 1):
        model_l2r.train()
        model_r2l.train()

        total_loss = 0.0
        n_batches = 0

        for (batch_l2r, batch_r2l) in zip(loader_l2r, loader_r2l):
            src_l2r, trg_l2r = batch_l2r
            src_r2l, trg_r2l = batch_r2l

            src_l2r = src_l2r.to(device)
            trg_l2r = trg_l2r.to(device)
            src_r2l = src_r2l.to(device)
            trg_r2l = trg_r2l.to(device)

            optimizer.zero_grad()

            logits_l2r = model_l2r(src_l2r, trg_l2r)
            logits_r2l = model_r2l(src_r2l, trg_r2l)

            ce_l2r = compute_ce_loss(logits_l2r, trg_l2r)
            ce_r2l = compute_ce_loss(logits_r2l, trg_r2l)
            loss = ce_l2r + ce_r2l

            if use_bidirectional:
                log_p = torch.log_softmax(logits_l2r, dim=-1)
                log_q = torch.log_softmax(logits_r2l, dim=-1)
                kl = symmetric_kl(log_p, log_q)
                loss = loss + kl_weight * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        print(f"[Epoch {epoch:02d}] loss={avg_loss:.4f}  ppl={ppl:.2f}")

        # save checkpoints
        torch.save(
            model_l2r.state_dict(),
            os.path.join(save_dir, f"gec_l2r_epoch{epoch}.pt"),
        )
        torch.save(
            model_r2l.state_dict(),
            os.path.join(save_dir, f"gec_r2l_epoch{epoch}.pt"),
        )

    print("[Train] Finished training.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configs/config.yml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_bidirectional(cfg)


if __name__ == "__main__":
    main()

