from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    CONTEXT_RESIDUES,
    DATA_DIR,
    MAX_NATURAL_JUNCTION_LENGTH,
    MIN_NATURAL_JUNCTION_LENGTH,
    MODELS_DIR,
    SEED,
)
from domain_fusion import AA, ConditionalLSTM, UNK_DOMAIN

SPECIAL = ("<PAD>", "<BOS>", "<EOS>")
MIN_DOMAIN_ID_COUNT = 2
VALIDATION_FRACTION = 0.12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_examples() -> list[dict]:
    path = DATA_DIR / "domain_pairs.csv"
    if not path.exists():
        raise FileNotFoundError("Run download_domains.py first.")

    df = pd.read_csv(path)
    required = {
        "protein_accession",
        "junction",
        "domain_a",
        "domain_b",
        "domain_a_length",
        "domain_b_length",
        "left_context",
        "right_context",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "domain_pairs.csv is from an obsolete dataset schema. Missing: "
            f"{sorted(missing)}. Re-run download_domains.py."
        )

    examples: list[dict] = []
    seen: set[tuple] = set()
    for row in df.itertuples(index=False):
        seq = str(row.junction).upper()
        left_context = str(row.left_context).upper()[-CONTEXT_RESIDUES:]
        right_context = str(row.right_context).upper()[:CONTEXT_RESIDUES]
        left_domain = str(row.domain_a)
        right_domain = str(row.domain_b)
        accession = str(row.protein_accession)

        if not MIN_NATURAL_JUNCTION_LENGTH <= len(seq) <= MAX_NATURAL_JUNCTION_LENGTH:
            continue
        if not seq or not left_context or not right_context:
            continue
        if not set(seq + left_context + right_context) <= AA:
            continue

        key = (accession, seq, left_context, right_context, left_domain, right_domain)
        if key in seen:
            continue
        seen.add(key)
        examples.append({
            "sequence": seq,
            "left_context": left_context,
            "right_context": right_context,
            "left_domain": left_domain,
            "right_domain": right_domain,
            "left_length": int(row.domain_a_length),
            "right_length": int(row.domain_b_length),
            "accession": accession,
        })

    if len(examples) < 20:
        raise RuntimeError(
            f"Only {len(examples)} usable natural junctions were found. "
            "The v3 model needs a broader dataset; rebuild with download_domains.py."
        )
    return examples


def build_domain_vocabulary(examples: list[dict]) -> dict[str, int]:
    counts = Counter()
    for ex in examples:
        counts[ex["left_domain"]] += 1
        counts[ex["right_domain"]] += 1

    # Rare IDs are mapped to UNK.  Their boundary sequences still condition the
    # generator, which avoids one-shot categorical embeddings pretending to be
    # learned domain knowledge.
    domains = sorted(d for d, n in counts.items() if n >= MIN_DOMAIN_ID_COUNT)
    return {UNK_DOMAIN: 0, **{domain: i + 1 for i, domain in enumerate(domains)}}


def build_vocab() -> dict[str, int]:
    return {token: i for i, token in enumerate(list(SPECIAL) + sorted(AA))}


class JunctionDataset(Dataset):
    def __init__(self, examples: list[dict], vocab: dict[str, int], domain_to_idx: dict[str, int]):
        self.examples = examples
        self.vocab = vocab
        self.domain_to_idx = domain_to_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = [self.vocab["<BOS>"]] + [self.vocab[x] for x in ex["sequence"]] + [self.vocab["<EOS>"]]
        left = [self.vocab[x] for x in ex["left_context"]]
        right = [self.vocab[x] for x in ex["right_context"]]
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "left_context": torch.tensor(left, dtype=torch.long),
            "right_context": torch.tensor(right, dtype=torch.long),
            "left_domain": int(self.domain_to_idx.get(ex["left_domain"], 0)),
            "right_domain": int(self.domain_to_idx.get(ex["right_domain"], 0)),
            "domain_lengths": torch.tensor([ex["left_length"], ex["right_length"]], dtype=torch.float32),
            "target_length": len(ex["sequence"]),
        }


def collate(batch: list[dict], pad_idx: int):
    max_seq = max(len(x["tokens"]) for x in batch)
    max_left = max(len(x["left_context"]) for x in batch)
    max_right = max(len(x["right_context"]) for x in batch)

    inputs = torch.full((len(batch), max_seq - 1), pad_idx, dtype=torch.long)
    targets = torch.full((len(batch), max_seq - 1), pad_idx, dtype=torch.long)
    left_context = torch.full((len(batch), max_left), pad_idx, dtype=torch.long)
    right_context = torch.full((len(batch), max_right), pad_idx, dtype=torch.long)

    for i, item in enumerate(batch):
        tokens = item["tokens"]
        inputs[i, :len(tokens) - 1] = tokens[:-1]
        targets[i, :len(tokens) - 1] = tokens[1:]
        left_context[i, :len(item["left_context"])] = item["left_context"]
        right_context[i, :len(item["right_context"])] = item["right_context"]

    return {
        "inputs": inputs,
        "targets": targets,
        "left_context": left_context,
        "right_context": right_context,
        "left_domain": torch.tensor([x["left_domain"] for x in batch], dtype=torch.long),
        "right_domain": torch.tensor([x["right_domain"] for x in batch], dtype=torch.long),
        "domain_lengths": torch.stack([x["domain_lengths"] for x in batch]),
        "target_length": torch.tensor([x["target_length"] for x in batch], dtype=torch.long),
    }


def split_by_accession(examples: list[dict], seed: int):
    groups = sorted({x["accession"] for x in examples})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val_groups = max(1, int(round(len(groups) * VALIDATION_FRACTION)))
    val_groups = set(groups[:n_val_groups])
    train = [x for x in examples if x["accession"] not in val_groups]
    val = [x for x in examples if x["accession"] in val_groups]

    if not train or not val:
        # Extremely small synthetic/debug datasets can have too few proteins.
        cut = max(1, int(round(len(examples) * (1 - VALIDATION_FRACTION))))
        train, val = examples[:cut], examples[cut:]
    return train, val


def move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def model_forward(model: ConditionalLSTM, batch: dict):
    return model(
        batch["inputs"],
        batch["left_context"],
        batch["right_context"],
        batch["left_domain"],
        batch["right_domain"],
        batch["domain_lengths"],
        batch["target_length"],
    )


def evaluate(
    model: ConditionalLSTM,
    loader: DataLoader,
    token_criterion,
    length_criterion,
    pad_idx: int,
    device: str,
):
    model.eval()
    token_loss_sum = length_loss_sum = 0.0
    token_count = example_count = 0
    shuffled_token_loss_sum = shuffled_token_count = 0
    correct_lengths = 0

    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            logits, length_logits = model_forward(model, batch)
            token_loss = token_criterion(logits.reshape(-1, logits.size(-1)), batch["targets"].reshape(-1))
            length_loss = length_criterion(length_logits, batch["target_length"])

            n_tokens = int((batch["targets"] != pad_idx).sum().item())
            n_examples = int(batch["inputs"].size(0))
            token_loss_sum += float(token_loss.item()) * n_tokens
            length_loss_sum += float(length_loss.item()) * n_examples
            token_count += n_tokens
            example_count += n_examples
            correct_lengths += int((length_logits.argmax(dim=-1) == batch["target_length"]).sum().item())

            if n_examples > 1:
                perm = torch.roll(torch.arange(n_examples, device=device), shifts=1)
                shuffled = dict(batch)
                for key in ("left_context", "right_context", "left_domain", "right_domain", "domain_lengths"):
                    shuffled[key] = batch[key][perm]
                shuffled_logits, _ = model_forward(model, shuffled)
                shuffled_loss = token_criterion(
                    shuffled_logits.reshape(-1, shuffled_logits.size(-1)),
                    batch["targets"].reshape(-1),
                )
                shuffled_token_loss_sum += float(shuffled_loss.item()) * n_tokens
                shuffled_token_count += n_tokens

    token_loss = token_loss_sum / max(token_count, 1)
    length_loss = length_loss_sum / max(example_count, 1)
    shuffled_loss = shuffled_token_loss_sum / max(shuffled_token_count, 1) if shuffled_token_count else float("nan")
    return {
        "token_loss": token_loss,
        "length_loss": length_loss,
        "total_loss": token_loss + 0.20 * length_loss,
        "perplexity": math.exp(min(20.0, token_loss)),
        "length_accuracy": correct_lengths / max(example_count, 1),
        "shuffled_context_token_loss": shuffled_loss,
        "conditioning_gap": shuffled_loss - token_loss if math.isfinite(shuffled_loss) else float("nan"),
    }


def train(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    patience: int,
):
    set_seed(seed)
    examples = load_training_examples()
    train_examples, val_examples = split_by_accession(examples, seed)

    domain_to_idx = build_domain_vocabulary(train_examples)
    vocab = build_vocab()
    pad_idx = vocab["<PAD>"]

    train_dataset = JunctionDataset(train_examples, vocab, domain_to_idx)
    val_dataset = JunctionDataset(val_examples, vocab, domain_to_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_idx),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, pad_idx),
    )

    model = ConditionalLSTM(
        vocab_size=len(vocab),
        num_domains=len(domain_to_idx),
        pad_idx=pad_idx,
        embedding_dim=64,
        context_embedding_dim=32,
        context_hidden_dim=32,
        domain_embedding_dim=16,
        condition_dim=128,
        hidden_dim=160,
        num_layers=2,
        dropout=0.20,
        max_target_length=MAX_NATURAL_JUNCTION_LENGTH,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    token_criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.03)
    length_criterion = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    best_metrics = None
    epochs_without_improvement = 0

    print(
        f"Training examples={len(train_examples)}, validation={len(val_examples)}, "
        f"domain IDs with >= {MIN_DOMAIN_ID_COUNT} observations={len(domain_to_idx)-1}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        token_loss_sum = length_loss_sum = 0.0
        token_count = example_count = 0

        for raw in train_loader:
            batch = move_batch(raw, device)

            # Domain-ID dropout forces the network to remain useful for unseen
            # pairs and rely on terminal sequence chemistry instead of memorizing IDs.
            if model.training:
                for key in ("left_domain", "right_domain"):
                    mask = torch.rand(batch[key].shape, device=device) < 0.12
                    batch[key] = batch[key].masked_fill(mask, 0)

            optimizer.zero_grad(set_to_none=True)
            logits, length_logits = model_forward(model, batch)
            token_loss = token_criterion(
                logits.reshape(-1, logits.size(-1)), batch["targets"].reshape(-1)
            )
            length_loss = length_criterion(length_logits, batch["target_length"])
            loss = token_loss + 0.20 * length_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_tokens = int((batch["targets"] != pad_idx).sum().item())
            n_examples = int(batch["inputs"].size(0))
            token_loss_sum += float(token_loss.item()) * n_tokens
            length_loss_sum += float(length_loss.item()) * n_examples
            token_count += n_tokens
            example_count += n_examples

        train_token = token_loss_sum / max(token_count, 1)
        train_length = length_loss_sum / max(example_count, 1)
        metrics = evaluate(
            model, val_loader, token_criterion, length_criterion, pad_idx, device
        )
        scheduler.step(metrics["total_loss"])
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_token={train_token:.4f} train_len={train_length:.4f} "
            f"val_token={metrics['token_loss']:.4f} val_ppl={metrics['perplexity']:.2f} "
            f"val_len_acc={metrics['length_accuracy']:.3f} "
            f"context_gap={metrics['conditioning_gap']:.4f} lr={lr:.2e}"
        )

        if metrics["total_loss"] < best_val - 1e-4:
            best_val = metrics["total_loss"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)

    length_counts = Counter(len(x["sequence"]) for x in train_examples)
    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "domain_to_idx": domain_to_idx,
        "num_domains": len(domain_to_idx),
        "embedding_dim": 64,
        "context_embedding_dim": 32,
        "context_hidden_dim": 32,
        "domain_embedding_dim": 16,
        "condition_dim": 128,
        "hidden_dim": 160,
        "num_layers": 2,
        "dropout": 0.20,
        "max_target_length": MAX_NATURAL_JUNCTION_LENGTH,
        "checkpoint_version": 3,
        "conditioning": "terminal_sequences+domain_ids+domain_lengths+target_length",
        "training_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "length_counts": dict(sorted(length_counts.items())),
        "seed": seed,
    }
    MODELS_DIR.mkdir(exist_ok=True)
    output = MODELS_DIR / "protein_generator.pt"
    torch.save(checkpoint, output)

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"state_dict", "vocab", "domain_to_idx"}
    }
    metadata["vocabulary_size"] = len(vocab)
    metadata["learned_domain_ids"] = len(domain_to_idx) - 1
    metadata["best_validation"] = best_metrics
    metadata["validation_split"] = "grouped by protein_accession"
    metadata["rare_domain_policy"] = f"IDs with <{MIN_DOMAIN_ID_COUNT} observations map to {UNK_DOMAIN}; sequence context remains active"
    with open(MODELS_DIR / "protein_generator_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {output}")
    print(f"Saved: {MODELS_DIR / 'protein_generator_metadata.json'}")


def main():
    parser = argparse.ArgumentParser(description="Train the v3 sequence-conditioned linker generator.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
