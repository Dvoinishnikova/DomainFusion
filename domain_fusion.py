from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from config import (
    CONTEXT_RESIDUES,
    DATA_DIR,
    ESMFOLD_TIMEOUT,
    ESMFOLD_URL,
    MAX_LINKER_CANDIDATES,
    MAX_LINKER_LENGTH,
    MIN_LINKER_LENGTH,
    MODELS_DIR,
    PLDDT_RANK_RESOLUTION,
    REMOTE_ESMFOLD_MAX_LENGTH,
    RESULTS_DIR,
    SEED,
    AI_REQUESTED_CANDIDATES,
)

AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWY")
CHARGED = set("DEKR")

EMPIRICAL_LINKERS = (
    "GGSGG",
    "GGSGGSG",
    "EAAAK",
    "EAAAKEAAAK",
)

HEURISTIC_LINKERS = (
    "GGGGS",
    "GGGGGS",
    "GGGGGGS",
    "GSGS",
    "GSGSG",
    "GSGSGS",
    "GSGSGSG",
    "EAAAK",
    "EAAAKEAAAK",
    "AEAAAKEAAAKA",
    "PAPAP",
    "PAPAPAP",
    "PAPAPAPAP",
)

AI_TEMPERATURE = 0.90
AI_TOP_K = 8
AI_MAX_ATTEMPTS = 80
UNK_DOMAIN = "<UNK_DOMAIN>"


@dataclass
class Candidate:
    domain_a: str
    domain_b: str
    linker: str
    category: str
    sequence: str
    linker_length: int
    hydrophobic_fraction: float
    charged_fraction: float
    proline_fraction: float
    gly_ser_fraction: float
    natural_pair_probability: float = 0.0
    natural_pair_count: int = 0

    # Generator diagnostics are computed for every candidate, including
    # heuristics.  They are deliberately not part of the structural score.
    model_log_likelihood_per_residue: float | None = None
    model_perplexity: float | None = None
    predicted_length_probability: float | None = None

    domain_a_plddt: float | None = None
    domain_b_plddt: float | None = None
    protein_plddt: float | None = None
    linker_plddt: float | None = None
    whole_plddt: float | None = None
    junction_plddt: float | None = None
    flank_plddt: float | None = None
    junction_drop: float | None = None

    domain_a_baseline_plddt: float | None = None
    domain_b_baseline_plddt: float | None = None
    domain_a_preservation_lddt: float | None = None
    domain_b_preservation_lddt: float | None = None
    domain_preservation_score: float | None = None
    junction_drop_score: float | None = None
    absolute_score: float | None = None
    score: float | None = None  # comparative 0-100 rank score
    score_delta_from_best: float | None = None

    pdb_path: str | None = None
    fold_error: str | None = None


class ConditionalLSTM(torch.nn.Module):
    """Sequence- and domain-conditioned linker generator.

    Domain identity alone is sparse and does not generalize.  This model encodes
    the C-terminal residues of the left domain and N-terminal residues of the
    right domain with bidirectional GRUs, adds optional domain-ID embeddings and
    domain lengths, and injects the resulting context into *every* decoder step.
    A separate length head predicts a linker-length distribution from the same
    context.  Generation is therefore exact-length and does not depend on an EOS
    lottery.
    """

    def __init__(
        self,
        vocab_size: int,
        num_domains: int,
        pad_idx: int,
        embedding_dim: int = 64,
        context_embedding_dim: int = 32,
        context_hidden_dim: int = 32,
        domain_embedding_dim: int = 16,
        condition_dim: int = 128,
        hidden_dim: int = 160,
        num_layers: int = 2,
        dropout: float = 0.20,
        max_target_length: int = 30,
    ):
        super().__init__()
        self.pad_idx = int(pad_idx)
        self.max_target_length = int(max_target_length)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.context_embedding = torch.nn.Embedding(
            vocab_size, context_embedding_dim, padding_idx=pad_idx
        )
        self.left_context_encoder = torch.nn.GRU(
            context_embedding_dim,
            context_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.right_context_encoder = torch.nn.GRU(
            context_embedding_dim,
            context_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.domain_embedding = torch.nn.Embedding(
            num_domains, domain_embedding_dim, padding_idx=0
        )

        base_dim = 4 * context_hidden_dim + 2 * domain_embedding_dim + 2
        self.base_projection = torch.nn.Sequential(
            torch.nn.Linear(base_dim, condition_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
        )
        self.decoder_condition = torch.nn.Sequential(
            torch.nn.Linear(condition_dim + 1, condition_dim),
            torch.nn.Tanh(),
        )
        self.hidden_projection = torch.nn.Linear(condition_dim, hidden_dim)
        self.cell_projection = torch.nn.Linear(condition_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim + condition_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = torch.nn.Linear(hidden_dim, vocab_size)
        self.length_head = torch.nn.Linear(condition_dim, max_target_length + 1)

    def _encode_side(self, tokens: torch.Tensor, encoder: torch.nn.GRU) -> torch.Tensor:
        lengths = (tokens != self.pad_idx).sum(dim=1).clamp(min=1)
        embedded = self.context_embedding(tokens)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = encoder(packed)
        return torch.cat([hidden[-2], hidden[-1]], dim=-1)

    def encode_base_context(
        self,
        left_context: torch.Tensor,
        right_context: torch.Tensor,
        left_domain_idx: torch.Tensor,
        right_domain_idx: torch.Tensor,
        domain_lengths: torch.Tensor,
    ) -> torch.Tensor:
        left_seq = self._encode_side(left_context, self.left_context_encoder)
        right_seq = self._encode_side(right_context, self.right_context_encoder)
        left_id = self.domain_embedding(left_domain_idx)
        right_id = self.domain_embedding(right_domain_idx)

        # Compress raw lengths to a stable 0-~1 scale.  Sequence context carries
        # local chemistry; length provides coarse domain geometry information.
        length_features = torch.log1p(domain_lengths.float()) / math.log1p(600.0)
        merged = torch.cat(
            [left_seq, right_seq, left_id, right_id, length_features], dim=-1
        )
        return self.base_projection(merged)

    def condition_with_length(
        self, base_context: torch.Tensor, target_lengths: torch.Tensor
    ) -> torch.Tensor:
        target = target_lengths.float().unsqueeze(-1) / float(self.max_target_length)
        return self.decoder_condition(torch.cat([base_context, target], dim=-1))

    def initial_state(self, condition: torch.Tensor):
        h = torch.tanh(self.hidden_projection(condition))
        c = torch.tanh(self.cell_projection(condition))
        return (
            h.unsqueeze(0).repeat(self.num_layers, 1, 1),
            c.unsqueeze(0).repeat(self.num_layers, 1, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        left_context: torch.Tensor,
        right_context: torch.Tensor,
        left_domain_idx: torch.Tensor,
        right_domain_idx: torch.Tensor,
        domain_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.encode_base_context(
            left_context,
            right_context,
            left_domain_idx,
            right_domain_idx,
            domain_lengths,
        )
        condition = self.condition_with_length(base, target_lengths)
        embedded = self.embedding(tokens)
        repeated = condition.unsqueeze(1).expand(-1, tokens.size(1), -1)
        decoder_input = torch.cat([embedded, repeated], dim=-1)
        output, _ = self.lstm(decoder_input, self.initial_state(condition))
        return self.output(output), self.length_head(base)

    @torch.no_grad()
    def length_distribution(
        self,
        left_context: torch.Tensor,
        right_context: torch.Tensor,
        left_domain_idx: torch.Tensor,
        right_domain_idx: torch.Tensor,
        domain_lengths: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        base = self.encode_base_context(
            left_context,
            right_context,
            left_domain_idx,
            right_domain_idx,
            domain_lengths,
        )
        return torch.softmax(self.length_head(base), dim=-1)

    @torch.no_grad()
    def generate(
        self,
        left_context: torch.Tensor,
        right_context: torch.Tensor,
        left_domain_idx: torch.Tensor,
        right_domain_idx: torch.Tensor,
        domain_lengths: torch.Tensor,
        target_length: int,
        bos_idx: int,
        invalid_output_indices: Iterable[int],
        temperature: float = AI_TEMPERATURE,
        top_k: int = AI_TOP_K,
    ) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        target_tensor = torch.tensor([target_length], dtype=torch.long, device=device)
        base = self.encode_base_context(
            left_context,
            right_context,
            left_domain_idx,
            right_domain_idx,
            domain_lengths,
        )
        condition = self.condition_with_length(base, target_tensor)
        state = self.initial_state(condition)
        token = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
        generated: list[int] = []

        for _ in range(target_length):
            emb = self.embedding(token[:, -1:])
            step_condition = condition.unsqueeze(1)
            output, state = self.lstm(torch.cat([emb, step_condition], dim=-1), state)
            logits = self.output(output[:, -1, :]) / max(float(temperature), 1e-6)
            for idx in invalid_output_indices:
                logits[:, int(idx)] = -float("inf")

            if top_k > 0 and top_k < logits.size(-1):
                top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
                probs = torch.softmax(top_values, dim=-1)
                sampled_local = torch.multinomial(probs, 1)
                sampled = top_indices.gather(1, sampled_local)
            else:
                sampled = torch.multinomial(torch.softmax(logits, dim=-1), 1)

            idx = int(sampled.item())
            generated.append(idx)
            token = sampled

        return generated


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    domains = pd.read_csv(DATA_DIR / "domains.csv")
    proteins = pd.read_csv(DATA_DIR / "proteins.csv")
    pairs_path = DATA_DIR / "domain_pairs.csv"
    pairs = pd.read_csv(pairs_path) if pairs_path.exists() else pd.DataFrame()

    model_path = MODELS_DIR / "domain_compatibility.json"
    compatibility: dict = {}
    if model_path.exists():
        with open(model_path, encoding="utf-8") as f:
            compatibility = json.load(f)
    return domains, proteins, pairs, compatibility


def load_generator(device: str = "cpu"):
    path = MODELS_DIR / "protein_generator.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train_generator.py first.")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    version = int(checkpoint.get("checkpoint_version", 1))
    if version < 3:
        raise RuntimeError(
            "protein_generator.pt uses the obsolete domain-ID-only architecture. "
            "Rebuild data and run train_generator.py to create a v3 checkpoint."
        )

    vocab = checkpoint["vocab"]
    model = ConditionalLSTM(
        vocab_size=len(vocab),
        num_domains=checkpoint["num_domains"],
        pad_idx=vocab["<PAD>"],
        embedding_dim=checkpoint.get("embedding_dim", 64),
        context_embedding_dim=checkpoint.get("context_embedding_dim", 32),
        context_hidden_dim=checkpoint.get("context_hidden_dim", 32),
        domain_embedding_dim=checkpoint.get("domain_embedding_dim", 16),
        condition_dim=checkpoint.get("condition_dim", 128),
        hidden_dim=checkpoint.get("hidden_dim", 160),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.20),
        max_target_length=checkpoint.get("max_target_length", 30),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, checkpoint["domain_to_idx"], checkpoint


def pair_prior(model: dict, a: str, b: str) -> tuple[float, int]:
    row = model.get("pairs", {}).get(str(a), {}).get(str(b))
    if not row:
        return 0.0, 0
    return float(row["conditional_probability"]), int(row["count"])


def heuristic_linkers() -> list[str]:
    return [
        x for x in HEURISTIC_LINKERS
        if MIN_LINKER_LENGTH <= len(x) <= MAX_LINKER_LENGTH and set(x) <= AA
    ]


def features(linker: str) -> dict[str, float]:
    n = max(len(linker), 1)
    return {
        "hydrophobic_fraction": sum(x in HYDROPHOBIC for x in linker) / n,
        "charged_fraction": sum(x in CHARGED for x in linker) / n,
        "proline_fraction": linker.count("P") / n,
        "gly_ser_fraction": sum(x in "GS" for x in linker) / n,
    }


def _tokenize_context(seq: str, vocab: dict[str, int], device: str) -> torch.Tensor:
    pad = vocab["<PAD>"]
    tokens = [vocab[x] for x in seq[-CONTEXT_RESIDUES:] if x in AA]
    if not tokens:
        tokens = [pad]
    return torch.tensor([tokens], dtype=torch.long, device=device)


def _domain_index(domain: str, domain_to_idx: dict[str, int]) -> int:
    return int(domain_to_idx.get(str(domain), domain_to_idx.get(UNK_DOMAIN, 0)))


def _model_context_tensors(
    seq_a: str,
    seq_b: str,
    domain_a: str,
    domain_b: str,
    vocab: dict[str, int],
    domain_to_idx: dict[str, int],
    device: str,
):
    left = _tokenize_context(seq_a[-CONTEXT_RESIDUES:], vocab, device)
    right_tokens = [vocab[x] for x in seq_b[:CONTEXT_RESIDUES] if x in AA]
    if not right_tokens:
        right_tokens = [vocab["<PAD>"]]
    right = torch.tensor([right_tokens], dtype=torch.long, device=device)
    left_idx = torch.tensor([_domain_index(domain_a, domain_to_idx)], dtype=torch.long, device=device)
    right_idx = torch.tensor([_domain_index(domain_b, domain_to_idx)], dtype=torch.long, device=device)
    lengths = torch.tensor([[len(seq_a), len(seq_b)]], dtype=torch.float32, device=device)
    return left, right, left_idx, right_idx, lengths


def _valid_length_probabilities(
    model: ConditionalLSTM,
    tensors,
) -> np.ndarray:
    probs = model.length_distribution(*tensors).detach().cpu().numpy()[0]
    mask = np.zeros_like(probs)
    lo = max(MIN_LINKER_LENGTH, 0)
    hi = min(MAX_LINKER_LENGTH, len(probs) - 1)
    mask[lo:hi + 1] = probs[lo:hi + 1]
    total = float(mask.sum())
    if total <= 0:
        mask[lo:hi + 1] = 1.0
        total = float(mask.sum())
    return mask / total


def generate_ai_linkers(
    domain_a: str,
    domain_b: str,
    seq_a: str,
    seq_b: str,
    model: ConditionalLSTM,
    vocab: dict[str, int],
    domain_to_idx: dict[str, int],
    requested: int = AI_REQUESTED_CANDIDATES,
    max_attempts: int = AI_MAX_ATTEMPTS,
    seed: int = SEED,
    device: str = "cpu",
) -> list[str]:
    """Generate diverse exact-length candidates for seen or unseen domain pairs."""
    set_seed(seed)
    tensors = _model_context_tensors(
        seq_a, seq_b, domain_a, domain_b, vocab, domain_to_idx, device
    )
    length_probs = _valid_length_probabilities(model, tensors)
    allowed_lengths = np.arange(len(length_probs))

    reverse = {idx: token for token, idx in vocab.items()}
    invalid = [vocab[x] for x in ("<PAD>", "<BOS>", "<EOS>")]
    accepted: list[str] = []

    # Alternate high-probability lengths with probabilistic draws so candidates
    # are not eight stochastic copies at one length.
    ranked_lengths = [
        int(x) for x in np.argsort(length_probs)[::-1]
        if MIN_LINKER_LENGTH <= int(x) <= MAX_LINKER_LENGTH
    ]

    for attempt in range(max_attempts):
        if len(accepted) >= requested:
            break
        if attempt < len(ranked_lengths):
            target_length = ranked_lengths[attempt]
        else:
            target_length = int(np.random.choice(allowed_lengths, p=length_probs))

        indices = model.generate(
            *tensors,
            target_length=target_length,
            bos_idx=vocab["<BOS>"],
            invalid_output_indices=invalid,
            temperature=AI_TEMPERATURE,
            top_k=AI_TOP_K,
        )
        seq = "".join(reverse[i] for i in indices)
        if len(seq) == target_length and set(seq) <= AA and seq not in accepted:
            accepted.append(seq)

    return accepted


def select_domain_pair(
    a: str,
    b: str,
    domains: pd.DataFrame,
    proteins: pd.DataFrame,
    pairs: pd.DataFrame,
):
    """Choose sequences for the requested domain IDs.

    Prefer a natural co-occurring instance because it preserves a real domain
    context.  For a novel pair, use representative occurrences independently;
    the v3 generator can still condition on their terminal sequences via the
    <UNK_DOMAIN> fallback when needed.
    """
    if not pairs.empty and {"domain_a", "domain_b"} <= set(pairs.columns):
        natural = pairs[
            (pairs["domain_a"].astype(str) == str(a))
            & (pairs["domain_b"].astype(str) == str(b))
        ]
        if not natural.empty and {"domain_a_sequence", "domain_b_sequence"} <= set(natural.columns):
            row = natural.sort_values(["domain_a_length", "domain_b_length"]).iloc[0]
            return str(row["domain_a_sequence"]), str(row["domain_b_sequence"]), True

    protein_map = proteins.set_index("accession")

    def occurrence(domain_id: str) -> str:
        subset = domains[domains["domain_id"].astype(str) == str(domain_id)].copy()
        if "domain_type" in subset.columns:
            domain_rows = subset[subset["domain_type"].astype(str).str.lower() == "domain"]
            if not domain_rows.empty:
                subset = domain_rows
        if subset.empty:
            raise ValueError(f"Domain accession {domain_id!r} was not found.")
        if "representative" in subset.columns:
            rep = subset[subset["representative"].astype(bool)]
            if not rep.empty:
                subset = rep
        subset = subset.sort_values("length")
        for _, row in subset.iterrows():
            accession = str(row["accession"])
            if accession not in protein_map.index:
                continue
            full = str(protein_map.loc[accession]["sequence"])
            seq = full[int(row["start"]) - 1:int(row["end"])]
            if seq and set(seq) <= AA:
                return seq
        raise ValueError(f"No source sequence could be recovered for {domain_id!r}.")

    return occurrence(a), occurrence(b), False


def sequence_model_metrics(
    linker: str,
    seq_a: str,
    seq_b: str,
    domain_a: str,
    domain_b: str,
    model: ConditionalLSTM,
    vocab: dict[str, int],
    domain_to_idx: dict[str, int],
    device: str,
) -> tuple[float | None, float | None, float | None]:
    if not linker or not set(linker) <= AA or len(linker) > model.max_target_length:
        return None, None, None

    tensors = _model_context_tensors(
        seq_a, seq_b, domain_a, domain_b, vocab, domain_to_idx, device
    )
    left, right, left_idx, right_idx, lengths = tensors
    tokens = [vocab["<BOS>"]] + [vocab[x] for x in linker]
    targets = [vocab[x] for x in linker] + [vocab["<EOS>"]]
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    target = torch.tensor([targets], dtype=torch.long, device=device)
    target_length = torch.tensor([len(linker)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, length_logits = model(
            inp, left, right, left_idx, right_idx, lengths, target_length
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        ll_per_residue = float(gathered.mean().item())
        perplexity = float(math.exp(min(20.0, -ll_per_residue)))
        length_prob = float(torch.softmax(length_logits, dim=-1)[0, len(linker)].item())
    return ll_per_residue, perplexity, length_prob


def build_candidates(
    a: str,
    b: str,
    domains: pd.DataFrame,
    proteins: pd.DataFrame,
    pairs: pd.DataFrame,
    compatibility: dict,
    include_ai: bool = True,
    ai_device: str = "cpu",
    ai_seed: int = SEED,
):
    prior, count = pair_prior(compatibility, a, b)
    seq_a, seq_b, natural_pair_instance = select_domain_pair(a, b, domains, proteins, pairs)

    raw: list[tuple[str, str]] = []
    raw.extend((x, "empirical") for x in EMPIRICAL_LINKERS)
    raw.extend((x, "heuristic") for x in heuristic_linkers())

    model = vocab = domain_to_idx = None
    if include_ai:
        model, vocab, domain_to_idx, _ = load_generator(ai_device)
        ai_linkers = generate_ai_linkers(
            a,
            b,
            seq_a,
            seq_b,
            model,
            vocab,
            domain_to_idx,
            seed=ai_seed,
            device=ai_device,
        )
        raw.extend((x, "ai") for x in ai_linkers)

    seen: set[str] = set()
    rows: list[Candidate] = []
    for linker, category in raw:
        if linker in seen:
            continue
        seen.add(linker)
        if not (MIN_LINKER_LENGTH <= len(linker) <= MAX_LINKER_LENGTH and set(linker) <= AA):
            continue
        candidate = Candidate(
            domain_a=a,
            domain_b=b,
            linker=linker,
            category=category,
            sequence=seq_a + linker + seq_b,
            linker_length=len(linker),
            natural_pair_probability=prior,
            natural_pair_count=count,
            **features(linker),
        )
        if model is not None and vocab is not None and domain_to_idx is not None:
            ll, ppl, length_prob = sequence_model_metrics(
                linker,
                seq_a,
                seq_b,
                a,
                b,
                model,
                vocab,
                domain_to_idx,
                ai_device,
            )
            candidate.model_log_likelihood_per_residue = ll
            candidate.model_perplexity = ppl
            candidate.predicted_length_probability = length_prob
        rows.append(candidate)
        if len(rows) >= MAX_LINKER_CANDIDATES:
            break

    return rows, seq_a, seq_b, natural_pair_instance


def _residue_scores(pdb_text: str) -> list[float]:
    scores: dict[tuple[str, int, str], float] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        try:
            chain = line[21].strip()
            residue_number = int(line[22:26].strip())
            insertion = line[26].strip()
            bfactor = float(line[60:66].strip())
        except (ValueError, IndexError):
            continue
        scores[(chain, residue_number, insertion)] = bfactor
    ordered = sorted(scores.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    return [value for _, value in ordered]



def _ca_coordinates(pdb_text: str) -> np.ndarray:
    coords: list[tuple[float, float, float]] = []
    seen: set[tuple[str, int, str]] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        try:
            chain = line[21].strip()
            residue_number = int(line[22:26].strip())
            insertion = line[26].strip()
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except (ValueError, IndexError):
            continue
        key = (chain, residue_number, insertion)
        if key in seen:
            continue
        seen.add(key)
        coords.append(xyz)
    return np.asarray(coords, dtype=float)


def ca_lddt(reference: np.ndarray, model: np.ndarray, cutoff: float = 15.0) -> float | None:
    """CA-only lDDT-style similarity on a 0-100 scale.

    This uses the standard lDDT distance-error thresholds (0.5, 1, 2, 4 Å)
    over residue pairs closer than 15 Å in the reference.  It is invariant to
    rigid-body orientation and therefore appropriate for comparing an isolated
    domain with the same sequence embedded in a fusion.
    """
    if len(reference) != len(model) or len(reference) < 3:
        return None
    ref_delta = reference[:, None, :] - reference[None, :, :]
    model_delta = model[:, None, :] - model[None, :, :]
    ref_dist = np.linalg.norm(ref_delta, axis=-1)
    model_dist = np.linalg.norm(model_delta, axis=-1)
    mask = np.triu((ref_dist > 0) & (ref_dist < cutoff), k=1)
    if not np.any(mask):
        return None
    error = np.abs(ref_dist[mask] - model_dist[mask])
    per_pair = sum((error < threshold).astype(float) for threshold in (0.5, 1.0, 2.0, 4.0)) / 4.0
    return float(np.mean(per_pair) * 100.0)


def domain_preservation_lddt(
    baseline_pdb: str,
    fusion_pdb: str,
    fusion_start: int,
    domain_length: int,
) -> float | None:
    baseline = _ca_coordinates(baseline_pdb)
    fusion = _ca_coordinates(fusion_pdb)
    if len(baseline) < domain_length or len(fusion) < fusion_start + domain_length:
        return None
    return ca_lddt(baseline[:domain_length], fusion[fusion_start:fusion_start + domain_length])

def mean_plddt(pdb_text: str) -> float | None:
    values = _residue_scores(pdb_text)
    return float(np.mean(values)) if values else None


def plddt_by_region(pdb_text: str, domain_a_len: int, linker_len: int, domain_b_len: int):
    scores = _residue_scores(pdb_text)
    expected = domain_a_len + linker_len + domain_b_len
    if len(scores) < expected:
        return {
            "domain_a_plddt": None,
            "domain_b_plddt": None,
            "protein_plddt": None,
            "linker_plddt": None,
            "whole_plddt": None,
            "junction_plddt": None,
            "flank_plddt": None,
            "junction_drop": None,
        }
    scores = scores[:expected]

    def mean(values: Iterable[float]):
        values = list(values)
        return float(np.mean(values)) if values else None

    a_scores = scores[:domain_a_len]
    linker_start = domain_a_len
    linker_scores = scores[linker_start:linker_start + linker_len]
    b_scores = scores[linker_start + linker_len:expected]

    flank_indices = list(range(max(0, linker_start - 3), linker_start))
    flank_indices += list(
        range(linker_start + linker_len, min(expected, linker_start + linker_len + 3))
    )
    junction_indices = list(
        range(max(0, linker_start - 3), min(expected, linker_start + linker_len + 3))
    )

    junction = mean(scores[i] for i in junction_indices)
    flank = mean(scores[i] for i in flank_indices)
    return {
        "domain_a_plddt": mean(a_scores),
        "domain_b_plddt": mean(b_scores),
        "protein_plddt": mean(a_scores + b_scores),
        "linker_plddt": mean(linker_scores),
        "whole_plddt": mean(scores),
        "junction_plddt": junction,
        "flank_plddt": flank,
        "junction_drop": (flank - junction) if flank is not None and junction is not None else None,
    }


def fold_raw(sequence: str, pdb_path: Path) -> str:
    if len(sequence) > REMOTE_ESMFOLD_MAX_LENGTH:
        raise ValueError(
            f"Remote ESMFold accepts at most {REMOTE_ESMFOLD_MAX_LENGTH} residues; "
            f"this sequence has {len(sequence)}. Use shorter domain constructs or a local structure predictor."
        )
    response = requests.post(
        ESMFOLD_URL,
        data=sequence,
        timeout=ESMFOLD_TIMEOUT,
        headers={"Content-Type": "text/plain"},
    )
    response.raise_for_status()
    pdb = response.text
    pdb_path.write_text(pdb, encoding="utf-8")
    return pdb


def fold_sequence(
    sequence: str,
    pdb_path: Path,
    domain_a_len: int,
    linker_len: int,
    domain_b_len: int,
):
    pdb = fold_raw(sequence, pdb_path)
    return plddt_by_region(pdb, domain_a_len, linker_len, domain_b_len)


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def score_components(
    whole_plddt: float | None,
    junction_plddt: float | None,
    linker_plddt: float | None,
    domain_a_plddt: float | None,
    domain_b_plddt: float | None,
    junction_drop: float | None,
    domain_a_baseline_plddt: float | None = None,
    domain_b_baseline_plddt: float | None = None,
    domain_preservation_score: float | None = None,
) -> dict[str, float | None]:
    domain_values = [x for x in (domain_a_plddt, domain_b_plddt) if x is not None]
    domain_mean = float(np.mean(domain_values)) if domain_values else None

    preservation = domain_preservation_score
    if preservation is None and (
        domain_a_plddt is not None
        and domain_b_plddt is not None
        and domain_a_baseline_plddt is not None
        and domain_b_baseline_plddt is not None
    ):
        # Fallback when baseline coordinates are unavailable.  The preferred
        # preservation metric is CA-lDDT, assigned before scoring in main().
        loss_a = max(0.0, domain_a_baseline_plddt - domain_a_plddt)
        loss_b = max(0.0, domain_b_baseline_plddt - domain_b_plddt)
        preservation = clamp(100.0 - 2.5 * (loss_a + loss_b) / 2.0)

    drop_score = None
    if junction_drop is not None:
        drop_score = clamp(100.0 - 4.0 * max(0.0, junction_drop))

    metrics = {
        "junction": (junction_plddt, 0.25),
        "domains": (domain_mean, 0.20),
        "linker": (linker_plddt, 0.10),
        "whole": (whole_plddt, 0.10),
        "preservation": (preservation, 0.25),
        "junction_drop": (drop_score, 0.10),
    }
    weighted = [(clamp(v), w) for v, w in metrics.values() if v is not None]
    absolute = None
    if weighted:
        denom = sum(w for _, w in weighted)
        absolute = sum(v * w for v, w in weighted) / denom

    return {
        "domain_preservation_score": preservation,
        "junction_drop_score": drop_score,
        "absolute_score": absolute,
    }


def structural_score(
    whole_plddt: float | None,
    junction_plddt: float | None,
    junction_drop: float | None,
    linker_plddt: float | None = None,
    domain_a_plddt: float | None = None,
    domain_b_plddt: float | None = None,
    domain_a_baseline_plddt: float | None = None,
    domain_b_baseline_plddt: float | None = None,
    domain_preservation_score: float | None = None,
) -> float | None:
    """Absolute 0-100 structural score; retained as a public helper."""
    return score_components(
        whole_plddt,
        junction_plddt,
        linker_plddt,
        domain_a_plddt,
        domain_b_plddt,
        junction_drop,
        domain_a_baseline_plddt,
        domain_b_baseline_plddt,
        domain_preservation_score,
    )["absolute_score"]


def apply_scores(candidates: list[Candidate]) -> None:
    """Compute absolute quality and a tie-aware within-pair comparative score.

    `absolute_score` remains physically interpretable on the pLDDT scale.  `score`
    is a percentile rank after quantizing absolute scores to 0.1 points, so
    changes at the 1e-6 level cannot alter the ranking.
    """
    for c in candidates:
        parts = score_components(
            c.whole_plddt,
            c.junction_plddt,
            c.linker_plddt,
            c.domain_a_plddt,
            c.domain_b_plddt,
            c.junction_drop,
            c.domain_a_baseline_plddt,
            c.domain_b_baseline_plddt,
            c.domain_preservation_score,
        )
        c.domain_preservation_score = parts["domain_preservation_score"]
        c.junction_drop_score = parts["junction_drop_score"]
        c.absolute_score = parts["absolute_score"]

    valid = [(i, c.absolute_score) for i, c in enumerate(candidates) if c.absolute_score is not None]
    if not valid:
        return

    values = pd.Series(
        [round(float(v) / PLDDT_RANK_RESOLUTION) * PLDDT_RANK_RESOLUTION for _, v in valid]
    )
    ranks = values.rank(method="average", pct=True) * 100.0
    best = max(float(v) for _, v in valid)
    for (idx, value), rank in zip(valid, ranks):
        candidates[idx].score = float(rank)
        candidates[idx].score_delta_from_best = best - float(value)


def print_quality_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    columns = [
        "category", "linker", "linker_length", "whole_plddt", "junction_plddt",
        "domain_a_plddt", "domain_b_plddt", "linker_plddt", "junction_drop",
        "domain_a_preservation_lddt", "domain_b_preservation_lddt",
        "domain_preservation_score", "absolute_score", "score",
        "model_perplexity", "predicted_length_probability",
    ]
    available = [x for x in columns if x in df.columns]
    print("\n" + "=" * 120)
    print("STRUCTURAL QUALITY (absolute_score) AND WITHIN-PAIR RANK (score)")
    print("=" * 120)
    print(df[available].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 120)


def choose_domains(domains: pd.DataFrame):
    counts = (
        domains.groupby(["domain_id", "domain_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print("\nTop domains:")
    print(counts.head(30).to_string(index=False))
    return input("\nDomain A accession: ").strip(), input("Domain B accession: ").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate and compare sequence-conditioned AI and heuristic protein linkers."
    )
    parser.add_argument("--domain-a")
    parser.add_argument("--domain-b")
    parser.add_argument("--fold", action="store_true")
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    domains, proteins, pairs, compatibility = load_data()
    a, b = (
        (args.domain_a, args.domain_b)
        if args.domain_a and args.domain_b
        else choose_domains(domains)
    )

    candidates, seq_a, seq_b, natural_instance = build_candidates(
        a,
        b,
        domains,
        proteins,
        pairs,
        compatibility,
        include_ai=not args.no_ai,
        ai_device=args.device,
        ai_seed=args.seed,
    )
    if not candidates:
        raise RuntimeError("No linker candidates were generated.")

    RESULTS_DIR.mkdir(exist_ok=True)
    print(
        f"\nGenerated {len(candidates)} unique candidates for {a} -> {b}. "
        f"Natural co-occurring source instance: {natural_instance}."
    )

    baseline_a = baseline_b = None
    baseline_pdb_a = baseline_pdb_b = None
    if args.fold:
        print("\nFolding isolated domains once to establish domain-preservation baselines...")
        try:
            baseline_pdb_a = fold_raw(seq_a, RESULTS_DIR / "baseline_domain_a.pdb")
            baseline_a = mean_plddt(baseline_pdb_a)
            time.sleep(max(0.0, args.sleep))
            baseline_pdb_b = fold_raw(seq_b, RESULTS_DIR / "baseline_domain_b.pdb")
            baseline_b = mean_plddt(baseline_pdb_b)
            time.sleep(max(0.0, args.sleep))
            print(f"Baseline pLDDT: domain A={baseline_a:.2f}, domain B={baseline_b:.2f}")
        except Exception as exc:
            print(f"WARNING: isolated-domain baseline failed: {exc}")

        print("\nRunning ESMFold for fusion candidates...")
        for i, c in enumerate(candidates, 1):
            c.domain_a_baseline_plddt = baseline_a
            c.domain_b_baseline_plddt = baseline_b
            safe = re.sub(r"[^A-Z]", "_", c.linker)
            path = RESULTS_DIR / f"candidate_{i:03d}_{safe}.pdb"
            try:
                info = fold_sequence(c.sequence, path, len(seq_a), c.linker_length, len(seq_b))
                for key, value in info.items():
                    setattr(c, key, value)
                c.pdb_path = str(path)
                fusion_pdb = path.read_text(encoding="utf-8")
                if baseline_pdb_a is not None:
                    c.domain_a_preservation_lddt = domain_preservation_lddt(
                        baseline_pdb_a, fusion_pdb, 0, len(seq_a)
                    )
                if baseline_pdb_b is not None:
                    c.domain_b_preservation_lddt = domain_preservation_lddt(
                        baseline_pdb_b, fusion_pdb, len(seq_a) + c.linker_length, len(seq_b)
                    )
                preserved = [
                    x for x in (c.domain_a_preservation_lddt, c.domain_b_preservation_lddt)
                    if x is not None
                ]
                c.domain_preservation_score = float(np.mean(preserved)) if len(preserved) == 2 else None
                parts = score_components(
                    c.whole_plddt,
                    c.junction_plddt,
                    c.linker_plddt,
                    c.domain_a_plddt,
                    c.domain_b_plddt,
                    c.junction_drop,
                    baseline_a,
                    baseline_b,
                    c.domain_preservation_score,
                )
                print(
                    f"{i:02d}/{len(candidates)} | {c.category:9s} | {c.linker:20s} | "
                    f"junction={c.junction_plddt:.2f} | whole={c.whole_plddt:.2f} | "
                    f"absolute={parts['absolute_score']:.2f}"
                )
            except Exception as exc:
                c.fold_error = str(exc)
                print(f"ESMFold failed for {c.linker}: {exc}")
            time.sleep(max(0.0, args.sleep))
    else:
        print("Structural fields are empty because --fold was not supplied.")

    apply_scores(candidates)
    out = pd.DataFrame([asdict(c) for c in candidates])
    out = out.sort_values(
        ["score", "absolute_score"], ascending=False, na_position="last"
    ).reset_index(drop=True)
    output_path = RESULTS_DIR / "fusion_candidates.csv"
    out.to_csv(output_path, index=False)
    print_quality_summary(out)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
