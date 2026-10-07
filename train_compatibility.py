from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from config import DATA_DIR, MODELS_DIR


def main() -> None:
    path = DATA_DIR / "domain_pairs.csv"
    if not path.exists():
        raise FileNotFoundError("Run download_domains.py first.")

    df = pd.read_csv(path)
    required = {"domain_a", "domain_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    pair_counts = Counter(
        zip(df["domain_a"].astype(str), df["domain_b"].astype(str))
    )
    outgoing = Counter(df["domain_a"].astype(str))

    model = {}
    for (a, b), count in pair_counts.items():
        model.setdefault(a, {})[b] = {
            "count": int(count),
            "conditional_probability": float(count / outgoing[a]),
        }

    output = {
        "description": "Natural domain adjacency prior; not a functional guarantee.",
        "num_pairs": int(len(df)),
        "num_unique_domain_pairs": int(len(pair_counts)),
        "pairs": model,
    }

    path = MODELS_DIR / "domain_compatibility.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {path}")
    print(f"Natural pairs: {len(df)}")
    print(f"Unique pairs: {len(pair_counts)}")


if __name__ == "__main__":
    main()
