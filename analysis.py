from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import PLDDT_RANK_RESOLUTION, RESULTS_DIR


def _metric_summary(group: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return {
        f"{column}_mean": float(values.mean()) if len(values) else None,
        f"{column}_median": float(values.median()) if len(values) else None,
        f"{column}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None,
        f"{column}_best": float(values.max()) if len(values) else None,
    }


def comparative_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in df.groupby("category", sort=True):
        row = {
            "category": category,
            "generated_candidates": int(len(group)),
            "valid_structural_candidates": int(pd.to_numeric(group["absolute_score"], errors="coerce").notna().sum()),
        }
        for metric in (
            "absolute_score",
            "score",
            "whole_plddt",
            "junction_plddt",
            "domain_a_plddt",
            "domain_b_plddt",
            "linker_plddt",
            "junction_drop",
            "domain_a_preservation_lddt",
            "domain_b_preservation_lddt",
            "domain_preservation_score",
            "model_perplexity",
            "predicted_length_probability",
            "linker_length",
        ):
            if metric in group.columns:
                row.update(_metric_summary(group, metric))
        rows.append(row)
    return pd.DataFrame(rows)


def score_resolution_report(df: pd.DataFrame) -> pd.DataFrame:
    absolute = pd.to_numeric(df["absolute_score"], errors="coerce").dropna()
    if absolute.empty:
        return pd.DataFrame([{
            "n": 0,
            "absolute_range": np.nan,
            "absolute_std": np.nan,
            "quantization_resolution": PLDDT_RANK_RESOLUTION,
            "unique_quantized_scores": 0,
            "comparative_score_range": np.nan,
        }])

    quantized = (absolute / PLDDT_RANK_RESOLUTION).round() * PLDDT_RANK_RESOLUTION
    relative = pd.to_numeric(df["score"], errors="coerce").dropna()
    return pd.DataFrame([{
        "n": int(len(absolute)),
        "absolute_range": float(absolute.max() - absolute.min()),
        "absolute_std": float(absolute.std(ddof=1)) if len(absolute) > 1 else 0.0,
        "quantization_resolution": PLDDT_RANK_RESOLUTION,
        "unique_quantized_scores": int(quantized.nunique()),
        "comparative_score_range": float(relative.max() - relative.min()) if len(relative) else np.nan,
    }])


def _weighted_average(row: pd.Series, excluded: str | None = None) -> float:
    domain_vals = [row.get("domain_a_plddt"), row.get("domain_b_plddt")]
    domain_vals = [float(x) for x in domain_vals if pd.notna(x)]
    domain_mean = float(np.mean(domain_vals)) if domain_vals else np.nan

    components = {
        "junction": (row.get("junction_plddt"), 0.25),
        "domains": (domain_mean, 0.20),
        "linker": (row.get("linker_plddt"), 0.10),
        "whole": (row.get("whole_plddt"), 0.10),
        "preservation": (row.get("domain_preservation_score"), 0.25),
        "junction_drop": (row.get("junction_drop_score"), 0.10),
    }
    valid = [
        (float(value), weight)
        for name, (value, weight) in components.items()
        if name != excluded and pd.notna(value)
    ]
    if not valid:
        return np.nan
    return sum(value * weight for value, weight in valid) / sum(weight for _, weight in valid)


def ablation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = [None, "junction", "domains", "linker", "whole", "preservation", "junction_drop"]
    for excluded in variants:
        scores = df.apply(lambda row: _weighted_average(row, excluded), axis=1).dropna()
        rows.append({
            "variant": "full" if excluded is None else f"without_{excluded}",
            "n": int(len(scores)),
            "mean": float(scores.mean()) if len(scores) else np.nan,
            "median": float(scores.median()) if len(scores) else np.nan,
            "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0 if len(scores) == 1 else np.nan,
        })
    return pd.DataFrame(rows)


def sequence_correlations(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "hydrophobic_fraction",
        "charged_fraction",
        "proline_fraction",
        "gly_ser_fraction",
        "linker_length",
        "model_perplexity",
        "predicted_length_probability",
    ]
    targets = [
        "absolute_score",
        "whole_plddt",
        "junction_plddt",
        "junction_drop",
        "linker_plddt",
        "domain_preservation_score",
    ]

    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        for target in targets:
            if target not in df.columns:
                continue
            pair = df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
            can_correlate = (
                len(pair) > 1
                and pair[feature].nunique(dropna=True) > 1
                and pair[target].nunique(dropna=True) > 1
            )
            rows.append({
                "feature": feature,
                "target": target,
                "n": len(pair),
                "spearman_rho": (
                    float(pair[feature].corr(pair[target], method="spearman"))
                    if can_correlate else np.nan
                ),
            })
    return pd.DataFrame(rows)


def case_studies(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["absolute_score"]).sort_values(
        ["score", "absolute_score"], ascending=False
    ).reset_index(drop=True)
    if work.empty:
        return work
    indices = sorted(set([0, len(work) // 2, len(work) - 1]))
    selected = work.iloc[indices].copy()
    selected.insert(0, "rank_position", [i + 1 for i in indices])
    columns = [
        "rank_position",
        "category",
        "linker",
        "linker_length",
        "absolute_score",
        "score",
        "whole_plddt",
        "junction_plddt",
        "domain_a_plddt",
        "domain_b_plddt",
        "domain_preservation_score",
        "junction_drop",
        "linker_plddt",
        "model_perplexity",
        "pdb_path",
    ]
    return selected[[c for c in columns if c in selected.columns]]


def make_plot(df: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    categories = sorted(df["category"].dropna().unique())
    groups = [
        pd.to_numeric(df.loc[df["category"] == category, "absolute_score"], errors="coerce").dropna().values
        for category in categories
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    if any(len(group) for group in groups):
        ax.boxplot(groups, tick_labels=categories)
    ax.set_xlabel("Candidate category")
    ax.set_ylabel("Absolute structural score (0–100)")
    ax.set_title("Fusion-linker structural score distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze fusion-candidate experiments.")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "fusion_candidates.csv")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing result file: {args.input}")

    df = pd.read_csv(args.input)
    required = {
        "category",
        "absolute_score",
        "score",
        "whole_plddt",
        "junction_plddt",
        "linker_plddt",
        "domain_a_plddt",
        "domain_b_plddt",
        "junction_drop",
        "junction_drop_score",
        "domain_preservation_score",
        "linker_length",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Result file is missing v3 columns: {sorted(missing)}")

    RESULTS_DIR.mkdir(exist_ok=True)

    comparison = comparative_analysis(df)
    comparison.to_csv(RESULTS_DIR / "comparison_by_category.csv", index=False)

    resolution = score_resolution_report(df)
    resolution.to_csv(RESULTS_DIR / "score_resolution.csv", index=False)

    ablation = ablation_analysis(df)
    ablation.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)

    seq_corr = sequence_correlations(df)
    seq_corr.to_csv(RESULTS_DIR / "sequence_correlations.csv", index=False)

    cases = case_studies(df)
    cases.to_csv(RESULTS_DIR / "case_studies.csv", index=False)

    print("\nCOMPARATIVE ANALYSIS")
    print(comparison.to_string(index=False))
    print("\nSCORE RESOLUTION")
    print(resolution.to_string(index=False))
    print("\nABLATION")
    print(ablation.to_string(index=False))
    print("\nSEQUENCE/MODEL CORRELATIONS")
    print(seq_corr.to_string(index=False))
    print("\nCASE STUDIES")
    print(cases.to_string(index=False))

    if args.plot:
        try:
            make_plot(df, RESULTS_DIR / "score_distributions.png")
            print(f"\nSaved: {RESULTS_DIR / 'score_distributions.png'}")
        except ImportError:
            print("Plot skipped: install matplotlib to enable --plot.")

    print(f"\nSaved analysis tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
