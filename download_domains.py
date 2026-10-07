from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    CONTEXT_RESIDUES,
    DATA_DIR,
    GO_TERMS,
    MAX_BROAD_PROTEINS,
    MAX_DOMAIN_LENGTH,
    MAX_NATURAL_JUNCTION_LENGTH,
    MAX_PROTEIN_LENGTH,
    MAX_PROTEINS_PER_CLASS,
    MIN_DOMAIN_LENGTH,
    MIN_NATURAL_JUNCTION_LENGTH,
    MIN_PROTEIN_LENGTH,
)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
INTERPRO_URL = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{accession}/"
AA = set("ACDEFGHIKLMNPQRSTVWY")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "DomainFusionResearch/3.0 (student research pipeline)",
    "Accept": "application/json",
})


def _next_link(response: requests.Response) -> str | None:
    """Return RFC5988 rel=next URL used by the UniProt REST API."""
    link = response.headers.get("Link", "")
    for part in link.split(","):
        if 'rel="next"' not in part:
            continue
        left = part.find("<")
        right = part.find(">", left + 1)
        if left >= 0 and right > left:
            return part[left + 1:right]
    return None


def fetch_uniprot_query(query: str, source_class: str, max_records: int) -> list[dict[str, Any]]:
    """Fetch a paginated UniProtKB result set.

    We keep only canonical 20-aa sequences.  Pagination is essential because a
    single UniProt REST response is not a complete dataset export.
    """
    fields = "accession,id,protein_name,sequence,length"
    params = {
        "query": query,
        "format": "json",
        "size": min(500, max_records),
        "fields": fields,
    }

    rows: list[dict[str, Any]] = []
    url: str | None = UNIPROT_URL
    first = True
    while url and len(rows) < max_records:
        response = SESSION.get(url, params=params if first else None, timeout=90)
        response.raise_for_status()
        first = False

        for item in response.json().get("results", []):
            seq = str(item.get("sequence", {}).get("value", "")).upper()
            accession = str(item.get("primaryAccession", ""))
            if not accession or not seq or not set(seq) <= AA:
                continue
            if not MIN_PROTEIN_LENGTH <= len(seq) <= MAX_PROTEIN_LENGTH:
                continue

            description = item.get("proteinDescription", {})
            recommended = description.get("recommendedName", {})
            full_name = recommended.get("fullName", {}).get("value", "")
            rows.append({
                "accession": accession,
                "protein_id": item.get("uniProtkbId", ""),
                "protein_name": full_name,
                "sequence": seq,
                "length": len(seq),
                "source_class": source_class,
            })
            if len(rows) >= max_records:
                break

        url = _next_link(response)

    return rows


def fetch_seed_proteins() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    broad_query = (
        f"reviewed:true AND length:[{MIN_PROTEIN_LENGTH} TO {MAX_PROTEIN_LENGTH}]"
    )
    print("Downloading broad reviewed UniProtKB seed set...")
    rows.extend(fetch_uniprot_query(broad_query, "broad_swissprot", MAX_BROAD_PROTEINS))

    for class_name, go_id in GO_TERMS.items():
        go_clean = go_id.replace("GO:", "")
        query = (
            f"reviewed:true AND go:{go_clean} "
            f"AND length:[{MIN_PROTEIN_LENGTH} TO {MAX_PROTEIN_LENGTH}]"
        )
        print(f"Downloading enrichment set {class_name} ({go_id})...")
        try:
            rows.extend(fetch_uniprot_query(query, class_name, MAX_PROTEINS_PER_CLASS))
        except Exception as exc:
            print(f"WARNING: {class_name}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No proteins were downloaded from UniProtKB.")

    # Keep one accession record and then remove exact sequence duplicates.  This
    # reduces near-duplicate supervision and makes group validation less leaky.
    df = df.drop_duplicates("accession").drop_duplicates("sequence").reset_index(drop=True)
    return df


def extract_fragments(location: dict[str, Any]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for fragment in location.get("fragments") or []:
        start = fragment.get("start")
        end = fragment.get("end")
        if isinstance(start, dict):
            start = start.get("value")
        if isinstance(end, dict):
            end = end.get("value")
        try:
            start_i, end_i = int(start), int(end)
        except (TypeError, ValueError):
            continue
        if 1 <= start_i <= end_i:
            result.append((start_i, end_i))
    return result


def fetch_interpro(accession: str) -> list[dict[str, Any]]:
    """Fetch all integrated InterPro annotations for one UniProt accession."""
    url: str | None = INTERPRO_URL.format(accession=accession)
    params: dict[str, Any] | None = {"page_size": 200}
    domains: list[dict[str, Any]] = []

    while url:
        response = SESSION.get(url, params=params, timeout=90)
        params = None
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()

        for result in payload.get("results", []):
            metadata = result.get("metadata", {})
            entry_acc = str(metadata.get("accession", ""))
            name = str(metadata.get("name", ""))
            entry_type = str(metadata.get("type", "")).strip().lower()
            if entry_type not in {"domain", "family", "homologous_superfamily"}:
                continue

            for protein in result.get("proteins", []):
                for location in protein.get("entry_protein_locations", []):
                    representative = bool(location.get("representative", True))
                    for start, end in extract_fragments(location):
                        length = end - start + 1
                        if not MIN_DOMAIN_LENGTH <= length <= MAX_DOMAIN_LENGTH:
                            continue
                        domains.append({
                            "accession": accession,
                            "domain_id": entry_acc,
                            "domain_name": name,
                            "domain_type": entry_type,
                            "start": start,
                            "end": end,
                            "length": length,
                            "representative": representative,
                        })

        url = payload.get("next")

    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in domains:
        unique[(row["domain_id"], row["start"], row["end"])] = row
    return list(unique.values())


def _nearest_nonoverlapping_pairs(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return nearest downstream non-overlapping domain annotations.

    The old implementation zipped annotation rows.  A nested/overlapping record
    between two true domains then caused the real adjacency to disappear.  Here,
    each left domain searches for the nearest downstream start after its end.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i, left in enumerate(records):
        valid = [right for right in records[i + 1:] if int(right["start"]) > int(left["end"])]
        if not valid:
            continue
        nearest_start = min(int(x["start"]) for x in valid)
        nearest = [x for x in valid if int(x["start"]) == nearest_start]
        # Multiple InterPro entries can map to the same physical interval.  Keep
        # a deterministic representative rather than multiplying one junction.
        right = sorted(
            nearest,
            key=lambda x: (not bool(x.get("representative", True)), -int(x["length"]), str(x["domain_id"])),
        )[0]
        pairs.append((left, right))
    return pairs


def make_domain_pairs(proteins: pd.DataFrame, domains: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Integrated InterPro "domain" entries provide boundaries suitable for
    # linker extraction.  Family/superfamily annotations are retained in
    # domains.csv for lookup, but are not treated as physical linker boundaries.
    domain_only = domains[domains["domain_type"].astype(str).str.lower() == "domain"].copy()

    protein_map = proteins.set_index("accession")
    for accession, group in domain_only.groupby("accession"):
        if accession not in protein_map.index:
            continue
        protein = protein_map.loc[accession]
        sequence = str(protein["sequence"])

        records = (
            group.sort_values(["start", "end", "domain_id"])
            .drop_duplicates(["domain_id", "start", "end"])
            .to_dict("records")
        )

        seen_boundaries: set[tuple[int, int]] = set()
        for left, right in _nearest_nonoverlapping_pairs(records):
            junction_start = int(left["end"]) + 1
            junction_end = int(right["start"]) - 1
            junction = sequence[junction_start - 1:junction_end]

            if not MIN_NATURAL_JUNCTION_LENGTH <= len(junction) <= MAX_NATURAL_JUNCTION_LENGTH:
                continue
            if not set(junction) <= AA:
                continue

            boundary_key = (int(left["end"]), int(right["start"]))
            if boundary_key in seen_boundaries:
                continue
            seen_boundaries.add(boundary_key)

            left_seq = sequence[int(left["start"]) - 1:int(left["end"])]
            right_seq = sequence[int(right["start"]) - 1:int(right["end"])]
            if not left_seq or not right_seq or not set(left_seq + right_seq) <= AA:
                continue

            rows.append({
                "protein_accession": accession,
                "source_class": protein.get("source_class", ""),
                "domain_a": str(left["domain_id"]),
                "domain_a_name": left["domain_name"],
                "domain_a_start": int(left["start"]),
                "domain_a_end": int(left["end"]),
                "domain_a_length": len(left_seq),
                "domain_b": str(right["domain_id"]),
                "domain_b_name": right["domain_name"],
                "domain_b_start": int(right["start"]),
                "domain_b_end": int(right["end"]),
                "domain_b_length": len(right_seq),
                "domain_a_sequence": left_seq,
                "domain_b_sequence": right_seq,
                "left_context": left_seq[-CONTEXT_RESIDUES:],
                "right_context": right_seq[:CONTEXT_RESIDUES],
                "junction": junction,
                "junction_length": len(junction),
            })

    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    proteins_df = fetch_seed_proteins()
    proteins_df.to_csv(DATA_DIR / "proteins.csv", index=False)

    print(f"Unique proteins retained: {len(proteins_df)}")
    print("Downloading paginated InterPro annotations (resumable cache)...")

    cache_path = DATA_DIR / "interpro_annotations_raw.csv"
    processed_path = DATA_DIR / "interpro_processed.txt"
    processed: set[str] = set()
    if processed_path.exists():
        processed = {x.strip() for x in processed_path.read_text(encoding="utf-8").splitlines() if x.strip()}

    for accession in tqdm(proteins_df["accession"]):
        accession = str(accession)
        if accession in processed:
            continue
        try:
            rows = fetch_interpro(accession)
            if rows:
                pd.DataFrame(rows).to_csv(
                    cache_path,
                    mode="a",
                    header=not cache_path.exists(),
                    index=False,
                )
            with processed_path.open("a", encoding="utf-8") as handle:
                handle.write(accession + "\n")
            processed.add(accession)
        except Exception as exc:
            print(f"\nWARNING {accession}: {exc}")
        time.sleep(0.03)

    if not cache_path.exists():
        raise RuntimeError("No InterPro annotations were returned.")
    domains_df = pd.read_csv(cache_path).drop_duplicates(
        ["accession", "domain_id", "start", "end"]
    ).reset_index(drop=True)
    domains_df.to_csv(DATA_DIR / "domains.csv", index=False)

    pairs_df = make_domain_pairs(proteins_df, domains_df)
    if pairs_df.empty:
        raise RuntimeError("No valid natural domain junctions were extracted.")
    pairs_df.to_csv(DATA_DIR / "domain_pairs.csv", index=False)

    print(f"InterPro annotations: {len(domains_df)}")
    print(f"Natural domain junctions: {len(pairs_df)}")
    print(f"Unique ordered domain pairs: {pairs_df[['domain_a', 'domain_b']].drop_duplicates().shape[0]}")
    print("Created data/proteins.csv, data/domains.csv, data/domain_pairs.csv")


if __name__ == "__main__":
    main()
