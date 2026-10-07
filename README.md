# Domain Fusion Linker Pipeline v3

This revision replaces the old domain-ID-only LSTM with a sequence-conditioned linker generator and replaces the near-degenerate structural score with a broader structural comparison.

## What changed

The old checkpoint encoded only the ordered InterPro IDs in the initial LSTM hidden state. That is a sparse supervision problem: most exact domain pairs occur too few times to learn useful embeddings, unseen pairs cannot be generated at all, and the conditioning signal is easy for the decoder to forget. The v3 model instead encodes the C-terminal boundary sequence of domain A and the N-terminal boundary sequence of domain B with bidirectional GRUs, adds recurrently observed InterPro IDs and domain lengths as auxiliary context, predicts linker length explicitly, and concatenates the context vector to every decoder step. Rare or unseen InterPro IDs map to `<UNK_DOMAIN>` while their sequence context remains active.

The training data builder now uses proteins up to 1200 aa, follows UniProt and InterPro pagination, uses InterPro `domain` entries as physical boundaries, searches for the nearest downstream non-overlapping domain instead of simply zipping annotation rows, stores the actual domain sequences and 32-residue boundary contexts, and writes a resumable InterPro cache. Natural junctions from 2 to 30 residues are used for training; generation is constrained to 4 to 20 residues by default.

Validation is grouped by source protein accession, not randomly by junction row. This prevents junctions from the same protein context from leaking across train and validation sets. Training reports `conditioning_gap`, defined as the validation token loss with shuffled domain contexts minus the loss with the correct contexts. A positive gap is evidence that the decoder is using context; it is not by itself proof of biological validity.

## Scoring

`absolute_score` is reported on a 0–100 scale. It combines junction pLDDT, mean domain pLDDT, linker pLDDT, whole-fusion pLDDT, junction-drop quality, and a domain-preservation term. The preferred preservation term is a CA-only lDDT-style comparison of each isolated predicted domain against the same residues inside the fusion, using a 15 Å inclusion radius and the 0.5/1/2/4 Å distance-error thresholds. If coordinate preservation cannot be calculated, the pipeline falls back to a weaker pLDDT-retention estimate.

`score` is deliberately a different quantity: it is a within-domain-pair percentile rank computed after quantizing `absolute_score` to 0.1 points. Therefore changes such as 0.0000055 cannot alter the rank. This gives useful separation without pretending that floating-point-scale pLDDT differences are meaningful.

The generator likelihood and predicted linker-length probability are reported for every candidate, including heuristics, but they are not part of the structural score. This avoids building a ranking that is mechanically biased toward the AI model that generated the sequence.

## Run order

Create a fresh environment and install dependencies:

```bash
pip install -r requirements.txt
```

Build the dataset. The InterPro stage is resumable through `data/interpro_annotations_raw.csv` and `data/interpro_processed.txt`:

```bash
python download_domains.py
python train_compatibility.py
```

Train the v3 generator. CUDA is used by default when PyTorch detects it:

```bash
python train_generator.py --epochs 60 --batch-size 64
```

Generate and evaluate a domain pair:

```bash
python domain_fusion.py --domain-a IPRXXXXXX --domain-b IPRYYYYYY --fold
```

Analyze the resulting candidate table:

```bash
python analysis.py --plot
```

The public ESM Metagenomic Atlas folding endpoint accepts sequences up to 400 residues. The pipeline now checks this explicitly. For longer domain constructs, use shorter constructs or a local structure-prediction workflow rather than treating the API error as a linker failure.

## Important interpretation

A high score means that a candidate looks favorable under this computational screen. It does not establish expression, solubility, activity, proteolytic stability, or functional independence of the fused domains. Those require experimental validation. Flexible GS-rich, rigid EAAAK-type and proline-rich linker families have different intended physical behaviors, so no sequence-only or structure-confidence score can define a universally optimal linker.

## References underlying the redesign

Chen X, Zaro JL, Shen WC. Fusion protein linkers: property, design and functionality. *Advanced Drug Delivery Reviews*. 2013;65(10):1357-1369. doi:10.1016/j.addr.2012.09.039.

Mariani V, Biasini M, Barbato A, Schwede T. lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests. *Bioinformatics*. 2013;29(21):2722-2728. doi:10.1093/bioinformatics/btt473.

InterPro API documentation: https://interpro-documentation.readthedocs.io/en/latest/api.html

EMBL-EBI UniProt programmatic access: https://www.ebi.ac.uk/training/online/courses/uniprot-exploring-protein-sequence-and-functional-info/getting-data-from-uniprot/accessing-uniprot-data-programmatically/

ESM Metagenomic Atlas API documentation: https://esmatlas.com/about
