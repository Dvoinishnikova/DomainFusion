from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

for directory in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    directory.mkdir(exist_ok=True)

# Dataset construction.  The generator is no longer trained on GO-class labels;
# the GO queries are only used to diversify the seed set with experimentally
# curated proteins.  A broad Swiss-Prot query is added by download_domains.py.
GO_TERMS = {
    "kinase_activity": "GO:0016301",
    "dna_binding": "GO:0003677",
    "rna_binding": "GO:0003723",
    "metal_binding": "GO:0008270",
    "calcium_binding": "GO:0005509",
}

MAX_PROTEINS_PER_CLASS = 300
MAX_BROAD_PROTEINS = 1200
MIN_PROTEIN_LENGTH = 80
MAX_PROTEIN_LENGTH = 1200

MIN_DOMAIN_LENGTH = 20
MAX_DOMAIN_LENGTH = 600
MIN_NATURAL_JUNCTION_LENGTH = 2
MAX_NATURAL_JUNCTION_LENGTH = 30
CONTEXT_RESIDUES = 32

# Candidate generation constraints.  These are intentionally narrower than the
# training range so the model can learn from somewhat longer natural junctions
# without emitting impractically long synthetic linkers by default.
MIN_LINKER_LENGTH = 4
MAX_LINKER_LENGTH = 20
MAX_LINKER_CANDIDATES = 40
AI_REQUESTED_CANDIDATES = 8

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_TIMEOUT = 180
REMOTE_ESMFOLD_MAX_LENGTH = 400

# Relative ranking ignores changes below this resolution, which prevents
# floating-point-scale pLDDT differences from producing fake precision.
PLDDT_RANK_RESOLUTION = 0.1

SEED = 42
