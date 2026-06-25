"""
Central configuration for the InvasiveLens pipeline.

Nothing in this file requires network access to import. Paths are relative
to the project root so the same config works locally and on Colab (just
change DATA_ROOT).
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data_raw"          # raw downloaded images land here
MANIFEST_DIR = PROJECT_ROOT / "manifests"       # CSVs describing image/label/region
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (DATA_ROOT, MANIFEST_DIR, CHECKPOINT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Class list — validated invasive-native pairs for classification
# Each invasive species is paired with a candidate native/non-invasive look-alike.
# Confidence reflects how well-documented the visual confusion is in literature.
# See src/data/inspect_counts.py for per-species GBIF availability validation.
# ---------------------------------------------------------------------------
CANDIDATE_PAIRS = [
    {
        "invasive": "Arundo donax",
        "native": "Phragmites australis",
        "pair_confidence": "high",
        "notes": "Distinguish by stem diameter; hairy vs. glabrous floral glumes. "
                 "Both in Portugal; documented confusion in ID literature.",
    },
    {
        "invasive": "Cortaderia selloana",
        "native": "Ammophila arenaria",
        "pair_confidence": "medium",
        "notes": "Confusion strongest in juvenile (non-flowering) plants on dunes. "
                 "Both sand-tolerant coastal species.",
    },
    {
        "invasive": "Acacia dealbata",
        "native": "Acacia longifolia",  # Also invasive but more established/naturalised in PT — paired for visual confusion, not invasive-vs-native
        "pair_confidence": "medium",
        "notes": "Both Acacias are invasive in Portugal; A. longifolia is more "
                 "established/naturalised while A. dealbata is more aggressively "
                 "spreading. Paired because field teams frequently confuse the two. "
                 "A. melanoxylon less common; focus on dealbata vs longifolia distinction.",
    },
]

# Image size expected by both EfficientNetV2-S and ResNet50 ImageNet weights
IMAGE_SIZE = 224
# Windows multiprocessing in DataLoader often fails; use 0 there.
NUM_WORKERS = 0 if os.name == "nt" else 2
BATCH_SIZE = 32
SEED = 42

# ---------------------------------------------------------------------------
# Training hyperparameters (can be overridden via CLI flags)
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0  # L2 regularization coefficient (0 = disabled)
DROPOUT_RATE = 0.3   # Used by BaselineCNN
EPOCHS_DEFAULT = 15
OPTIMIZER = "adam"   # "adam" or "sgd"

# ---------------------------------------------------------------------------
# Hyperparameter tuning grid (for grid search experiments)
# ---------------------------------------------------------------------------
HYPERPARAMETER_GRID = {
    "learning_rate": [1e-5, 1e-4, 1e-3],
    "batch_size": [16, 32, 64],
    "dropout": [0.2, 0.3, 0.5],
}

# ---------------------------------------------------------------------------
# Geographic stratification (see src/data/splits.py)
#
# NOTE: grouping by whole administrative district was tried first and
# rejected — testing it against data shaped like the real INVASORAS.PT
# distribution (Lisboa district alone ~60% of records) showed a single
# dominant district forces a k-fold to put >=60% of the data in one fold,
# since a group can't be split across folds. Grouping by spatial grid cell
# (with district / macro-region as fallbacks for sparse cells) fixes this:
# dense metro areas span many cells and so CAN be split across folds.
# ---------------------------------------------------------------------------
N_FOLDS = 5
MCNEMAR_FOLD_INDEX = 0   # which fold doubles as the fixed comparison set
GRID_CELL_SIZE_DEG = 0.2   # ~20km at Portugal's latitude
MIN_GRID_SIZE = 15        # below this, fall back to the district label
MIN_DISTRICT_SIZE = 30    # below this, fall back to the macro-region label

MACRO_REGION_MAP = {
    # Portuguese district -> macro-region, used only as the last fallback
    # for districts too sparse to support their own grid cell or district
    # group. Extend as needed once real per-district counts are known.
    "Lisboa": "Centro-Litoral",
    "Setúbal": "Centro-Litoral",
    "Aveiro": "Centro-Litoral",
    "Coimbra": "Centro-Litoral",
    "Porto": "Norte-Litoral",
    "Braga": "Norte-Litoral",
    "Bragança": "Norte-Litoral",
    "Faro": "Sul",
    "Beja": "Sul",
    "Évora": "Sul",
    "Madeira": "Ilhas",
    "Açores": "Ilhas",
}
