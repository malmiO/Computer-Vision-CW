"""
config.py - single source of truth for every setting in the project.

Why a central config?
    Reproducibility is marked in the rubric (Code Quality, 10 marks). Keeping
    seeds, paths, class names and hyper-parameters in one place means every
    script (audit, preprocessing, training, evaluation, app) uses identical
    settings, and the report can quote them from one file.

Edit the PATHS section to match where your dataset lives. The defaults assume
a Kaggle notebook with the APTOS 2019 competition data attached.
"""
from pathlib import Path
import os
import random

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use so runs are repeatable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:  # the audit/preprocess scripts do not need TF
        pass


# --------------------------------------------------------------------------
# Paths - detected automatically for Colab, Kaggle or a local machine, and
# overridable with environment variables (DR_RAW_DIR, DR_LABELS_CSV,
# DR_WORK_DIR, DR_CACHE_DIR).
#
#   Colab : raw data on the fast local disk (/content/aptos), outputs on
#           Google Drive so they survive a disconnect, image cache local
#           (Drive is slow for thousands of small files).
#   Kaggle: data mounted read-only under /kaggle/input, outputs in /kaggle/working.
#   Local : everything under ~/dr-sight-work.
# --------------------------------------------------------------------------
if Path("/kaggle/working").exists():
    ENVIRONMENT = "kaggle"
    _raw, _work, _cache = ("/kaggle/input/aptos2019-blindness-detection",
                           "/kaggle/working", "/kaggle/working/cache")
elif Path("/content").exists():
    ENVIRONMENT = "colab"
    _drive = Path("/content/drive/MyDrive")
    _raw = "/content/aptos"
    _work = str(_drive / "CV-CW") if _drive.exists() else "/content/CV-CW-work"
    _cache = "/content/cache"
else:
    ENVIRONMENT = "local"
    _home = Path.home() / "dr-sight-work"
    _raw, _work, _cache = str(_home / "aptos"), str(_home), str(_home / "cache")

RAW_IMAGE_DIR = Path(os.getenv("DR_RAW_DIR", f"{_raw}/train_images"))
LABELS_CSV = Path(os.getenv("DR_LABELS_CSV", f"{_raw}/train.csv"))
WORK_DIR = Path(os.getenv("DR_WORK_DIR", _work))

CACHE_DIR = Path(os.getenv("DR_CACHE_DIR", _cache))  # preprocessed images, one folder per variant/size
# The compact 1024 px working copy made in phase 1 (unpacked on Colab's local disk)
WORKING_COPY_DIR = Path(os.getenv("DR_WORKING_COPY", "/content/aptos_1024/images"
                                  if ENVIRONMENT == "colab" else f"{_work}/aptos_1024/images"))
OUTPUT_DIR = WORK_DIR / "outputs"     # everything that goes into the report
FIG_DIR = OUTPUT_DIR / "figures"
RUNS_DIR = OUTPUT_DIR / "runs"        # one sub-folder per training experiment
SPLITS_CSV = OUTPUT_DIR / "splits.csv"
QUALITY_JSON = OUTPUT_DIR / "quality_thresholds.json"

for _d in (CACHE_DIR, FIG_DIR, RUNS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:  # read-only location (e.g. the hosted web app) - not needed there
        pass

# --------------------------------------------------------------------------
# Labels - International Clinical Diabetic Retinopathy (ICDR) scale
# (Wilkinson et al., 2003)
# --------------------------------------------------------------------------
CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)

# Grade >= 2 is "referable DR" in screening programmes and in the
# Gulshan et al. (2016) and Abramoff et al. (2018) studies.
REFERABLE_FROM_GRADE = 2

# Folder names people commonly use when a Kaggle upload is organised as
# one folder per class instead of a CSV. Extend if your dataset differs.
FOLDER_NAME_TO_GRADE = {
    "0": 0, "no_dr": 0, "nodr": 0, "no dr": 0, "normal": 0, "healthy": 0,
    "1": 1, "mild": 1, "mild_dr": 1, "mild npdr": 1,
    "2": 2, "moderate": 2, "moderate_dr": 2, "moderate npdr": 2,
    "3": 3, "severe": 3, "severe_dr": 3, "severe npdr": 3,
    "4": 4, "proliferate_dr": 4, "proliferative": 4, "proliferative_dr": 4, "pdr": 4,
}

# --------------------------------------------------------------------------
# Data split (stratified, after duplicate removal)
# --------------------------------------------------------------------------
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
PHASH_DUPLICATE_DISTANCE = 4   # Hamming distance (of 64 bits) treated as "same photo"

# --------------------------------------------------------------------------
# Pre-processing variants compared in the ablation study
# --------------------------------------------------------------------------
PREPROCESS_VARIANTS = {
    "P0_raw":          "FOV crop + square pad + resize (baseline)",
    "P1_clahe":        "P0 + CLAHE on the L channel of LAB (contrast adjustment)",
    "P2_clahe_unsharp": "P0 + median denoise + CLAHE + unsharp mask (contrast + edge enhancement)",
    "P3_ben_graham":   "P0 + Gaussian local-average subtraction (Graham, 2015)",
}
DEFAULT_VARIANT = "P2_clahe_unsharp"

# --------------------------------------------------------------------------
# Backbones compared in the transfer-learning benchmark.
#   size  = native ImageNet resolution of the network
#   prep  = how the network expects pixels to be scaled
#           "none"  -> the Keras model rescales internally (EfficientNet, MobileNetV3)
#           "tf"    -> scale to [-1, 1]            (ResNetV2)
#           "torch" -> /255 then ImageNet mean/std (DenseNet)
# --------------------------------------------------------------------------
BACKBONES = {
    "efficientnet_b0":   {"size": 224, "prep": "none"},
    "efficientnet_b3":   {"size": 300, "prep": "none"},
    "resnet50v2":        {"size": 224, "prep": "tf"},
    "densenet121":       {"size": 224, "prep": "torch"},
    "mobilenetv3_large": {"size": 224, "prep": "none"},
}
DEFAULT_BACKBONE = "efficientnet_b3"

# --------------------------------------------------------------------------
# Training hyper-parameters (defaults; train.py exposes them as CLI flags)
# --------------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS_HEAD = 5          # phase 1: backbone frozen, only the new heads learn
EPOCHS_FINETUNE = 25     # phase 2: top of the backbone unfrozen
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
UNFREEZE_FRACTION = 0.3  # fraction of backbone layers (from the top) to unfreeze
DROPOUT = 0.4
LABEL_SMOOTHING = 0.05
FOCAL_GAMMA = 2.0
CB_BETA = 0.999          # "effective number of samples" beta (Cui et al., 2019)
REFERABLE_LOSS_WEIGHT = 0.5
EARLY_STOP_PATIENCE = 6
REDUCE_LR_PATIENCE = 2

# Target used to pick the referable-DR decision threshold on the validation
# set. 0.87 mirrors the sensitivity reported for the first FDA-authorised
# autonomous DR system (Abramoff et al., 2018).
TARGET_REFERABLE_SENSITIVITY = 0.87

# Test-time augmentation: the 8 symmetries of a square (4 rotations x flip).
# They are exact (no interpolation), so TTA cannot introduce artefacts.
TTA_VIEWS = 8
UNCERTAINTY_STD_THRESHOLD = 0.15  # above this TTA spread -> "human review" flag
