"""
dataset_audit.py - understand and clean the dataset BEFORE any modelling.

What it does (each step produces evidence for the report):
  1. Builds an index of (image path, grade) from either a labels CSV
     (APTOS style: id_code, diagnosis) or a folder-per-class layout.
  2. Scans every image ONCE and records its resolution, four image-quality
     measures and a 64-bit perceptual hash.        -> outputs/image_quality.csv
  3. Draws the exploratory figures.                -> figures/eda_*.png
  4. Finds exact and near-duplicate photos from the hashes. Duplicates that
     land in both train and test inflate scores (data leakage); duplicates
     with different grades are label noise.        -> figures/duplicates.png
  5. Keeps one image per duplicate group (drops groups whose grades
     disagree) and makes a stratified 70/15/15 split. -> outputs/splits.csv

Usage:
    python src/dataset_audit.py                     # APTOS CSV layout
    python src/dataset_audit.py --folders /path     # folder-per-class layout
"""
from __future__ import annotations

import argparse
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib
if "ipykernel" not in sys.modules:   # scripts save figures only; notebooks keep inline display
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import config
from preprocess import crop_fov, load_rgb, quality_metrics

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ==========================================================================
# 1. Index
# ==========================================================================
def index_from_csv(csv_path: Path, image_dir: Path) -> pd.DataFrame:
    """APTOS layout: train.csv with columns id_code, diagnosis."""
    df = pd.read_csv(csv_path)
    id_col = "id_code" if "id_code" in df else df.columns[0]
    label_col = "diagnosis" if "diagnosis" in df else df.columns[1]
    files = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    df["path"] = df[id_col].astype(str).map(lambda s: str(files.get(s, "")))
    missing = int((df["path"] == "").sum())
    if missing:
        print(f"WARNING: {missing} rows in the CSV have no matching image file")
    df = df[df["path"] != ""]
    return pd.DataFrame({"id": df[id_col].astype(str), "path": df["path"],
                         "grade": df[label_col].astype(int)})


def index_from_folders(root: Path) -> pd.DataFrame:
    """Folder-per-class layout, e.g. root/Mild/xxx.png (any nesting depth)."""
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        key = p.parent.name.strip().lower()
        if key in config.FOLDER_NAME_TO_GRADE:
            rows.append({"id": p.stem, "path": str(p), "grade": config.FOLDER_NAME_TO_GRADE[key]})
    if not rows:
        raise RuntimeError("No class folders recognised - extend FOLDER_NAME_TO_GRADE in config.py")
    return pd.DataFrame(rows)


# ==========================================================================
# 2. Perceptual hash (written out rather than imported from a library, so the
#    notebook needs no internet access and the method can be explained)
# ==========================================================================
def phash_bits(gray: np.ndarray) -> np.ndarray:
    """64-bit perceptual hash of a grayscale image, as a 0/1 float vector.

    Shrink to 32x32, take the discrete cosine transform, keep the top-left
    8x8 block (the lowest spatial frequencies, i.e. the coarse structure)
    and set each bit to "is this coefficient above the block median?".
    Two photos of the same eye keep almost the same coarse structure even
    after resizing or re-compression, so their hashes differ in only a few
    bits, while different eyes differ in many.
    """
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    return (low > np.median(low)).flatten().astype(np.float32)


def audit_one(path: str) -> dict:
    """Read an image once and return its size, quality measures and hash.

    The hash is computed on the FOV-cropped, contrast-equalised GREEN channel:
    cropping stops the black border (the same shape in every photo) from
    dominating the hash, and the green channel carries the vessel pattern,
    which is unique to each eye - it is used for retinal biometrics.
    """
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    green = cv2.createCLAHE(2.0, (8, 8)).apply(crop_fov(rgb)[..., 1])
    return {"path": path, "width": w, "height": h,
            **quality_metrics(rgb), "bits": phash_bits(green)}


def scan_images(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    """One pass over the dataset; OpenCV releases the GIL so threads help."""
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(audit_one, df["path"].tolist()))
    return df.merge(pd.DataFrame(rows), on="path")


# ==========================================================================
# 3. De-duplication
# ==========================================================================
def duplicate_groups(bits: np.ndarray, max_dist: int) -> np.ndarray:
    """Group images whose hashes differ by <= max_dist bits (union-find).

    Hamming distance for 0/1 vectors = 64 - (matching ones + matching zeros),
    which is two matrix products - fast even for tens of thousands of images.
    """
    n = len(bits)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, 2048):                  # chunked to bound memory
        block = bits[start:start + 2048]
        same = block @ bits.T + (1 - block) @ (1 - bits).T
        bi, bj = np.where(bits.shape[1] - same <= max_dist)
        bi = bi + start
        keep = bj > bi                               # upper triangle only
        for i, j in zip(bi[keep], bj[keep]):
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one image per duplicate group; drop groups with conflicting grades."""
    bits = np.stack(df["bits"].to_numpy())
    df = df.copy()
    df["dup_group"] = duplicate_groups(bits, config.PHASH_DUPLICATE_DISTANCE)
    sizes = df.groupby("dup_group")["id"].transform("size")
    n_grades = df.groupby("dup_group")["grade"].transform("nunique")

    conflicting = df[(sizes > 1) & (n_grades > 1)]
    consistent_dups = df[(sizes > 1) & (n_grades == 1)]
    kept = df[n_grades == 1].drop_duplicates("dup_group", keep="first")

    stats = {
        "images_in": int(len(df)),
        "duplicate_groups": int((df.groupby("dup_group").size() > 1).sum()),
        "images_in_duplicate_groups": int((sizes > 1).sum()),
        "images_in_conflicting_groups": int(len(conflicting)),
        "images_removed": int(len(df) - len(kept)),
        "images_out": int(len(kept)),
    }
    plot_duplicates(pd.concat([conflicting, consistent_dups]))
    if stats["images_removed"] > 0.15 * stats["images_in"]:
        print("WARNING: de-duplication would remove more than 15% of the data. Inspect "
              "figures/duplicates.png - if the pairs are NOT the same photo, lower "
              "PHASH_DUPLICATE_DISTANCE in config.py and re-run.")
    return kept.drop(columns="dup_group"), stats


def plot_duplicates(dups: pd.DataFrame, max_groups: int = 4, out: Path | None = None) -> None:
    """Show a few duplicate groups with their grades (evidence figure)."""
    out = out or config.FIG_DIR / "duplicates.png"
    groups = [g for _, g in dups.groupby("dup_group")][:max_groups]
    if not groups:
        print("No duplicate pairs found.")
        return
    fig, axes = plt.subplots(len(groups), 2, figsize=(6, 3 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        for c in range(2):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(g):
                row = g.iloc[c]
                ax.imshow(Image.open(row["path"]).convert("RGB").resize((256, 256)))
                ax.set_title(f"{row['id'][:12]}  grade={row['grade']}", fontsize=9)
    fig.suptitle("Near-duplicate pairs found by perceptual hashing")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ==========================================================================
# 4. Exploratory figures
# ==========================================================================
def plot_class_distribution(df: pd.DataFrame, out: Path | None = None) -> None:
    out = out or config.FIG_DIR / "eda_class_distribution.png"
    counts = df["grade"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([config.CLASS_NAMES[i] for i in counts.index], counts.values, color="#2b6f8e")
    ax.bar_label(bars, labels=[f"{v}\n({v / counts.sum():.1%})" for v in counts.values], fontsize=8)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_resolution_quality(meta: pd.DataFrame, out: Path | None = None) -> None:
    out = out or config.FIG_DIR / "eda_resolution_quality.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].scatter(meta["width"], meta["height"], s=6, c=meta["grade"], cmap="viridis")
    axes[0].set(xlabel="width (px)", ylabel="height (px)", title="Original resolutions")
    meta.boxplot(column="sharpness", by="grade", ax=axes[1])
    axes[1].set(yscale="log", title="Sharpness by grade", xlabel="grade")
    meta.boxplot(column="brightness", by="grade", ax=axes[2])
    axes[2].set(title="Brightness by grade", xlabel="grade")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_samples(df: pd.DataFrame, out: Path | None = None, per_class: int = 3) -> None:
    out = out or config.FIG_DIR / "eda_samples.png"
    fig, axes = plt.subplots(per_class, config.NUM_CLASSES, figsize=(13, 2.7 * per_class))
    for g in range(config.NUM_CLASSES):
        subset = df[df.grade == g]
        samples = subset.sample(min(per_class, len(subset)), random_state=config.SEED)
        for r in range(per_class):
            ax = axes[r, g]
            ax.axis("off")
            if r < len(samples):
                ax.imshow(Image.open(samples.iloc[r]["path"]).convert("RGB").resize((256, 256)))
        axes[0, g].set_title(config.CLASS_NAMES[g], fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ==========================================================================
# 5. Split
# ==========================================================================
def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """70 / 15 / 15 split that keeps the grade proportions in every part."""
    rest, test = train_test_split(df, test_size=config.TEST_FRACTION,
                                  stratify=df["grade"], random_state=config.SEED)
    val_share = config.VAL_FRACTION / (1 - config.TEST_FRACTION)
    train, val = train_test_split(rest, test_size=val_share,
                                  stratify=rest["grade"], random_state=config.SEED)
    return pd.concat([train.assign(split="train"), val.assign(split="val"),
                      test.assign(split="test")]).reset_index(drop=True)


def split_table(splits: pd.DataFrame) -> pd.DataFrame:
    """Grade x split counts, the table that goes into report section 2."""
    table = pd.crosstab(splits["grade"].map(lambda g: config.CLASS_NAMES[g]), splits["split"])
    table = table.reindex(config.CLASS_NAMES)[["train", "val", "test"]]
    table.loc["Total"] = table.sum()
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", type=Path, default=None,
                    help="root of a folder-per-class dataset (instead of the CSV)")
    args = ap.parse_args()
    config.set_global_seed()

    df = index_from_folders(args.folders) if args.folders else \
        index_from_csv(config.LABELS_CSV, config.RAW_IMAGE_DIR)
    print(f"Indexed {len(df)} images")

    meta = scan_images(df)
    meta.drop(columns="bits").to_csv(config.OUTPUT_DIR / "image_quality.csv", index=False)

    kept, dup_stats = deduplicate(meta)
    print("De-duplication:", json.dumps(dup_stats, indent=2))
    (config.OUTPUT_DIR / "dedup_stats.json").write_text(json.dumps(dup_stats, indent=2))

    plot_class_distribution(kept)
    plot_resolution_quality(kept)
    plot_samples(kept)

    splits = stratified_split(kept.drop(columns="bits"))
    splits.to_csv(config.SPLITS_CSV, index=False)
    table = split_table(splits)
    print(table)
    table.to_csv(config.OUTPUT_DIR / "split_table.csv")


if __name__ == "__main__":
    main()
