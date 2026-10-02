"""
preprocess.py - fundus-aware image pre-processing and image-quality metrics.

Pipeline (each step is a separate, testable function):

    load -> crop field-of-view -> pad to square -> resize
         -> [variant-specific enhancement] -> circular mask

Variants (compared in the ablation study, see config.PREPROCESS_VARIANTS):
    P0_raw            baseline, geometry only
    P1_clahe          + CLAHE on the lightness channel (contrast adjustment)
    P2_clahe_unsharp  + median denoise, CLAHE and unsharp masking (edge enhancement)
    P3_ben_graham     + Gaussian local-average subtraction (Graham, 2015)

Why these steps?
  * Fundus photos have a large black border whose size differs per camera.
    Cropping it makes the retina fill the frame, so resizing does not waste
    pixels on background.
  * Padding to a square before resizing keeps the optic disc and vessels
    round; stretching would distort lesion shapes.
  * CLAHE (Zuiderveld, 1994) raises local contrast without blowing out the
    bright optic disc, which global histogram equalisation does. It is
    applied to L of LAB so colour (red haemorrhages, yellow exudates),
    which is diagnostic, is not altered.
  * Unsharp masking sharpens small, high-frequency structures such as
    microaneurysms and vessel edges.
  * The median filter removes salt-and-pepper sensor noise while keeping
    edges, unlike a mean filter.

Run as a script to cache every image once (deterministic, so training is fast
and reproducible):
    python src/preprocess.py --variant P2_clahe_unsharp --size 300
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config


# ==========================================================================
# 1. Geometry
# ==========================================================================
def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image from disk as an RGB uint8 array."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fov_mask(img: np.ndarray, tol: int = 10) -> np.ndarray:
    """Binary mask of the circular field of view (pixels brighter than `tol`)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > tol).astype(np.uint8)
    # Morphological opening removes isolated bright specks in the border.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def crop_fov(img: np.ndarray, tol: int = 10) -> np.ndarray:
    """Crop the black border so the retina touches the image edges."""
    mask = fov_mask(img, tol)
    ys, xs = np.where(mask > 0)
    if len(xs) < 0.05 * mask.size:      # almost-black image: do not crop
        return img
    return img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad with black so width == height (keeps the retina circular)."""
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def circular_mask(img: np.ndarray, scale: float = 0.96) -> np.ndarray:
    """Black out everything outside a centred circle (removes enhancement halo)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * scale), 1, -1)
    return img * mask[..., None]


# ==========================================================================
# 2. Enhancement
# ==========================================================================
def clahe_lab(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on lightness only, so diagnostic colour is preserved."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def unsharp_mask(img: np.ndarray, sigma: float = 2.0, amount: float = 1.0) -> np.ndarray:
    """Edge enhancement: img + amount * (img - blurred)."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def ben_graham(img: np.ndarray, sigma_ratio: float = 30.0) -> np.ndarray:
    """Subtract the local average colour (Graham, 2015).

    4*I - 4*Gaussian(I) + 128 removes slow illumination changes between
    cameras and leaves lesions and vessels on a neutral grey background.
    """
    sigma = img.shape[0] / sigma_ratio
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)


# ==========================================================================
# 3. Full pipeline
# ==========================================================================
def preprocess(img: np.ndarray, variant: str = config.DEFAULT_VARIANT,
               size: int = 300) -> np.ndarray:
    """Apply one named pre-processing variant and return a size x size RGB image."""
    if variant not in config.PREPROCESS_VARIANTS:
        raise ValueError(f"Unknown variant {variant}")
    out = pad_to_square(crop_fov(img))
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)

    if variant == "P1_clahe":
        out = clahe_lab(out)
    elif variant == "P2_clahe_unsharp":
        out = cv2.medianBlur(out, 3)
        out = clahe_lab(out)
        out = unsharp_mask(out)
    elif variant == "P3_ben_graham":
        out = ben_graham(out)
    return circular_mask(out)


def pipeline_stages(img: np.ndarray, size: int = 300) -> dict[str, np.ndarray]:
    """Every intermediate stage of P2, used to draw the step-by-step figure."""
    cropped = crop_fov(img)
    squared = pad_to_square(cropped)
    resized = cv2.resize(squared, (size, size), interpolation=cv2.INTER_AREA)
    denoised = cv2.medianBlur(resized, 3)
    clahe = clahe_lab(denoised)
    sharp = unsharp_mask(clahe)
    return {
        "1. Original": img,
        "2. FOV crop": cropped,
        "3. Pad + resize": resized,
        "4. Median denoise": denoised,
        "5. CLAHE (L channel)": clahe,
        "6. Unsharp mask": sharp,
        "7. Circular mask": circular_mask(sharp),
    }


# ==========================================================================
# 4. Image quality metrics (used by the quality gate)
# ==========================================================================
def quality_metrics(img: np.ndarray) -> dict[str, float]:
    """Cheap, interpretable quality measures computed inside the field of view.

    sharpness   variance of the Laplacian of the green channel (low = blurred)
    brightness  mean V (HSV) inside the FOV        (low = under-exposed)
    contrast    std of the green channel in FOV    (low = washed out)
    clipped     share of FOV pixels at the top of the scale (high = burnt-out
                highlights, i.e. real over-exposure: detail is lost, which a
                high mean brightness alone does not imply - phase 2b)
    noise       median absolute difference between the green channel and its
                3x3 median (high = sensor noise or heavy JPEG artefacts). A
                median filter removes speckle but keeps edges, so what is left
                is mostly noise rather than anatomy.
    fov_ratio   share of the retina's bounding box that is retina: about 0.785
                (pi/4) for a complete circle, rising towards 1.0 when the camera
                cut the circle off; unusually low = irregular or partial retina

    The green channel is used because vessels and lesions have the highest
    contrast there. The cropped retina is resized straight to 512x512 (no
    padding) - exactly as in the phase 1 audit, so thresholds fitted on the
    phase 1 measurements apply unchanged to new photos in the app.
    """
    small = cv2.resize(crop_fov(img), (512, 512), interpolation=cv2.INTER_AREA)
    mask = fov_mask(small).astype(bool)
    if mask.sum() < 100:
        return {"sharpness": 0.0, "brightness": 0.0, "contrast": 0.0,
                "clipped": 0.0, "noise": 0.0, "fov_ratio": 0.0}
    green = small[..., 1]
    lap = cv2.Laplacian(green, cv2.CV_64F)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    residual = cv2.absdiff(green, cv2.medianBlur(green, 3))
    return {
        "sharpness": float(lap[mask].var()),
        "brightness": float(hsv[..., 2][mask].mean()),
        "contrast": float(green[mask].std()),
        "clipped": float((small[mask].max(axis=-1) >= 250).mean()),
        "noise": float(np.median(residual[mask])),
        "fov_ratio": float(mask.mean()),
    }


def fit_quality_thresholds(metrics: pd.DataFrame, pct: float = 1.0) -> dict:
    """Data-driven thresholds: a photo is refused if it is worse than the
    `pct`-th percentile of the TRAINING set on a "lower is worse" measure, or
    worse than the (100-pct)-th percentile on a "higher is worse" measure.
    Percentiles avoid hand-picked magic numbers and adapt to the dataset.

    Chosen after inspecting what each check actually refused (D20, D53):
      * fov_ratio is not used: its lowest values belonged to well-framed photos
        from one camera whose circle has a small notch.
      * mean brightness has no upper limit: the photos it refused were bright
        but perfectly gradable. Real over-exposure is measured by `clipped`.
      * clipped and noise were added after the phase 6 stress test showed the
        gate missed over-exposure, sensor noise and JPEG artefacts.
    """
    thr = {
        "sharpness_min": float(np.percentile(metrics["sharpness"], pct)),
        "brightness_min": float(np.percentile(metrics["brightness"], pct)),
        "contrast_min": float(np.percentile(metrics["contrast"], pct)),
    }
    for higher_is_worse in ("clipped", "noise"):          # added in phase 7
        if higher_is_worse in metrics:
            thr[f"{higher_is_worse}_max"] = float(np.percentile(metrics[higher_is_worse], 100 - pct))
    return thr


def assess_quality(img: np.ndarray, thresholds: dict) -> dict:
    """Return metrics, a gradable flag and human-readable reasons."""
    m = quality_metrics(img)
    if m["fov_ratio"] == 0.0:            # quality_metrics found no retina at all
        return {"metrics": m, "gradable": False,
                "reasons": ["no retina detected - is this a fundus photo?"]}
    checks = [
        ("sharpness", "sharpness_min", "lower", "image is blurred / out of focus"),
        ("brightness", "brightness_min", "lower", "image is under-exposed (too dark)"),
        ("contrast", "contrast_min", "lower", "contrast too low to see lesions"),
        ("clipped", "clipped_max", "higher", "over-exposed: bright areas are burnt out"),
        ("noise", "noise_max", "higher", "too noisy or heavily compressed"),
    ]
    reasons = [text for key, thr_key, direction, text in checks
               if thr_key in thresholds and (m[key] < thresholds[thr_key] if direction == "lower"
                                             else m[key] > thresholds[thr_key])]
    return {"metrics": m, "gradable": not reasons, "reasons": reasons}


# ==========================================================================
# 5. Loading the split (with the compact working copy on Colab)
# ==========================================================================
def load_splits(path: Path = config.SPLITS_CSV) -> pd.DataFrame:
    """Read splits.csv (id, path, grade, split) and point `path` at the 1024 px
    working copy when it exists (Colab), otherwise keep the original paths."""
    df = pd.read_csv(path)[["id", "path", "grade", "split"]]
    wc = config.WORKING_COPY_DIR
    if wc.exists() and any(wc.glob("*.jpg")):
        df["path"] = df["id"].map(lambda i: str(wc / f"{i}.jpg"))
    return df


# ==========================================================================
# 6. Batch caching (run once per variant/size)
# ==========================================================================
def cache_dataframe(df: pd.DataFrame, variant: str, size: int,
                    workers: int = 4) -> pd.DataFrame:
    """Pre-process every image in df['path'] and save PNGs to the cache.

    Returns a copy of df with a new 'cached_path' column. Existing files are
    skipped, so this is safe to re-run.
    """
    out_dir = config.CACHE_DIR / f"{variant}_{size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _one(src: str) -> str:
        dst = out_dir / (Path(src).stem + ".png")
        if not dst.exists():
            img = preprocess(load_rgb(src), variant, size)
            cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return str(dst)

    with ThreadPoolExecutor(workers) as pool:
        cached = list(pool.map(_one, df["path"].tolist()))
    out = df.copy()
    out["cached_path"] = cached
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache pre-processed images")
    ap.add_argument("--variant", default=config.DEFAULT_VARIANT,
                    choices=list(config.PREPROCESS_VARIANTS))
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()

    splits = load_splits()
    cache_dataframe(splits, args.variant, args.size)
    print(f"Cached {len(splits)} images -> {config.CACHE_DIR / f'{args.variant}_{args.size}'}")

    # Quality thresholds are fitted on TRAIN images only (no test leakage).
    if not config.QUALITY_JSON.exists():
        train = splits[splits.split == "train"]
        table = config.OUTPUT_DIR / "image_quality.csv"      # written by dataset_audit.py
        if table.exists():
            metrics = pd.read_csv(table).merge(train[["path"]], on="path")
        else:
            metrics = pd.DataFrame([quality_metrics(load_rgb(p)) for p in train["path"]])
        thr = fit_quality_thresholds(metrics)
        config.QUALITY_JSON.write_text(json.dumps(thr, indent=2))
        print("Quality thresholds:", thr)


if __name__ == "__main__":
    main()
