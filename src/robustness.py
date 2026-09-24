"""
robustness.py - a SYNTHETIC stress-test set ("DR-C") and a shortcut probe.

Why synthetic data here:
    Public DR datasets come mostly from clinic-grade cameras. Real screening in
    rural clinics or with smartphone adapters produces blurred, dark, noisy or
    heavily compressed photos. Labelled examples of those are not available, but
    they can be simulated from the labelled test set, following the corruption-
    benchmark method of Hendrycks and Dietterich (2019). The grade stays valid
    because only the acquisition is degraded, not the anatomy.
    Synthetic images are used ONLY for evaluation, never for training.

Two analyses:
  run_robustness  QWK / accuracy under 7 corruptions x 3 severities, and the
                  share of each that the quality gate would refuse.
  blur_probe      the "sharp means healthy" shortcut (phases 1-3), tested on
                  the trained model: blur healthy test photos and see whether
                  the model starts calling them diseased.

Usage:
    python src/robustness.py --run bench_densenet121
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import matplotlib
if "ipykernel" not in sys.modules:   # scripts save figures only; notebooks keep inline display
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from losses_metrics import expected_grade, qwk, stage_probs, temperature_scale
from preprocess import assess_quality, load_rgb, load_splits, preprocess

_rng = np.random.default_rng(config.SEED)


# ---- corruptions applied to the RAW photo (before pre-processing) ---------
def defocus_blur(img, s):
    return cv2.GaussianBlur(img, (0, 0), [2, 5, 9][s - 1] * img.shape[1] / 1000)


def motion_blur(img, s):
    k = int([9, 19, 31][s - 1] * img.shape[1] / 1000) | 1
    kernel = np.zeros((k, k), np.float32); kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def low_light(img, s):
    return (255 * (img / 255.0) ** [1.6, 2.2, 3.0][s - 1]).astype(np.uint8)


def over_exposure(img, s):
    return cv2.convertScaleAbs(img, alpha=[1.4, 1.8, 2.4][s - 1], beta=[10, 25, 40][s - 1])


def sensor_noise(img, s):
    return np.clip(img + _rng.normal(0, [8, 16, 28][s - 1], img.shape), 0, 255).astype(np.uint8)


def jpeg_compression(img, s):
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, [30, 15, 6][s - 1]])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def uneven_illumination(img, s):
    w = img.shape[1]
    grad = 1 - [0.35, 0.55, 0.75][s - 1] * (np.arange(w) / w)
    return np.clip(img * grad[None, :, None], 0, 255).astype(np.uint8)


CORRUPTIONS = {
    "defocus_blur": defocus_blur, "motion_blur": motion_blur, "low_light": low_light,
    "over_exposure": over_exposure, "sensor_noise": sensor_noise,
    "jpeg_compression": jpeg_compression, "uneven_illumination": uneven_illumination,
}


def _predict(model, meta, raws, batch: int = 32):
    """Pre-process raw photos exactly as in training, predict, calibrate."""
    x = np.stack([preprocess(im, meta["variant"], meta["size"]) for im in raws]).astype("float32")
    probs = np.concatenate([stage_probs(model(x[i:i + batch], training=False)) for i in range(0, len(x), batch)])
    return temperature_scale(probs, meta.get("temperature", 1.0))


def sample_test(n: int, grades=None) -> pd.DataFrame:
    """A reproducible, grade-stratified sample of test photos (1024 px working copy)."""
    test = load_splits().query("split == 'test'")
    if grades is not None:
        test = test[test.grade.isin(grades)]
    frac = min(1.0, n / len(test))
    return (test.groupby("grade", group_keys=False)
                .sample(frac=frac, random_state=config.SEED) if frac < 1 else test).reset_index(drop=True)


def run_robustness(model, meta, thresholds: dict, n: int = 200, progress=print) -> pd.DataFrame:
    """QWK, accuracy and quality-gate refusals for every corruption and severity."""
    test = sample_test(n)
    raws, y = [load_rgb(p) for p in test.path], test.grade.values
    rows = []
    for name, fn in [("clean", None)] + list(CORRUPTIONS.items()):
        for s in ([0] if fn is None else [1, 2, 3]):
            imgs = raws if fn is None else [fn(im, s) for im in raws]
            pred = _predict(model, meta, imgs).argmax(1)
            refused = np.mean([not assess_quality(im, thresholds)["gradable"] for im in imgs])
            rows.append({"corruption": name, "severity": s, "qwk": qwk(y, pred),
                         "accuracy": float((pred == y).mean()), "gate_refuses": float(refused)})
            progress(f"  {name:20s} severity {s}: QWK {rows[-1]['qwk']:.3f}, gate refuses {refused:.0%}")
    return pd.DataFrame(rows).round(4)


def gate_coverage(thresholds: dict, n: int = 200, progress=print) -> pd.DataFrame:
    """How often the quality gate refuses each corruption - no model needed.

    Used in phase 7 to check that the added checks close the gaps the phase 6
    stress test found (over-exposure, sensor noise, JPEG artefacts).
    """
    raws = [load_rgb(p) for p in sample_test(n).path]
    rows = []
    for name, fn in [("clean", None)] + list(CORRUPTIONS.items()):
        for s in ([0] if fn is None else [1, 2, 3]):
            imgs = raws if fn is None else [fn(im, s) for im in raws]
            checks = [assess_quality(im, thresholds) for im in imgs]
            reasons = [r for c in checks for r in c["reasons"]]
            rows.append({"corruption": name, "severity": s,
                         "gate_refuses": float(np.mean([not c["gradable"] for c in checks])),
                         "main_reason": pd.Series(reasons).mode().iloc[0] if reasons else "-"})
            progress(f"  {name:20s} severity {s}: refuses {rows[-1]['gate_refuses']:.0%}")
    return pd.DataFrame(rows).round(4)


def blur_probe(model, meta, n: int = 100) -> pd.DataFrame:
    """Blur healthy (grade 0) test photos: does the model start seeing disease?"""
    test = sample_test(n, grades=[0])
    raws = [load_rgb(p) for p in test.path]
    rows = []
    for s in [0, 1, 2, 3]:
        imgs = raws if s == 0 else [defocus_blur(im, s) for im in raws]
        p = _predict(model, meta, imgs)
        rows.append({"blur_severity": s, "images": len(imgs),
                     "mean_expected_grade": float(expected_grade(p).mean()),
                     "called_any_DR_%": float((p.argmax(1) >= 1).mean() * 100),
                     "called_referable_%": float((p.argmax(1) >= 2).mean() * 100)})
    return pd.DataFrame(rows).round(3)


def plot_robustness(df: pd.DataFrame):
    clean = df.loc[df.corruption == "clean", "qwk"].item()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in CORRUPTIONS:
        part = df[df.corruption == name]
        axes[0].plot([0, 1, 2, 3], [clean] + part.qwk.tolist(), "o-", label=name)
        axes[1].plot([1, 2, 3], part.gate_refuses * 100, "o-", label=name)
    axes[0].set(xlabel="severity (0 = clean)", ylabel="QWK", title="Model accuracy on synthetic degraded photos")
    axes[1].set(xlabel="severity", ylabel="% refused by the quality gate", title="Does the gate catch them?")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "eval_robustness.png", dpi=150); plt.close(fig)
    return config.FIG_DIR / "eval_robustness.png"


def plot_examples():
    img = load_rgb(sample_test(5).path.iloc[0])
    fig, axes = plt.subplots(1, len(CORRUPTIONS) + 1, figsize=(2.3 * (len(CORRUPTIONS) + 1), 2.7))
    axes[0].imshow(img); axes[0].set_title("clean", fontsize=8)
    for ax, (name, fn) in zip(axes[1:], CORRUPTIONS.items()):
        ax.imshow(fn(img, 3)); ax.set_title(name.replace("_", "\n") + " (sev. 3)", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "eval_robustness_examples.png", dpi=150); plt.close(fig)
    return config.FIG_DIR / "eval_robustness_examples.png"


def plot_blur_probe(probe: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(probe.blur_severity, probe["called_any_DR_%"], color="#c0504d", alpha=.8, label="% called diseased")
    ax1.set(xlabel="blur severity applied to healthy photos (0 = original)", ylabel="% of healthy photos called diseased",
            xticks=probe.blur_severity, title="Shortcut probe: does blur alone make the model see disease?")
    ax2 = ax1.twinx(); ax2.plot(probe.blur_severity, probe.mean_expected_grade, "ko-", label="mean predicted grade")
    ax2.set(ylabel="mean predicted grade (0-4)", ylim=(0, 4))     # full scale: small shifts stay small
    ax1.set_ylim(0, 100)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.88), fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "eval_blur_probe.png", dpi=150); plt.close(fig)
    return config.FIG_DIR / "eval_blur_probe.png"


def main() -> None:
    from model import load_for_inference
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    run_dir = config.RUNS_DIR / args.run
    meta = json.loads((run_dir / "meta.json").read_text())
    model = load_for_inference(run_dir / "best.keras")
    out = config.OUTPUT_DIR / "evaluation"; out.mkdir(parents=True, exist_ok=True)
    df = run_robustness(model, meta, json.loads(config.QUALITY_JSON.read_text()), args.n)
    df.to_csv(out / "robustness.csv", index=False)
    blur_probe(model, meta).to_csv(out / "blur_probe.csv", index=False)
    plot_robustness(df); plot_examples()


if __name__ == "__main__":
    main()
