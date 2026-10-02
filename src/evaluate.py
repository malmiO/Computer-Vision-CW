"""
evaluate.py - test-set evaluation: everything criterion 6 asks for, plus
clinically meaningful extras. Written as small functions so the phase 6
notebook can run and explain one analysis at a time; `main()` runs them all.

Rule for the whole module: anything that is TUNED (temperature, decision
thresholds, the human-review threshold) is fitted on the VALIDATION split.
The test split is used once, only to measure.

Outputs: tables and JSON in outputs/evaluation/, figures in outputs/figures/
(all prefixed eval_), and the validated settings written into the run's
meta.json for the web app.

Usage:
    python src/evaluate.py --run bench_densenet121
"""
from __future__ import annotations

import argparse
import json
import sys

import matplotlib
if "ipykernel" not in sys.modules:   # scripts save figures only; notebooks keep inline display
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)

import config
from data import images_only, make_dataset
from losses_metrics import (apply_thresholds, bootstrap_ci, expected_calibration_error,
                            expected_grade, fit_temperature, optimise_thresholds, qwk,
                            referable_probs, stage_probs, temperature_scale)
from model import load_for_inference
from preprocess import cache_dataframe, load_splits

EVAL_DIR = config.OUTPUT_DIR / "evaluation"
FIG = config.FIG_DIR
NAMES = config.CLASS_NAMES
SHORT = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


# ==========================================================================
# 1. Loading
# ==========================================================================
def load_run(run: str):
    """The trained model (as float32, for CPU) and its meta.json."""
    run_dir = config.RUNS_DIR / run
    meta = json.loads((run_dir / "meta.json").read_text())
    return load_for_inference(run_dir / "best.keras"), meta


def eval_splits(meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation and test rows with their pre-processed image paths."""
    splits = cache_dataframe(load_splits(), meta["variant"], meta["size"])
    return (splits[splits.split == "val"].reset_index(drop=True),
            splits[splits.split == "test"].reset_index(drop=True))


# ==========================================================================
# 2. Prediction with test-time augmentation
# ==========================================================================
def dihedral_views(x: tf.Tensor) -> list[tf.Tensor]:
    """The 8 exact symmetries of a square image batch - the same group used for
    training augmentation, so TTA never shows the model an unfamiliar view."""
    views = []
    for k in range(4):
        r = tf.image.rot90(x, k)
        views += [r, tf.image.flip_left_right(r)]
    return views


def predict(model, df: pd.DataFrame, size: int, tta: bool, batch: int = 16):
    """Return (stage_probs, referable_probs, tta_spread). tta_spread is the mean
    standard deviation of the stage probabilities across the 8 views."""
    P, R, S = [], [], []
    for x in images_only(make_dataset(df, size, batch)):
        outs = [model(v, training=False) for v in (dihedral_views(x) if tta else [x])]
        sp = np.stack([stage_probs(o) for o in outs])
        rp = np.stack([referable_probs(o) for o in outs])
        P.append(sp.mean(0)); R.append(rp.mean(0)); S.append(sp.std(0).mean(1))
    return np.concatenate(P), np.concatenate(R), np.concatenate(S)


# ==========================================================================
# 3. Settings fitted on VALIDATION
# ==========================================================================
def threshold_for_sensitivity(y_bin, scores, target: float) -> float:
    """Highest threshold whose sensitivity reaches the target."""
    fpr, tpr, thr = roc_curve(y_bin, scores)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.5


def fit_on_validation(pv, rv, y_val, review_fraction: float = 0.10) -> dict:
    """Temperature, ordinal cut-points, referable threshold and the confidence
    below which a case is sent for human review - all from validation data."""
    T = fit_temperature(pv, y_val)
    pv_cal = temperature_scale(pv, T)
    return {
        "temperature": T,
        "ordinal_cutpoints": optimise_thresholds(expected_grade(pv_cal), y_val),
        "referable_threshold": threshold_for_sensitivity(
            (y_val >= config.REFERABLE_FROM_GRADE).astype(int), rv, config.TARGET_REFERABLE_SENSITIVITY),
        "review_confidence": float(np.percentile(pv_cal.max(1), 100 * review_fraction)),
        "review_fraction": review_fraction,
    }


# ==========================================================================
# 4. Metrics
# ==========================================================================
def sens_spec(y_bin, pred_bin) -> tuple[float, float]:
    tp = np.sum((y_bin == 1) & (pred_bin == 1)); fn = np.sum((y_bin == 1) & (pred_bin == 0))
    tn = np.sum((y_bin == 0) & (pred_bin == 0)); fp = np.sum((y_bin == 0) & (pred_bin == 1))
    return float(tp / max(tp + fn, 1)), float(tn / max(tn + fp, 1))


def core_metrics(y, probs_cal, ref_scores, settings: dict) -> dict:
    """Headline test metrics, with bootstrap 95% intervals for the main ones."""
    pred = probs_cal.argmax(1)
    y_ref = (y >= config.REFERABLE_FROM_GRADE).astype(int)
    sens, spec = sens_spec(y_ref, (ref_scores >= settings["referable_threshold"]).astype(int))
    macro = lambda a, b: f1_score(a, b, average="macro")
    m = {
        "accuracy": accuracy_score(y, pred), "accuracy_ci": bootstrap_ci(y, pred, accuracy_score),
        "qwk": qwk(y, pred), "qwk_ci": bootstrap_ci(y, pred),
        "macro_f1": macro(y, pred), "macro_f1_ci": bootstrap_ci(y, pred, macro),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "ordinal_decoding_qwk": qwk(y, apply_thresholds(expected_grade(probs_cal), settings["ordinal_cutpoints"])),
        "referable_auc": roc_auc_score(y_ref, ref_scores),
        "referable_sensitivity": sens, "referable_specificity": spec,
        "any_dr_auc": roc_auc_score((y >= 1).astype(int), 1 - probs_cal[:, 0]),
    }
    return {k: (tuple(round(float(x), 4) for x in v) if isinstance(v, tuple) else round(float(v), 4))
            for k, v in m.items()}


def classification_table(y, pred) -> pd.DataFrame:
    rep = classification_report(y, pred, labels=range(config.NUM_CLASSES), target_names=NAMES,
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(3)


def group_table(y, pred, groups: pd.Series, min_n: int = 15) -> pd.DataFrame:
    """Performance per subgroup (camera, image quality, ...)."""
    rows = []
    for g, idx in pd.Series(range(len(y))).groupby(groups.values):
        idx = idx.values
        if len(idx) < min_n:
            continue
        yt, yp = y[idx], pred[idx]
        rows.append({"group": g, "images": len(idx), "no_dr_share": round(float((yt == 0).mean()), 3),
                     "grades_present": int(len(set(yt))), "accuracy": round(accuracy_score(yt, yp), 3),
                     "qwk": round(qwk(yt, yp), 3) if len(set(yt)) > 1 else np.nan,
                     "macro_f1": round(f1_score(yt, yp, average="macro"), 3)})
    cols = ["group", "images", "no_dr_share", "grades_present", "accuracy", "qwk", "macro_f1"]
    return (pd.DataFrame(rows, columns=cols).sort_values("images", ascending=False).reset_index(drop=True))


def camera_only_baseline(train_groups, train_y, test_groups, test_y) -> float:
    """QWK of a 'model' that ignores the eye and predicts the most common grade
    of each camera group - how far the camera shortcut alone would get."""
    majority = pd.Series(train_y).groupby(np.asarray(train_groups)).agg(lambda s: s.mode().iloc[0])
    overall = int(pd.Series(train_y).mode().iloc[0])
    pred = np.array([majority.get(g, overall) for g in test_groups])
    return round(qwk(test_y, pred), 4)


def selective_prediction(y, probs_cal, spread, coverages=(1.0, .95, .9, .85, .8, .75, .7, .6, .5)):
    """Accuracy and QWK when the least certain cases are handed to a human.
    Two uncertainty scores are compared: low confidence, and TTA disagreement."""
    conf = probs_cal.max(1)
    pred = probs_cal.argmax(1)
    y_ref = (y >= config.REFERABLE_FROM_GRADE).astype(int)
    rows = []
    for name, score in [("confidence", -conf), ("tta_spread", spread)]:
        order = np.argsort(score)                     # most certain first
        for c in coverages:
            keep = order[: int(round(c * len(y)))]
            rows.append({"uncertainty": name, "coverage": c, "accuracy": accuracy_score(y[keep], pred[keep]),
                         "qwk": qwk(y[keep], pred[keep]),
                         "referable_sensitivity": sens_spec(y_ref[keep], (pred[keep] >= 2).astype(int))[0]})
    return pd.DataFrame(rows).round(4)


# ==========================================================================
# 5. Figures (each saved to outputs/figures and returned as a path)
# ==========================================================================
def _save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150); plt.close(fig)
    return FIG / name


def plot_training_curves(run: str, epochs_head: int):
    """Accuracy and loss curves of the evaluated model's training run."""
    h = json.loads((config.RUNS_DIR / run / "history.json").read_text())
    ep = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tr, va, title) in zip(axes, [("stage_acc", "val_stage_acc", "Accuracy (stage)"),
                                          ("loss", "val_loss", "Loss (total)"), (None, "val_qwk", "Validation QWK")]):
        if tr: ax.plot(ep, h[tr], "o-", ms=3, label="train")
        ax.plot(ep, h[va], "o-", ms=3, label="validation")
        ax.axvline(epochs_head + 0.5, ls="--", c="grey", lw=1)
        ax.text(epochs_head + 0.6, ax.get_ylim()[0], " fine-tuning", fontsize=8, color="grey", va="bottom")
        ax.set(title=title, xlabel="epoch"); ax.legend(); ax.grid(alpha=.3)
    return _save(fig, "eval_training_curves.png")


def plot_confusion(y, pred):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, norm, title in [(axes[0], None, "Counts"), (axes[1], "true", "Row-normalised (= recall per grade)")]:
        cm = confusion_matrix(y, pred, labels=range(config.NUM_CLASSES), normalize=norm)
        ConfusionMatrixDisplay(cm, display_labels=SHORT).plot(ax=ax, cmap="Blues", colorbar=False,
                                                              values_format=".2f" if norm else "d")
        ax.set(title=title, xlabel="predicted grade", ylabel="true grade")
    return _save(fig, "eval_confusion_matrix.png")


def plot_per_class(table: pd.DataFrame):
    t = table.loc[NAMES, ["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(t)); w = 0.26
    for i, col in enumerate(t.columns):
        ax.bar(x + (i - 1) * w, t[col], w, label=col)
    ax.set(xticks=x, xticklabels=SHORT, ylim=(0, 1), ylabel="score", title="Precision, recall and F1 per grade (test set)")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    return _save(fig, "eval_per_class.png")


def plot_roc(y, probs_cal, ref_scores, settings: dict):
    y_ref = (y >= config.REFERABLE_FROM_GRADE).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in range(config.NUM_CLASSES):
        if len(set(y == c)) > 1:
            fpr, tpr, _ = roc_curve(y == c, probs_cal[:, c])
            axes[0].plot(fpr, tpr, label=f"{SHORT[c]} (AUC {roc_auc_score(y == c, probs_cal[:, c]):.3f})")
    fpr, tpr, thr = roc_curve(y_ref, ref_scores)
    axes[1].plot(fpr, tpr, c="#b03a2e", label=f"Referable DR (AUC {roc_auc_score(y_ref, ref_scores):.3f})")
    op = np.argmin(np.abs(thr - settings["referable_threshold"]))
    axes[1].plot(fpr[op], tpr[op], "ko", label="operating point (set on validation)")
    for ax, t in zip(axes, ["Per grade (one-vs-rest)", "Referable DR (grade ≥ 2)"]):
        ax.plot([0, 1], [0, 1], "k--", lw=.8)
        ax.set(xlabel="false positive rate", ylabel="true positive rate (sensitivity)", title=t); ax.legend(fontsize=8)
    return _save(fig, "eval_roc_curves.png")


def plot_reliability(probs_raw, probs_cal, y):
    ece_b, rows_b = expected_calibration_error(probs_raw, y)
    ece_a, rows_a = expected_calibration_error(probs_cal, y)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for rows, lab in [(rows_b, f"before scaling (ECE {ece_b:.3f})"), (rows_a, f"after temperature scaling (ECE {ece_a:.3f})")]:
        ax.plot([r[3] for r in rows], [r[2] for r in rows], "o-", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect calibration")
    ax.set(xlabel="mean confidence", ylabel="accuracy", title="Reliability diagram (test set)"); ax.legend(fontsize=8)
    return _save(fig, "eval_reliability.png"), ece_b, ece_a


def plot_selective(sel: pd.DataFrame, review_coverage: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric in zip(axes, ["accuracy", "qwk"]):
        for name, g in sel.groupby("uncertainty"):
            ax.plot(g.coverage * 100, g[metric], "o-", label=f"defer by {name.replace('_', ' ')}")
        ax.axhline(sel[(sel.coverage == 1.0)][metric].iloc[0], ls=":", c="grey", label="no deferral")
        ax.axvline(review_coverage * 100, ls="--", c="#b03a2e", lw=1, label="app's review rule")
        label = {"accuracy": "Accuracy", "qwk": "QWK"}[metric]
        ax.set(xlabel="% of cases the model decides (rest go to a human)", ylabel=label, title=f"{label} of the cases kept")
        ax.invert_xaxis(); ax.legend(fontsize=8); ax.grid(alpha=.3)
    return _save(fig, "eval_selective_prediction.png")


def gradcam_panel(model, meta, df, idx, probs_cal, y, name: str, title: str):
    """Input and Grad-CAM for the given test rows."""
    from gradcam import GradCAM, overlay
    cam = GradCAM(model, meta["gradcam_layer"])
    idx = list(idx)
    fig, axes = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6.6), squeeze=False)
    for i, j in enumerate(idx):
        img = tf.io.decode_png(tf.io.read_file(df.iloc[j]["cached_path"]), channels=3).numpy()
        pred = int(probs_cal[j].argmax())
        axes[0, i].imshow(img); axes[1, i].imshow(overlay(img, cam.heatmap(img.astype("float32"), pred)))
        axes[0, i].set_title(f"true {SHORT[y[j]]} | pred {SHORT[pred]} ({probs_cal[j].max():.0%})", fontsize=8)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    fig.suptitle(title, fontsize=10)
    return _save(fig, name)


# ==========================================================================
# 6. Record the validated settings for the app
# ==========================================================================
def save_settings(run: str, meta: dict, settings: dict, metrics: dict):
    meta = {**meta, **settings, "test_metrics": metrics}
    (config.RUNS_DIR / run / "meta.json").write_text(json.dumps(meta, indent=2))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "test_metrics.json").write_text(json.dumps({"run": run, **settings, **metrics}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    model, meta = load_run(args.run)
    val_df, test_df = eval_splits(meta)
    y_val, y = val_df.grade.values, test_df.grade.values
    pv, rv, _ = predict(model, val_df, meta["size"], tta=True)
    settings = fit_on_validation(pv, rv, y_val)
    pt, rt, st = predict(model, test_df, meta["size"], tta=True)
    pcal = temperature_scale(pt, settings["temperature"])
    metrics = core_metrics(y, pcal, rt, settings)
    table = classification_table(y, pcal.argmax(1)); table.to_csv(EVAL_DIR / "classification_report.csv")
    plot_training_curves(args.run, meta.get("epochs_head", config.EPOCHS_HEAD))
    plot_confusion(y, pcal.argmax(1)); plot_per_class(table); plot_roc(y, pcal, rt, settings)
    _, metrics["ece_before"], metrics["ece_after"] = plot_reliability(pt, pcal, y)
    save_settings(args.run, meta, settings, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
