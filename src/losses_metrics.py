"""
losses_metrics.py - custom loss, callback and metric helpers.

  * class_balanced_focal_loss - focal loss (Lin et al., 2017) with
    effective-number class weights (Cui et al., 2019) and label smoothing.
    Focal loss down-weights easy, confidently-correct images (mostly
    "No DR") so training time is spent on hard, borderline grades.
  * QWKCallback - computes Quadratic Weighted Kappa on the validation set
    after every epoch. QWK (Cohen, 1968) was the official metric of both
    Kaggle DR competitions because grades are ORDINAL: predicting 4 for a
    true 0 is far worse than predicting 1, and accuracy cannot see that.
    Early stopping, LR scheduling and checkpointing all monitor val_qwk.
  * ordinal decoding - turns the 5 probabilities into an expected grade
    and cuts it with thresholds tuned on the validation set.
  * calibration - expected calibration error and temperature scaling
    (Guo et al., 2017) so the confidence shown in the app is honest.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score, f1_score
from tensorflow import keras

import config

ops = keras.ops


# ==========================================================================
# Loss
# ==========================================================================
def class_balanced_focal_loss(class_weights, gamma: float = config.FOCAL_GAMMA,
                              label_smoothing: float = config.LABEL_SMOOTHING):
    """Weighted focal cross-entropy for one-hot targets."""
    w = ops.convert_to_tensor(np.asarray(class_weights, dtype="float32"))

    def loss(y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        k = ops.cast(ops.shape(y_true)[-1], "float32")
        y_smooth = y_true * (1.0 - label_smoothing) + label_smoothing / k
        p = ops.clip(ops.cast(y_pred, "float32"), 1e-7, 1.0 - 1e-7)
        focal = ops.power(1.0 - p, gamma)
        per_class = -y_smooth * focal * ops.log(p) * w
        return ops.sum(per_class, axis=-1)

    loss.__name__ = "cb_focal"
    return loss


# ==========================================================================
# Helpers for dict / list model outputs
# ==========================================================================
def stage_probs(pred) -> np.ndarray:
    """Extract the 5-class probabilities from model.predict output."""
    return np.asarray(pred["stage"] if isinstance(pred, dict) else pred[0])


def referable_probs(pred) -> np.ndarray:
    return np.asarray(pred["referable"] if isinstance(pred, dict) else pred[1]).ravel()


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def bootstrap_ci(y_true, y_pred, metric=qwk, n: int = 1000, seed: int = config.SEED):
    """95% confidence interval of a metric by resampling the evaluated images
    with replacement (Efron & Tibshirani, 1993). No retraining is needed, so it
    is a cheap way to see whether two runs really differ or only by chance."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = [metric(y_true[i], y_pred[i])
              for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ==========================================================================
# Callback
# ==========================================================================
class QWKCallback(keras.callbacks.Callback):
    """Adds val_qwk and val_macro_f1 to the epoch logs.

    Must be placed FIRST in the callbacks list so EarlyStopping,
    ReduceLROnPlateau and ModelCheckpoint can read val_qwk.
    """

    def __init__(self, val_images, val_labels: np.ndarray):
        super().__init__()
        self.val_images = val_images
        self.val_labels = np.asarray(val_labels)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probs = stage_probs(self.model.predict(self.val_images, verbose=0))
        preds = probs.argmax(1)
        logs["val_qwk"] = qwk(self.val_labels, preds)
        logs["val_macro_f1"] = float(f1_score(self.val_labels, preds, average="macro"))
        print(f" - val_qwk: {logs['val_qwk']:.4f} - val_macro_f1: {logs['val_macro_f1']:.4f}")


# ==========================================================================
# Ordinal decoding
# ==========================================================================
def expected_grade(probs: np.ndarray) -> np.ndarray:
    """Probability-weighted mean grade, a continuous 0-4 score."""
    return probs @ np.arange(probs.shape[1])


def apply_thresholds(scores: np.ndarray, thr) -> np.ndarray:
    return np.digitize(scores, np.sort(thr))


def optimise_thresholds(scores: np.ndarray, y: np.ndarray, iters: int = 3) -> list[float]:
    """Coordinate search for the 4 cut-points that maximise validation QWK."""
    thr = [0.5, 1.5, 2.5, 3.5]
    for _ in range(iters):
        for i in range(4):
            lo = thr[i - 1] + 0.05 if i else 0.0
            hi = thr[i + 1] - 0.05 if i < 3 else 4.0
            grid = np.linspace(lo, hi, 40)
            scores_i = [qwk(y, apply_thresholds(scores, thr[:i] + [t] + thr[i + 1:])) for t in grid]
            thr[i] = float(grid[int(np.argmax(scores_i))])
    return thr


# ==========================================================================
# Calibration
# ==========================================================================
def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10):
    """ECE plus the per-bin data needed for a reliability diagram."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, avg_conf = correct[m].mean(), conf[m].mean()
            ece += m.mean() * abs(acc - avg_conf)
            rows.append((lo, hi, acc, avg_conf, int(m.sum())))
    return float(ece), rows


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-7, 1)) / T
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T minimising validation negative log-likelihood."""
    def nll(T):
        p = temperature_scale(probs, T)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
    return float(minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded").x)
