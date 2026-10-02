"""
augment.py - "clinically plausible" data augmentation and class balancing.

Design rule: an augmented image must still be a believable fundus photo
with the SAME grade. Every transform below is chosen against that rule.

  Used                               Why it is safe for DR
  ---------------------------------  -----------------------------------------
  8 exact symmetries (flips +        Left/right eyes are mirror images and camera
  90-degree rotations)               alignment varies, so orientation carries no
                                     diagnostic meaning. Exact symmetries move no
                                     pixel between grid positions, so - unlike
                                     arbitrary-angle rotation or zoom, which
                                     resample and blurred images by 56% in phase
                                     3 - they add no blur. They also move camera
                                     marks (the notch) around.
  Brightness / contrast +-10 %       Mimics exposure differences between clinics.
  Sharpness jitter (shortcut-aware)  Random mild blur or sharpen. Phase 1/2 showed
                                     No DR photos are sharper than diseased ones;
                                     varying sharpness within every grade stops
                                     sharpness from being a reliable hint.
  Photometric changes are applied to the retina only, so the background stays
  exactly black as in every real pre-processed image.

  Deliberately NOT used              Why
  ---------------------------------  -----------------------------------------
  Arbitrary rotation / zoom          Resampling blurs every training image (phase
                                     3 measurement), so training photos would be
                                     systematically blurrier than test photos.
                                     Zoom is also redundant: the FOV crop already
                                     makes every retina fill the frame.
  Hue / saturation shifts            Colour is diagnostic (red haemorrhages,
                                     yellow exudates); shifting hue can erase
                                     or invent lesions.
  Cutout / random erasing            Can delete the only lesion -> wrong label.
  Strong shear / elastic warps       Distort vessel geometry unrealistically.
  Mixup / CutMix                     Blend two ORDINAL grades into a label with
                                     no clinical meaning.

Balancing (compared experimentally in train.py --balance):
  "none"        plain loss
  "weights"     class-balanced weights from the effective number of samples
                (Cui et al., 2019) inside a focal loss (Lin et al., 2017)
  "oversample"  minority grades re-sampled up to a target share; because
                every copy is augmented differently, copies are not identical
  "both"        weights + oversampling
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


SHARPNESS_JITTER = (0.35, 0.65)   # 0.5 = unchanged; ~x0.65 to ~x1.4 sharpness


def _layers():
    """Keras is imported lazily so the audit/pre-processing scripts do not need it."""
    import tensorflow as tf
    from tensorflow import keras

    class RandomTranspose(keras.layers.Layer):
        """Swap height and width for a random half of the images. Together with
        RandomFlip this samples all 8 exact symmetries of a square (the dihedral
        group D4): no interpolation, so no blur."""

        def call(self, x, training=None):
            if not training:
                return x
            swap = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.5
            return tf.where(swap, tf.transpose(x, [0, 2, 1, 3]), x)

    class RetinaOnly(keras.layers.Layer):
        """Apply photometric layers to the retina only. Brightness shifts would
        otherwise lift the black background to grey, which never happens in a
        real pre-processed image."""

        def __init__(self, inner, **kwargs):
            super().__init__(**kwargs)
            self.inner = keras.Sequential(inner)

        def call(self, x, training=None):
            retina = tf.cast(tf.reduce_max(x, axis=-1, keepdims=True) > 5.0, x.dtype)
            return self.inner(x, training=training) * retina

    return keras, RandomTranspose, RetinaOnly


def build_augmenter(seed: int | None = config.SEED, shortcut_aware: bool = False):
    """Keras preprocessing layers; only active when called with training=True.

    shortcut_aware=True adds random sharpness jitter (Keras RandomSharpness).
    """
    keras, RandomTranspose, RetinaOnly = _layers()
    L = keras.layers
    photometric = [
        L.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
        L.RandomContrast(0.1, seed=seed),
    ]
    if shortcut_aware:
        if not hasattr(L, "RandomSharpness"):
            raise ImportError("keras.layers.RandomSharpness needs Keras >= 3.8: pip install -U keras")
        photometric.append(L.RandomSharpness(factor=SHARPNESS_JITTER, value_range=(0, 255), seed=seed))
    name = "shortcut_aware_augmentation" if shortcut_aware else "fundus_safe_augmentation"
    return keras.Sequential([
        L.RandomFlip("horizontal_and_vertical", seed=seed),
        RandomTranspose(name="random_transpose"),
        RetinaOnly(photometric, name="retina_only_photometric"),
    ], name=name)


def effective_number_weights(labels: np.ndarray, beta: float = config.CB_BETA,
                             num_classes: int = config.NUM_CLASSES) -> np.ndarray:
    """Class-balanced weights (Cui et al., 2019).

    w_c = (1 - beta) / (1 - beta ** n_c), normalised to mean 1.
    Softer than inverse frequency: very rare classes are up-weighted but not
    so much that a handful of images dominates the gradient.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return (weights / weights.mean()).astype(np.float32)


def class_weight_schemes(labels: np.ndarray, num_classes: int = config.NUM_CLASSES) -> dict:
    """The three weighting schemes compared in phase 3, each normalised to mean 1."""
    counts = np.maximum(np.bincount(labels, minlength=num_classes).astype(np.float64), 1)
    inverse = 1.0 / counts
    sqrt_inverse = 1.0 / np.sqrt(counts)
    return {
        "none": np.ones(num_classes, np.float32),
        "inverse_frequency": (inverse / inverse.mean()).astype(np.float32),
        "sqrt_inverse": (sqrt_inverse / sqrt_inverse.mean()).astype(np.float32),
        "effective_number": effective_number_weights(labels, num_classes=num_classes),
    }


def oversample(df: pd.DataFrame, target_share: float = 0.5,
               seed: int = config.SEED) -> pd.DataFrame:
    """Re-sample each minority grade (with replacement) until it has at least
    `target_share` x the majority count. Only ever applied to TRAIN data."""
    max_n = df["grade"].value_counts().max()
    parts = []
    for g, part in df.groupby("grade"):
        need = int(target_share * max_n) - len(part)
        parts.append(part)
        if need > 0:
            parts.append(part.sample(need, replace=True, random_state=seed + int(g)))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_augmentation_grid(image_path: str, out_path, n: int = 8) -> None:
    """Evidence figure: one pre-processed image and n random augmentations."""
    import sys
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    img = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
    img = tf.cast(img, tf.float32)[None]
    aug = build_augmenter(seed=None, shortcut_aware=True)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.4))
    axes[0].imshow(img[0].numpy().astype("uint8"))
    axes[0].set_title("Input", fontsize=9)
    for i in range(1, n + 1):
        out = aug(img, training=True)[0].numpy().clip(0, 255).astype("uint8")
        axes[i].imshow(out)
        axes[i].set_title(f"Aug {i}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
