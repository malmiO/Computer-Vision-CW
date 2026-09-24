"""
data.py - tf.data input pipelines.

Each element is (image, {"stage": one_hot_5, "referable": 0/1}) so the same
dataset feeds both heads of the multi-task model.

Images arrive already pre-processed and resized (see preprocess.py), so the
pipeline only decodes, augments (train only), batches and prefetches. Pixel
values stay in [0, 255]; the model itself applies the scaling its backbone
expects (see model.py), which keeps training and the web app consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

import config

AUTOTUNE = tf.data.AUTOTUNE


def _decode(path: tf.Tensor, size: int) -> tf.Tensor:
    """Read a cached PNG as uint8. Resizing is a no-op when the cache size matches,
    and rounding back to uint8 then leaves every pixel value unchanged."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (size, size))
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def _targets(grade: tf.Tensor) -> dict:
    return {
        "stage": tf.one_hot(grade, config.NUM_CLASSES),
        "referable": tf.cast(grade >= config.REFERABLE_FROM_GRADE, tf.float32)[None],
    }


def make_dataset(df: pd.DataFrame, size: int, batch: int = config.BATCH_SIZE,
                 training: bool = False, augmenter=None) -> tf.data.Dataset:
    """Build a batched dataset from a dataframe with 'cached_path' and 'grade'."""
    paths = df["cached_path"].astype(str).values
    grades = df["grade"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, grades))
    # Decode every image once and keep it in memory as uint8 (~0.5 GB for the
    # oversampled training set at 224 px). Later epochs skip the disk read and PNG
    # decode, which is what limits speed on Colab's 2 CPU cores.
    ds = ds.map(lambda p, g: (_decode(p, size), g), num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, g: (tf.cast(x, tf.float32), _targets(g)), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def images_only(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Strip labels (for model.predict)."""
    return ds.map(lambda x, y: x)
