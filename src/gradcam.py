"""
gradcam.py - Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

For a chosen grade c, Grad-CAM weights each channel of the last convolutional
feature map by the average gradient of the class score with respect to that
channel, sums the weighted channels and keeps the positive part. The result is
a coarse heat-map of the regions that pushed the model towards grade c.

Why it matters here: a clinician will not trust a grade without seeing WHY.
If the heat-map sits on haemorrhages / exudates the model is looking at real
pathology; if it sits on the optic disc rim or camera artefacts it is a
"clever Hans" shortcut, which the error analysis should discuss.
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


class GradCAM:
    def __init__(self, model, layer_name: str):
        self.grad_model = keras.Model(
            model.inputs, [model.get_layer(layer_name).output, model.get_layer("stage").output])

    def heatmap(self, image: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        """image: HxWx3 float32 in [0, 255] (already pre-processed). Returns HxW in [0, 1]."""
        x = tf.convert_to_tensor(image[None].astype("float32"))
        with tf.GradientTape() as tape:
            fmap, probs = self.grad_model(x, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(probs[0]))
            # log-probability: larger, better-conditioned gradients than the raw
            # softmax output when a trained model is very confident
            score = tf.math.log(tf.cast(probs[:, class_idx], tf.float32) + 1e-7)
        # Gradient w.r.t. the ORIGINAL feature map; cast afterwards. (Casting
        # inside the tape would create a new tensor the score does not depend
        # on, which breaks models trained with mixed precision.)
        grads = tf.cast(tape.gradient(score, fmap), tf.float32)
        fmap = tf.cast(fmap, tf.float32)
        weights = tf.reduce_mean(grads, axis=(1, 2))                  # (1, C)
        cam = tf.nn.relu(tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1))[0].numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def overlay(image: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET-coloured heat-map over the RGB image (outside the retina stays dark)."""
    img = image.astype("uint8")
    colour = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype("uint8"), cv2.COLORMAP_JET),
                          cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img, 1 - alpha, colour, alpha, 0)
    retina = (img.max(axis=2) > 5)[..., None]
    return np.where(retina, out, img)
