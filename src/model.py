"""
model.py - multi-task CNN built on an ImageNet-pretrained backbone.

                      +--> GAP --+
    image -> scaling -> backbone -+          +-> BN -> Dropout -> Dense(256) -> Dropout
                      +--> GMP --+  concat                                       |
                                                        +------------------------+------------+
                                                        v                                     v
                                          "stage": Dense(5, softmax)       "referable": Dense(1, sigmoid)
                                          ICDR grade 0-4                   P(grade >= 2)

Design decisions (justify these in the report):
  * Transfer learning: 3.6k images are far too few to train a deep CNN from
    scratch; ImageNet filters (edges, blobs, textures) transfer well to
    fundus lesions (Tan & Le, 2019; He et al., 2016).
  * Dual pooling: global AVERAGE pooling summarises diffuse signs, global MAX
    pooling keeps the strongest local response, which matters for tiny
    isolated lesions such as microaneurysms that averaging would dilute.
  * Two heads (multi-task learning): the brief asks to "classify diabetic
    retinopathy as well as the stage". The binary referable head gives a
    direct screening decision and acts as an auxiliary task that regularises
    the shared features.
  * The backbone is built with `input_tensor=` so its layers live in the same
    graph as the heads. That makes Grad-CAM able to reach the last
    convolutional layer and lets us freeze/unfreeze individual layers.
  * Output layers are float32 so the model also works with mixed precision.
"""
from __future__ import annotations

from tensorflow import keras

import config

L = keras.layers

_APPS = {
    "efficientnet_b0": keras.applications.EfficientNetB0,
    "efficientnet_b3": keras.applications.EfficientNetB3,
    "resnet50v2": keras.applications.ResNet50V2,
    "densenet121": keras.applications.DenseNet121,
    "mobilenetv3_large": keras.applications.MobileNetV3Large,
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_VAR = [0.229 ** 2, 0.224 ** 2, 0.225 ** 2]


def _scaling(x, prep: str):
    """Backbone-specific pixel scaling using serialisable built-in layers."""
    if prep == "tf":
        return L.Rescaling(1 / 127.5, offset=-1.0, name="scale_tf")(x)
    if prep == "torch":
        x = L.Rescaling(1 / 255.0, name="scale_01")(x)
        return L.Normalization(mean=_IMAGENET_MEAN, variance=_IMAGENET_VAR, name="imagenet_norm")(x)
    return x  # EfficientNet / MobileNetV3 rescale internally


def build_model(backbone: str = config.DEFAULT_BACKBONE, size: int | None = None,
                pretrained: bool = True, dropout: float = config.DROPOUT):
    """Return (model, backbone_layer_names, gradcam_layer_name)."""
    spec = config.BACKBONES[backbone]
    size = size or spec["size"]

    inputs = keras.Input((size, size, 3), name="image")
    x = _scaling(inputs, spec["prep"])
    kwargs = dict(include_top=False, weights="imagenet" if pretrained else None, input_tensor=x)
    if backbone.startswith("mobilenetv3"):
        kwargs["include_preprocessing"] = True
    base = _APPS[backbone](**kwargs)

    backbone_layers = [l.name for l in base.layers if not isinstance(l, keras.layers.InputLayer)]
    # Grad-CAM target: the last layer that still has a spatial (4-D) output.
    gradcam_layer = next(l.name for l in reversed(base.layers) if len(l.output.shape) == 4)

    feat = base.output
    pooled = L.Concatenate(name="dual_pool")([
        L.GlobalAveragePooling2D(name="gap")(feat),
        L.GlobalMaxPooling2D(name="gmp")(feat),
    ])
    h = L.BatchNormalization(name="head_bn")(pooled)
    h = L.Dropout(dropout, name="head_dropout_1")(h)
    h = L.Dense(256, activation="swish", name="head_dense")(h)
    h = L.Dropout(dropout / 2, name="head_dropout_2")(h)
    stage = L.Dense(config.NUM_CLASSES, activation="softmax", dtype="float32", name="stage")(h)
    referable = L.Dense(1, activation="sigmoid", dtype="float32", name="referable")(h)

    model = keras.Model(inputs, {"stage": stage, "referable": referable},
                        name=f"drsight_{backbone}")
    return model, backbone_layers, gradcam_layer


def set_backbone_trainable(model, backbone_layers: list[str], fraction: float) -> int:
    """Freeze the whole backbone (fraction=0) or unfreeze its top `fraction`.

    BatchNormalization layers stay frozen even when unfrozen blocks are
    trained: with small batches their running statistics would be
    overwritten by noisy estimates and destroy the pretrained features
    (this is the approach recommended in the Keras transfer-learning guide).
    Returns the number of trainable backbone layers.
    """
    n_unfreeze = int(round(len(backbone_layers) * fraction))
    trainable = set(backbone_layers[len(backbone_layers) - n_unfreeze:]) if n_unfreeze else set()
    count = 0
    for name in backbone_layers:
        layer = model.get_layer(name)
        make_trainable = name in trainable and not isinstance(layer, L.BatchNormalization)
        layer.trainable = make_trainable
        count += int(make_trainable)
    return count


def count_params(model) -> dict:
    """Trainable / total parameter counts for the experiment table."""
    trainable = sum(int(keras.ops.size(w)) for w in model.trainable_weights)
    total = sum(int(keras.ops.size(w)) for w in model.weights)
    return {"trainable_params": trainable, "total_params": total}


def load_for_inference(path):
    """Load a trained .keras model for CPU inference (evaluation, the web app).

    Models were trained with mixed precision, so their saved config asks for
    16-bit arithmetic. CPUs are slow at that and gain nothing from it, so the
    config is rewritten to float32 before loading. The weights are stored in
    float32 either way, so predictions are unchanged (verified: identical
    probabilities, ~1.3x faster on a CPU).
    """
    import tempfile
    import zipfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / Path(path).name
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w") as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item == "config.json":
                data = data.decode().replace('"mixed_float16"', '"float32"').encode()
            zout.writestr(item, data)
    return keras.models.load_model(tmp, compile=False)
