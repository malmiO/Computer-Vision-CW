"""
train.py - two-phase transfer learning with full experiment logging.

Phase 1 (warm-up)   backbone frozen, only the new heads train at LR_HEAD.
                    Random heads produce large gradients; if the backbone
                    were trainable they would wreck the ImageNet features.
Phase 2 (fine-tune) top UNFREEZE_FRACTION of the backbone unfrozen (BN
                    kept frozen), 10x smaller LR so pretrained filters are
                    adapted, not overwritten.

Callbacks (all monitor validation QWK, the ordinal metric):
    QWKCallback -> EarlyStopping(restore_best_weights) -> ReduceLROnPlateau
    -> ModelCheckpoint(best only) -> CSVLogger

Every run writes outputs/runs/<run_name>/ with history.json, curves.png,
best.keras, meta.json and appends one line to outputs/experiments.csv, which
becomes the experiment table in the report.

Examples:
    python src/train.py --backbone efficientnet_b3 --variant P2_clahe_unsharp --balance both
    python src/train.py --backbone resnet50v2 --epochs_ft 10 --run_name bench_resnet
    python src/train.py --aug shortcut --run_name aug_shortcut
"""
from __future__ import annotations

import argparse
import sys
import json
import time

import matplotlib
if "ipykernel" not in sys.modules:   # scripts save figures only; notebooks keep inline display
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tensorflow import keras

import config
from augment import build_augmenter, effective_number_weights, oversample
from data import images_only, make_dataset
from losses_metrics import QWKCallback, bootstrap_ci, class_balanced_focal_loss, qwk, stage_probs
from model import build_model, count_params, set_backbone_trainable
from preprocess import cache_dataframe, load_splits


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE, choices=list(config.BACKBONES))
    ap.add_argument("--variant", default=config.DEFAULT_VARIANT, choices=list(config.PREPROCESS_VARIANTS))
    ap.add_argument("--balance", default="both", choices=["none", "weights", "oversample", "both"])
    ap.add_argument("--size", type=int, default=None, help="override backbone native size")
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--epochs_head", type=int, default=config.EPOCHS_HEAD)
    ap.add_argument("--epochs_ft", type=int, default=config.EPOCHS_FINETUNE)
    ap.add_argument("--lr_head", type=float, default=config.LR_HEAD)
    ap.add_argument("--lr_ft", type=float, default=config.LR_FINETUNE)
    ap.add_argument("--unfreeze", type=float, default=config.UNFREEZE_FRACTION)
    ap.add_argument("--dropout", type=float, default=config.DROPOUT)
    ap.add_argument("--aug", default="standard", choices=["none", "standard", "shortcut"],
                    help="none | standard fundus-safe | standard + sharpness jitter")
    ap.add_argument("--mixed_precision", action="store_true", help="faster on T4/P100 GPUs")
    ap.add_argument("--no_pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--seed", type=int, default=config.SEED,
                    help="changes weight init, shuffling and augmentation - not the split")
    ap.add_argument("--run_name", default=None)
    return ap.parse_args()


def compile_model(model, lr: float, class_weights) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={"stage": class_balanced_focal_loss(class_weights),
              "referable": keras.losses.BinaryCrossentropy()},
        loss_weights={"stage": 1.0, "referable": config.REFERABLE_LOSS_WEIGHT},
        metrics={"stage": [keras.metrics.CategoricalAccuracy(name="acc")],
                 "referable": [keras.metrics.AUC(name="auc")]},
    )


def callbacks(run_dir, val_ds, val_labels):
    return [
        QWKCallback(images_only(val_ds), val_labels),       # must be first
        keras.callbacks.EarlyStopping(monitor="val_qwk", mode="max",
                                      patience=config.EARLY_STOP_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_qwk", mode="max", factor=0.5,
                                          patience=config.REDUCE_LR_PATIENCE,
                                          min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"), monitor="val_qwk",
                                        mode="max", save_best_only=True),
        keras.callbacks.CSVLogger(str(run_dir / "log.csv"), append=True),
    ]


def plot_curves(history: dict, phase1_epochs: int, out_path) -> None:
    """Loss / accuracy / QWK curves with the phase boundary marked."""
    panels = [("loss", "Total loss"), ("stage_acc", "Stage accuracy"), ("qwk", "Quadratic weighted kappa")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(axes, panels):
        if key in history:
            ax.plot(history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="validation")
        ax.axvline(phase1_epochs - 0.5, ls="--", c="grey", lw=1)
        ax.text(phase1_epochs - 0.4, ax.get_ylim()[1], " fine-tune", va="top", fontsize=8, color="grey")
        ax.set(title=title, xlabel="epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config.set_global_seed(args.seed)

    size = args.size or config.BACKBONES[args.backbone]["size"]
    run_name = args.run_name or f"{args.backbone}_{args.variant}_{args.balance}"
    run_dir = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    splits = cache_dataframe(load_splits(), args.variant, size)
    train_df = splits[splits.split == "train"]
    val_df = splits[splits.split == "val"]
    if args.balance in ("oversample", "both"):
        train_df = oversample(train_df)
    class_weights = (effective_number_weights(splits[splits.split == "train"]["grade"].values)
                     if args.balance in ("weights", "both") else np.ones(config.NUM_CLASSES, "float32"))
    print("Train grade counts:", train_df["grade"].value_counts().sort_index().tolist())
    print("Class weights:", np.round(class_weights, 3).tolist())

    augmenter = None if args.aug == "none" else \
        build_augmenter(shortcut_aware=args.aug == "shortcut")   # built in float32 on purpose
    train_ds = make_dataset(train_df, size, args.batch, training=True, augmenter=augmenter)
    val_ds = make_dataset(val_df, size, args.batch)

    # ---------------- model ----------------
    if args.mixed_precision:   # float16 maths on the GPU, float32 output heads (see model.py)
        keras.mixed_precision.set_global_policy("mixed_float16")
    model, backbone_layers, gradcam_layer = build_model(args.backbone, size, pretrained=not args.no_pretrained,
                                                        dropout=args.dropout)
    t0 = time.time()

    # Phase 1: heads only
    set_backbone_trainable(model, backbone_layers, 0.0)
    compile_model(model, args.lr_head, class_weights)
    print("Phase 1 params:", count_params(model))
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, shuffle=False,  # tf.data already shuffles
                   callbacks=callbacks(run_dir, val_ds, val_df.grade.values), verbose=2)

    # Phase 2: fine-tune the top of the backbone
    n = set_backbone_trainable(model, backbone_layers, args.unfreeze)
    compile_model(model, args.lr_ft, class_weights)   # re-compile after changing trainable
    print(f"Phase 2: {n} backbone layers trainable, params:", count_params(model))
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_ft, shuffle=False,
                   callbacks=callbacks(run_dir, val_ds, val_df.grade.values), verbose=2)
    train_minutes = (time.time() - t0) / 60

    # ---------------- save evidence ----------------
    history = {k: [float(v) for v in h1.history.get(k, [])] + [float(v) for v in h2.history.get(k, [])]
               for k in set(h1.history) | set(h2.history)}
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    plot_curves(history, len(h1.history["loss"]), run_dir / "curves.png")

    # Inference speed on CPU/GPU (single image, averaged) - matters for hosting
    dummy = np.zeros((1, size, size, 3), "float32")
    model.predict(dummy, verbose=0)
    t = time.time()
    for _ in range(10):
        model.predict(dummy, verbose=0)
    ms_per_image = (time.time() - t) / 10 * 1000

    # Validation metrics of the SAVED model (not just the best logged epoch), with a
    # bootstrap 95% confidence interval so runs can be compared honestly.
    val_probs = stage_probs(model.predict(images_only(val_ds), verbose=0))
    y_val, p_val = val_df.grade.values, val_probs.argmax(1)
    ci_low, ci_high = bootstrap_ci(y_val, p_val)
    pd.DataFrame({"id": val_df.id.values, "grade": y_val, "pred": p_val,
                  **{f"p{c}": val_probs[:, c] for c in range(config.NUM_CLASSES)}}
                 ).to_csv(run_dir / "val_predictions.csv", index=False)
    best = int(np.argmax(history.get("val_qwk", [0])))
    train_acc, val_acc = history.get("stage_acc", [np.nan])[best], history.get("val_stage_acc", [np.nan])[best]

    meta = {
        "run_name": run_name, "backbone": args.backbone, "variant": args.variant,
        "balance": args.balance, "augmentation": args.aug, "size": size,
        "lr_head": args.lr_head, "lr_ft": args.lr_ft, "unfreeze": args.unfreeze, "dropout": args.dropout,
        "seed": args.seed, "batch": args.batch, "epochs_head": args.epochs_head, "epochs_ft": args.epochs_ft,
        "class_names": config.CLASS_NAMES, "gradcam_layer": gradcam_layer,
        "val_qwk": round(qwk(y_val, p_val), 4),
        "val_qwk_ci_low": round(ci_low, 4), "val_qwk_ci_high": round(ci_high, 4),
        "val_macro_f1": round(float(f1_score(y_val, p_val, average="macro")), 4),
        "val_accuracy": round(float((y_val == p_val).mean()), 4),
        "best_logged_val_qwk": round(max(history.get("val_qwk", [0])), 4),
        "best_epoch": best + 1, "epochs_run": len(history.get("loss", [])),
        "train_val_acc_gap": round(float(train_acc - val_acc), 4),
        "train_minutes": round(train_minutes, 1), "ms_per_image": round(ms_per_image, 1),
        **count_params(model),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    model.save(run_dir / "best.keras")   # EarlyStopping restored the best weights

    # One row per run. Read-concat-write keeps the columns aligned even if a
    # later version of this script records extra fields.
    exp_csv = config.OUTPUT_DIR / "experiments.csv"
    row = pd.DataFrame([{k: v for k, v in meta.items() if k != "class_names"}])
    table = pd.concat([pd.read_csv(exp_csv), row], ignore_index=True) if exp_csv.exists() else row
    table.to_csv(exp_csv, index=False)
    print(json.dumps({k: v for k, v in meta.items() if k != "class_names"}, indent=2))


if __name__ == "__main__":
    main()
