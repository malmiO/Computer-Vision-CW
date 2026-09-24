"""
report_figures.py - figures that document pre-processing, augmentation and the
experiment comparison. Run after dataset_audit.py (and again after training
runs to refresh the experiment charts).

Outputs in outputs/figures/:
  preprocessing_stages.png     one image through every step of P2
  preprocessing_histograms.png green-channel histograms before / after
  preprocessing_variants.png   the same 3 images under P0-P3
  augmentation_examples.png    8 random fundus-safe augmentations
  experiments_*.png            bar charts built from outputs/experiments.csv
"""
from __future__ import annotations

import sys
import matplotlib
if "ipykernel" not in sys.modules:   # scripts save figures only; notebooks keep inline display
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from preprocess import cache_dataframe, load_rgb, load_splits, pipeline_stages, preprocess


def preprocessing_figures(df: pd.DataFrame, size: int = 300) -> None:
    sample = df[df.grade == 2].sample(1, random_state=config.SEED).iloc[0]
    stages = pipeline_stages(load_rgb(sample["path"]), size)

    fig, axes = plt.subplots(1, len(stages), figsize=(2.6 * len(stages), 3))
    for ax, (title, im) in zip(axes, stages.items()):
        ax.imshow(im); ax.set_title(title, fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "preprocessing_stages.png", dpi=150); plt.close(fig)

    before, after = stages["3. Pad + resize"], stages["7. Circular mask"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, im, t in [(axes[0], before, "Before (resized only)"), (axes[1], after, "After P2 pipeline")]:
        g = im[..., 1][im.max(2) > 10]
        ax.hist(g, bins=64, color="#2e7d32")
        ax.set(title=f"{t}: green channel  (std={g.std():.1f})", xlabel="intensity", ylabel="pixels")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "preprocessing_histograms.png", dpi=150); plt.close(fig)

    picks = df.groupby("grade").sample(1, random_state=config.SEED).sort_values("grade").iloc[[0, 2, 4]]
    variants = list(config.PREPROCESS_VARIANTS)
    fig, axes = plt.subplots(len(picks), len(variants), figsize=(3 * len(variants), 3 * len(picks)))
    for r, (_, row) in enumerate(picks.iterrows()):
        raw = load_rgb(row["path"])
        for c, v in enumerate(variants):
            axes[r, c].imshow(preprocess(raw, v, size)); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(v, fontsize=10)
        axes[r, 0].text(-20, size / 2, config.CLASS_NAMES[row["grade"]], rotation=90,
                        va="center", ha="right", fontsize=10)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "preprocessing_variants.png", dpi=130); plt.close(fig)


def augmentation_figure(df: pd.DataFrame, size: int = 300) -> None:
    from augment import save_augmentation_grid
    one = cache_dataframe(df.head(1), config.DEFAULT_VARIANT, size)
    save_augmentation_grid(one.iloc[0]["cached_path"], config.FIG_DIR / "augmentation_examples.png")


def experiment_figures() -> None:
    path = config.OUTPUT_DIR / "experiments.csv"
    if not path.exists():
        return
    exp = pd.read_csv(path).drop_duplicates("run_name", keep="last")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    exp = exp.sort_values("best_val_qwk")
    axes[0].barh(exp["run_name"], exp["best_val_qwk"], color="#2b6f8e")
    axes[0].set(xlabel="best validation QWK", title="All experiments")
    axes[1].scatter(exp["ms_per_image"], exp["best_val_qwk"],
                    s=exp["total_params"] / 2e5, alpha=0.6, c="#b03a2e")
    for _, r in exp.iterrows():
        axes[1].annotate(r["backbone"], (r["ms_per_image"], r["best_val_qwk"]), fontsize=8)
    axes[1].set(xlabel="inference ms / image", ylabel="best validation QWK",
                title="Accuracy vs speed (bubble = parameters)")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "experiments_overview.png", dpi=150); plt.close(fig)
    exp.to_csv(config.OUTPUT_DIR / "experiments_clean.csv", index=False)


if __name__ == "__main__":
    splits = load_splits()
    train = splits[splits.split == "train"]
    preprocessing_figures(train)
    augmentation_figure(train)
    experiment_figures()
    print("Figures written to", config.FIG_DIR)
