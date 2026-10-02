"""
project_journey.py - the read-only "Project Journey" tab.

Nine collapsible stages, from raw dataset to live deployment. All numbers are the
measured results from the project's experiment logs; nothing here is computed at run time,
so the tab never loads the model.

Optional figures: put PNGs in docs/figures/ (names in FIGURES below). A missing file just
means that stage is shown as text only.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "docs" / "figures"

INTRO = ("From raw dataset to live deployment — every decision below is backed by a "
         "measurement, not a convention.")

# file name inside docs/figures/  ->  caption. Photo-free diagrams and charts only.
FIGURES = {
    1: ("fig06_duplicate_audit.png", "Duplicate audit: 184 groups, 47 with conflicting grades, 293 images removed."),
    2: ("fig_leakage_control.png", "What each split may be used for. The test set is opened once."),
    3: ("fig02_preprocessing_pipeline.png", "The seven pre-processing steps."),
    4: ("fig11_augmentation_sharpness_loss.png", "Sharpness lost by rotation + zoom versus the 8 exact symmetries."),
    5: ("backbone_benchmark.png", "Backbone benchmark: validation QWK with 95% intervals."),
    6: ("ablations.png", "Ablations on DenseNet121."),
    7: ("eval_confusion_matrix.png", "Test confusion matrix (506 images)."),
    8: ("eval_robustness.png", "Test QWK under synthetic corruptions."),
    9: ("fig_deployment_flow.png", "How one uploaded photo moves through the live app."),
}

STAGES = [
    {
        "title": "Choose & audit data",
        "what": "Selected APTOS 2019 (3,662 fundus photos) over EyePACS 2015, for documented provenance "
                "and origin (Aravind Eye Hospital, India).",
        "why": "Needed a dataset small enough to run 20 experiments on a free GPU budget, with a known, "
               "citable collection protocol.",
        "chips": [("184", "near-duplicate groups"), ("47", "with conflicting grades"),
                  ("293", "images removed (8.0%)"), ("3,369", "images kept")],
        "evidence": "Built a 64-bit perceptual hash (DCT-based) to find near-duplicate photos. Found 184 "
                    "near-duplicate groups (430 images), 47 of which had CONFLICTING grades between copies — "
                    "direct evidence of label noise. Removed 293 images (8.0%). 3,369 images remained.",
    },
    {
        "title": "Split without leakage",
        "what": "Stratified 70/15/15 split (2,357 / 506 / 506), done AFTER deduplication, at group level, "
                "seed 42.",
        "why": "A duplicate split across train/test would let the model memorise instead of generalise, "
               "inflating the test score.",
        "chips": [("2,357 / 506 / 506", "train / validation / test"),
                  ("39.8 / 39.9 / 39.7%", "referable share per split")],
        "evidence": "Referable-case share matched across all three splits (39.8% / 39.9% / 39.7%), "
                    "confirming the stratification held.",
    },
    {
        "title": "Pre-process",
        "what": "7-step pipeline — FOV crop, pad to square, resize (INTER_AREA), median filter, CLAHE on the "
                "L channel only, unsharp mask, circular mask.",
        "why": "Up to 40% of each photo was dead black border; 17 different cameras needed normalising "
               "without destroying diagnostic colour or fine lesions.",
        "chips": [("4", "pipeline variants compared"), ("0.644", "validation macro-F1 of the chosen P2")],
        "evidence": "Tested 4 pipeline variants head-to-head. The chosen pipeline (P2) won on validation "
                    "macro-F1 (0.644) while keeping natural colour, unlike the Ben Graham method (P3) which "
                    "false-coloured the image.",
    },
    {
        "title": "Augment & balance",
        "what": "8 exact pixel-permutation symmetries (flips/rotations/transpose), retina-only "
                "brightness/contrast jitter, shortcut-aware sharpness jitter, effective-number class weights "
                "+ partial oversampling + focal loss.",
        "why": "Standard rotation+zoom augmentation requires interpolation, which measurably destroys the "
               "fine detail the diagnosis depends on.",
        "chips": [("−59%", "sharpness lost to rotation + zoom"), ("0%", "lost to the 8 exact symmetries"),
                  ("0.672 → 0.626", "sharpness-shortcut AUC")],
        "evidence": "Arbitrary rotation+zoom cut image sharpness by 59%. The 8 exact symmetries used "
                    "instead: 0% loss. A measured \"sharpness shortcut\" (where sharper photos predicted "
                    "\"healthy\" at AUC 0.672) was reduced to AUC 0.626 by the shortcut-aware augmentation.",
    },
    {
        "title": "Pick a backbone",
        "what": "Benchmarked 5 ImageNet-pretrained CNNs (DenseNet121, MobileNetV3, EfficientNet-B0/B3, "
                "ResNet50V2) under IDENTICAL conditions.",
        "why": "Better ImageNet accuracy doesn't guarantee better transfer to medical images — had to "
               "measure, not assume.",
        "chips": [("0.8804", "DenseNet121 validation QWK"), ("0.850–0.906", "95% confidence interval"),
                  ("+0.012", "over the runner-up")],
        "evidence": "DenseNet121 won, validation QWK 0.8804 (95% CI 0.850-0.906), beating the runner-up by "
                    "0.012 — just outside the 0.01 tie margin.",
    },
    {
        "title": "Train & ablate",
        "what": "Two-phase transfer learning (3 epochs frozen head warm-up, then 12 epochs with the top 30% "
                "of the backbone unfrozen at a 10x lower learning rate). 20 training runs, 329 GPU-minutes "
                "total.",
        "why": "A freshly-initialised head produces noisy gradients that would destroy pretrained features "
               "if unfrozen too early.",
        "chips": [("20", "training runs"), ("329", "GPU-minutes"),
                  ("6 of 6", "unfreeze=0.6 runs collapsed"),
                  ("+0.145", "macro-F1 from balancing, accuracy 0.7885 both ways")],
        "evidence": "A hyperparameter (unfreeze=0.6) tuned on a cheap proxy model caused 6 of 6 DenseNet121 "
                    "runs to collapse (QWK as low as 0.0). Reverted to unfreeze=0.3 (stable, QWK 0.880) and "
                    "reported the failure honestly. Class balancing alone added +0.145 macro-F1 at IDENTICAL "
                    "accuracy (0.7885 both ways) — proof it redistributes errors toward rare, "
                    "sight-threatening grades rather than improving the headline number.",
    },
    {
        "title": "Evaluate (test set, opened once)",
        "what": "Final test on 506 held-out images, touched only once, after every decision above was "
                "already locked in on validation data.",
        "why": "To get an honest, unbiased estimate of real-world performance.",
        "chips": [("0.876", "test QWK"), ("79.5%", "accuracy"), ("90.6%", "referable sensitivity"),
                  ("92.1%", "referable specificity"), ("0.973", "referable AUC"),
                  ("0.204 → 0.027", "calibration error (ECE)")],
        "evidence": "Test QWK 0.876 (vs. validation 0.8804 — close agreement, confirming no leakage). "
                    "Accuracy 79.5%. Referable-DR sensitivity 90.6%, specificity 92.1%, AUC 0.973. "
                    "Calibration error (ECE) cut from 0.204 to 0.027 via temperature scaling.",
    },
    {
        "title": "Stress-test",
        "what": "Synthetic corruption benchmark — 7 corruption types (blur, noise, JPEG compression, low "
                "light, etc.) at 3 severities each, plus Grad-CAM visual explanation checks and a "
                "camera-only shortcut baseline.",
        "why": "A test score alone can't prove the model is looking at the retina instead of exploiting a "
               "shortcut like camera/resolution.",
        "chips": [("0.876 vs 0.614", "model vs camera-only QWK"),
                  ("0.837", "QWK inside the one camera group that can't reveal the grade"),
                  ("≈ 0", "QWK under heavy JPEG (severity 3)")],
        "evidence": "The model beats the camera-only floor (QWK 0.876 vs 0.614) and holds QWK 0.837 even "
                    "inside the one camera group that can't reveal the grade — that's the evidence it isn't "
                    "just reading the shortcut. It was most vulnerable to heavy JPEG compression (QWK near 0 "
                    "at severity 3). The quality gate also missed over-exposure and noise, which motivated "
                    "v2 (clipping + noise checks); JPEG still slips through.",
    },
    {
        "title": "Deploy",
        "what": "Live Streamlit web app with a quality gate, 8-view test-time augmentation, calibrated "
                "confidence, Grad-CAM, and a batch worklist for triaging multiple patients at once.",
        "why": "A model in a notebook helps nobody — it had to be a working, publicly reachable tool.",
        "chips": [("92 → 32 MB", "model size"), ("2.4 → 0.94 GB", "peak memory"),
                  ("3.12", "Python version pinned")],
        "evidence": "Solved 4 real deployment blockers — switched from Hugging Face Spaces (required "
                    "payment) to Streamlit Community Cloud (free); shrank the model from 92MB to 32MB to fit "
                    "GitHub's limit; cut peak memory from 2.4GB to 0.94GB by running TTA views one at a time "
                    "instead of batched; fixed a Python-version build failure by pinning Python 3.12.",
    },
]

# Reuses the app's palette (.streamlit/config.toml): teal #1F6F8B, panel #F1F5F7, ink #17313E.
_CSS = """
<style>
  .st-key-journey {counter-reset: stage; border-left: 2px solid #C9DCE4;
                   margin: 6px 0 0 16px; padding-left: 24px;}
  .st-key-journey [data-testid="stExpander"] {counter-increment: stage; position: relative;
        overflow: visible; background: #fff; border: 1px solid #D5E2E8; border-radius: 12px;
        margin-bottom: 14px;}
  .st-key-journey [data-testid="stExpander"] details {overflow: visible;}
  .st-key-journey [data-testid="stExpander"]::before {content: counter(stage); position: absolute;
        left: -40px; top: 9px; width: 28px; height: 28px; border-radius: 50%; background: #1F6F8B;
        color: #fff; font-weight: 700; font-size: .9rem; line-height: 28px; text-align: center;
        box-shadow: 0 0 0 4px #fff;}
  .st-key-journey [data-testid="stExpander"] summary p {font-weight: 650; font-size: 1.05rem;
        color: #17313E;}
  .jy-intro {font-size: 1.05rem; color: #17313E; margin: 0 0 6px 0;}
  .jy-label {font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
        color: #1F6F8B; margin: 10px 0 2px 0;}
  .jy-text {margin: 0; color: #17313E; font-size: .98rem; line-height: 1.5;}
  .jy-chips {display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 10px 0;}
  .jy-chip {background: #F1F5F7; border-left: 3px solid #1F6F8B; border-radius: 0 8px 8px 0;
        padding: 6px 12px; min-width: 110px; max-width: 100%;}
  .jy-chip b {display: block; font-size: 1.2rem; color: #1F6F8B; line-height: 1.25;}
  .jy-chip span {display: block; font-size: .78rem; color: #5B6770; line-height: 1.3;}
  .jy-ev {background: #F1F5F7; border-left: 3px solid #1F6F8B; border-radius: 0 8px 8px 0;
        padding: 8px 14px; font-size: .9rem; color: #17313E; line-height: 1.5; margin: 0;}
</style>
"""


def _p(label: str, text: str) -> str:
    return f"<div class='jy-label'>{label}</div><p class='jy-text'>{html.escape(text)}</p>"


def _chips(chips: list[tuple[str, str]]) -> str:
    cells = "".join(f"<div class='jy-chip'><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>"
                    for v, k in chips)
    return f"<div class='jy-chips'>{cells}</div>"


def _stage(n: int, s: dict) -> None:
    with st.expander(s["title"], expanded=(n == 1)):
        st.markdown(_p("What", s["what"]) + _p("Why", s["why"])
                    + "<div class='jy-label'>Evidence</div>" + _chips(s["chips"])
                    + f"<p class='jy-ev'>{html.escape(s['evidence'])}</p>",
                    unsafe_allow_html=True)
        name, caption = FIGURES.get(n, (None, ""))
        if name and os.path.exists(FIG_DIR / name):
            st.image(str(FIG_DIR / name), caption=caption, use_container_width=True)


def render() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(f"<p class='jy-intro'>{html.escape(INTRO)}</p>", unsafe_allow_html=True)
    with st.container(key="journey"):
        for n, stage in enumerate(STAGES, 1):
            _stage(n, stage)


if __name__ == "__main__":   # `streamlit run project_journey.py` previews the tab on its own
    st.set_page_config(page_title="Project Journey preview", layout="wide")
    render()
