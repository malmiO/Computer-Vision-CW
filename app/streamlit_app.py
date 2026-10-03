"""
streamlit_app.py - DR-Sight, the hosted screening prototype.

  upload -> quality gate -> pre-process -> 8-view TTA -> calibrated grade
         -> triage + human-review flag -> Grad-CAM -> PDF summary

Everything the app decides with (calibration temperature, referral threshold,
review threshold) was fitted on the validation split in phase 6 and travels in
model/meta.json, so the live app behaves exactly like the evaluated model.

Run locally:   streamlit run streamlit_app.py
On Hugging Face Spaces this file is the entry point (sdk: streamlit).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "src"):          # works in the repo and on Spaces
    sys.path.insert(0, str(p))
os.environ.setdefault("DR_WORK_DIR", str(Path(tempfile.gettempdir()) / "drsight"))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODEL_DIR = HERE / "model"
EXAMPLE_DIR = HERE / "examples"

TRIAGE = {   # colour, headline, what the user should do
    "Routine":    ("#2E7D4F", "Routine screening", "No referral indicated by this photo."),
    "Refer":      ("#A8620F", "Refer to an ophthalmologist", "Arrange a specialist appointment."),
    "Urgent":     ("#B42318", "Urgent referral", "Arrange a specialist appointment promptly."),
    "Ungradable": ("#5B6770", "Retake the photo", "The image cannot be graded reliably."),
}

st.set_page_config(page_title="DR-Sight", page_icon="👁️", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1150px; padding-top: 2.2rem;}
  #triage {border-radius: 12px; padding: 20px 24px; color: #fff; margin-bottom: 1rem;}
  #triage h2 {margin: 0 0 6px 0; font-size: 1.85rem; font-weight: 650; color: #fff;}
  #triage p  {margin: 2px 0; font-size: 1.02rem; color: #fff; opacity: .95;}
  .flag {display: inline-block; margin-top: 10px; padding: 4px 12px; border-radius: 999px;
         background: rgba(255,255,255,.22); color: #fff; font-weight: 600; font-size: .9rem;}
  .note {font-size: .86rem; color: #5B6770; border-left: 3px solid #1F6F8B; padding-left: 10px;}
  [data-testid="stMetricValue"] {font-size: 1.4rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model…")
def get_predictor():
    from inference import DRPredictor
    return DRPredictor(MODEL_DIR / "best.keras", MODEL_DIR / "meta.json",
                       MODEL_DIR / "quality_thresholds.json")


def triage_banner(res: dict) -> str:
    key = "Ungradable" if res["status"] == "ungradable" else res["triage"]
    colour, headline, action = TRIAGE[key]
    if res["status"] == "ungradable":
        detail = "Reasons: " + "; ".join(res["quality"]["reasons"]) + "."
        extra = "<p>Retake the photo, or tick “grade anyway” to see a low-trust result.</p>"
    else:
        detail = (f"{res['grade_name']} · {res['confidence']:.0%} confidence · "
                  f"referable-DR probability {res['referable_prob']:.0%}")
        extra = ("<span class='flag'>Low confidence — refer this case to a human grader</span>"
                 if res["needs_review"] else "")
    return (f"<div id='triage' style='background:{colour}'><h2>{headline}</h2>"
            f"<p>{detail}</p><p>{action}</p>{extra}</div>")


def show_result(res: dict, image: np.ndarray):
    st.markdown(triage_banner(res), unsafe_allow_html=True)
    if res["status"] == "graded":
        left, right = st.columns([3, 2])
        with left:
            # keep the clinical order of the grades, which a sorted chart would lose
            probs = pd.DataFrame({"grade": list(res["probs"]),
                                  "probability": [100 * v for v in res["probs"].values()]})
            st.dataframe(probs, hide_index=True, use_container_width=True,
                         column_config={"probability": st.column_config.ProgressColumn(
                             "probability", min_value=0, max_value=100, format="%.0f%%")})
        with right:
            st.metric("Predicted stage", res["grade_name"])
            st.metric("Any DR", f"{res['dr_present_prob']:.0%}")
            st.metric("Referable DR", f"{res['referable_prob']:.0%}")
        st.caption(res["advice"])
        cols = st.columns(3)
        for col, (img, cap) in zip(cols, [(image, "Uploaded photo"),
                                          (res["preprocessed"], "Pre-processed (contrast + edges)"),
                                          (res["gradcam"], f"Grad-CAM: what supported “{res['grade_name']}”")]):
            col.image(img, caption=cap, use_container_width=True)
    else:
        st.image(image, caption="Uploaded photo", width=380)

    q = res["quality"]["metrics"]
    with st.expander("Image-quality check"):
        st.write(pd.DataFrame([{k: round(v, 3) for k, v in q.items()}]))
        st.caption("Thresholds were fitted on the training photos: a photo is refused only if it is "
                   "worse than 99% of the images the model learned from.")

    from inference import pdf_report
    pdf_path = Path(tempfile.mkdtemp()) / "Aura_Retina_report.pdf"
    pdf_report(res, image, pdf_path)
    st.download_button("Download the screening summary (PDF)", pdf_path.read_bytes(),
                       file_name="Aura_Retina_report.pdf", mime="application/pdf")


def single_photo_tab():
    left, right = st.columns([2, 3])
    with left:
        upload = st.file_uploader("Fundus photo", type=["png", "jpg", "jpeg"])
        example = None
        if EXAMPLE_DIR.exists():
            names = ["—"] + sorted(p.name for p in EXAMPLE_DIR.glob("*.*"))
            chosen = st.selectbox("…or try an example photo", names)
            example = EXAMPLE_DIR / chosen if chosen != "—" else None
        force = st.checkbox("Grade anyway if the quality check fails")
        go = st.button("Screen photo", type="primary", use_container_width=True)
        source = upload or example
        if source is not None:
            st.image(str(source) if example and not upload else source, use_container_width=True)
    with right:
        if go and source is None:
            st.warning("Choose a photo first.")
        elif go:
            image = np.array(Image.open(source).convert("RGB"))
            with st.spinner("Checking quality, grading 8 orientations…"):
                res = get_predictor().predict(image, force=force)
            show_result(res, image)
        else:
            st.info("Upload a retinal photo, or pick an example, then press **Screen photo**.")


def batch_tab():
    st.write("Screen several photos at once and get a worklist with the most urgent cases first.")
    files = st.file_uploader("Fundus photos", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Build worklist", type="primary") and files:
        order = {"Urgent": 0, "Refer": 1, "Ungradable": 2, "Routine": 3}
        rows, progress = [], st.progress(0.0)
        for i, f in enumerate(files, 1):
            res = get_predictor().predict(np.array(Image.open(f).convert("RGB")))
            rows.append({"file": f.name,
                         "triage": "Ungradable" if res["status"] == "ungradable" else res["triage"],
                         "stage": res.get("grade_name", "—"),
                         "confidence": round(res.get("confidence", float("nan")), 3),
                         "referable probability": round(res.get("referable_prob", float("nan")), 3),
                         "needs human review": res.get("needs_review", True),
                         "quality issues": "; ".join(res["quality"]["reasons"]) or "none"})
            progress.progress(i / len(files))
        df = pd.DataFrame(rows).sort_values("triage", key=lambda s: s.map(order)).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download the worklist (CSV)", df.to_csv(index=False).encode(),
                           file_name="worklist.csv", mime="text/csv")


def model_card_tab():
    meta = get_predictor().meta
    t = meta.get("test_metrics", {})
    fmt = lambda k: f"{t[k]:.3f}" if isinstance(t.get(k), (int, float)) else "—"
    st.subheader("What this tool does")
    st.write("It grades a colour fundus photo on the five-level International Clinical DR scale, "
             "flags referable disease (moderate or worse), shows which regions supported the "
             "decision, and refuses photos that are too blurred, dark, washed out, over-exposed or "
             "noisy to grade.")
    st.subheader("Model")
    st.write(f"**{meta['backbone']}** pretrained on ImageNet, fine-tuned on APTOS 2019; input "
             f"{meta['size']}×{meta['size']} px; pre-processing `{meta['variant']}`; two output heads "
             f"(stage and referable); 8-view test-time augmentation; temperature-scaled probabilities "
             f"(T = {meta.get('temperature', 1):.2f}).")
    st.subheader("Held-out test results (506 images never used in training)")
    st.table(pd.DataFrame({
        "metric": ["Accuracy (5 grades)", "Quadratic weighted kappa", "Macro-F1",
                   "Referable DR: AUC", "Referable DR: sensitivity", "Referable DR: specificity",
                   "Calibration error (ECE)"],
        "value": [fmt("accuracy"), fmt("qwk"), fmt("macro_f1"), fmt("referable_auc"),
                  fmt("referable_sensitivity"), fmt("referable_specificity"), fmt("ece_after")]}))
    st.subheader("Intended use and limitations")
    st.write("- A coursework prototype to demonstrate decision support for screening, **not a "
             "medical device** and not a diagnosis.\n"
             "- Trained on one public dataset from a single country, so performance elsewhere is "
             "unknown; it has not been validated prospectively.\n"
             "- Weakest on Mild and Severe grades, which are the least common in the data.\n"
             "- Does not detect diabetic macular oedema or any other eye disease.\n"
             "- Heavily compressed or very noisy photos remain harder than clinic-quality ones.")


st.title("DR-Sight")
st.caption("Diabetic retinopathy screening support — grade, explanation and triage from a retinal photo.")
tabs = st.tabs(["Screen one photo", "Screen a batch", "About this model", "Project Journey"])
with tabs[0]:
    single_photo_tab()
with tabs[1]:
    batch_tab()
with tabs[2]:
    model_card_tab()
with tabs[3]:
    from project_journey import render as project_journey_tab
    project_journey_tab()
st.markdown("<p class='note'>This prototype built for a Computer Vision coursework. It does not "
            "provide a diagnosis. Every result must be confirmed by a qualified eye-care professional."
            "</p>", unsafe_allow_html=True)
