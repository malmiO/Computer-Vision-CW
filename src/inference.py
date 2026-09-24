"""
inference.py - the complete decision pipeline for ONE uploaded photo.

    photo -> quality gate -> pre-process -> 8-view TTA -> temperature scaling
          -> grade + referable decision + uncertainty -> Grad-CAM -> advice (+PDF)

It reads every setting (variant, size, temperature, thresholds, Grad-CAM layer)
from the run's meta.json written by train.py and evaluate.py, so the hosted app
behaves exactly like the evaluated model.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from gradcam import GradCAM, overlay
from model import load_for_inference
from losses_metrics import referable_probs, stage_probs, temperature_scale
from preprocess import assess_quality, preprocess


# Plain-language guidance per grade. Wording follows the referral logic of the
# ICO Guidelines for Diabetic Eye Care (Wong et al., 2018); check the current
# guideline before quoting intervals in the report. Not medical advice.
ADVICE = {
    0: ("Routine", "No signs of diabetic retinopathy detected. Continue routine diabetic eye screening."),
    1: ("Routine", "Mild non-proliferative changes. Re-screen as scheduled and keep blood sugar and blood pressure under control."),
    2: ("Refer", "Moderate non-proliferative DR. Refer to an ophthalmologist for examination."),
    3: ("Urgent", "Severe non-proliferative DR. Prompt referral to an ophthalmologist is recommended."),
    4: ("Urgent", "Proliferative DR. Urgent referral to an ophthalmologist is recommended."),
}


TTA_VIEWS = int(os.environ.get("DR_TTA_VIEWS", 8))   # 8 = full dihedral group; 4 = rotations only


def dihedral(x: np.ndarray) -> list:
    """The exact symmetries of one HxWx3 image, as a list of single images.

    Returned one by one rather than as a batch: predicting a batch of 8 needs
    about 2.4 GB of RAM, one at a time about 1.0 GB, which is what lets the app
    run on a free 1 GB host.
    """
    views = []
    for k in range(4):
        r = np.rot90(x, k)
        views += [r, r[:, ::-1]]
    return views[:TTA_VIEWS]


class DRPredictor:
    def __init__(self, model_path: str | Path, meta_path: str | Path, quality_path: str | Path):
        self.model = load_for_inference(model_path)       # float32: faster and exact on CPU
        self.meta = json.loads(Path(meta_path).read_text())
        self.quality_thr = json.loads(Path(quality_path).read_text())
        self.cam = GradCAM(self.model, self.meta["gradcam_layer"])
        self.names = self.meta["class_names"]

    def predict(self, rgb: np.ndarray, force: bool = False) -> dict:
        """rgb: uint8 HxWx3. Set force=True to grade even an ungradable photo."""
        quality = assess_quality(rgb, self.quality_thr)
        result = {"quality": quality, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                  "model": self.meta.get("run_name", "model")}
        if not quality["gradable"] and not force:
            result["status"] = "ungradable"
            return result

        pre = preprocess(rgb, self.meta["variant"], self.meta["size"])
        outs = [self.model(v[None], training=False) for v in dihedral(pre.astype("float32"))]
        sp = np.concatenate([stage_probs(o) for o in outs])
        rp = np.concatenate([referable_probs(o) for o in outs])
        probs = temperature_scale(sp.mean(0, keepdims=True), self.meta.get("temperature", 1.0))[0]
        spread = float(sp.std(0).mean())
        grade = int(probs.argmax())
        ref_p = float(rp.mean())
        heat = self.cam.heatmap(pre.astype("float32"), grade)

        result.update({
            "status": "graded",
            "probs": {n: float(p) for n, p in zip(self.names, probs)},
            "grade": grade, "grade_name": self.names[grade], "confidence": float(probs[grade]),
            "dr_present_prob": float(1 - probs[0]),
            "referable_prob": ref_p,
            "referable": ref_p >= self.meta.get("referable_threshold", 0.5),
            "tta_spread": spread,
            # Review threshold fitted on validation in phase 6: the least confident 10% go to a human
            "needs_review": bool(probs[grade] < self.meta.get("review_confidence", 0.5)),
            "triage": ADVICE[grade][0], "advice": ADVICE[grade][1],
            "preprocessed": pre, "gradcam": overlay(pre, heat),
        })
        # Safety rule: a referable-head alarm always wins over a "routine" grade.
        if result["referable"] and result["triage"] == "Routine":
            result["triage"] = "Refer"
            result["advice"] += " The screening head flags possible referable disease; please refer for review."
        return result


def pdf_report(result: dict, original: np.ndarray, out_path: str | Path) -> str:
    """One-page screening summary (fpdf2)."""
    import tempfile
    import cv2
    from fpdf import FPDF

    pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(True, 15)
    pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, "DR-Sight screening summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Generated {result['timestamp']}   Model: {result['model']}", new_x="LMARGIN", new_y="NEXT"); pdf.ln(3)

    tmp = Path(tempfile.mkdtemp())
    imgs = [("Uploaded", original), ("Pre-processed", result.get("preprocessed")),
            ("Grad-CAM", result.get("gradcam"))]
    x = 10
    for title, im in imgs:
        if im is None:
            continue
        p = tmp / f"{title}.png"
        cv2.imwrite(str(p), cv2.cvtColor(cv2.resize(im, (400, 400)), cv2.COLOR_RGB2BGR))
        pdf.image(str(p), x=x, y=32, w=60); pdf.set_xy(x, 94); pdf.cell(60, 5, title, align="C")
        x += 64
    pdf.set_xy(10, 104)

    pdf.set_font("Helvetica", "B", 12)
    if result["status"] == "ungradable":
        pdf.cell(0, 8, "Result: image quality insufficient - please retake the photo", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for r in result["quality"]["reasons"]:
            pdf.cell(0, 6, f"- {r}", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 8, f"Predicted stage: {result['grade_name']}  ({result['confidence']:.0%})", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Triage: {result['triage']}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, result["advice"]); pdf.ln(2)
        for name, p in result["probs"].items():
            pdf.cell(60, 6, name); pdf.cell(0, 6, f"{p:.1%}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"Referable DR probability: {result['referable_prob']:.1%}", new_x="LMARGIN", new_y="NEXT")
        if result["needs_review"]:
            pdf.set_text_color(170, 40, 40)
            pdf.cell(0, 6, "Low confidence: human grader review recommended.", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
    pdf.ln(4); pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, "Research prototype for a university coursework. Not a medical device and not a "
                         "diagnosis. All results must be confirmed by a qualified eye-care professional.")
    pdf.output(str(out_path))
    return str(out_path)
