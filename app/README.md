---
title: DR-Sight
emoji: 👁️
colorFrom: blue
colorTo: gray
sdk: streamlit
app_file: streamlit_app.py
pinned: false
short_description: Explainable diabetic retinopathy screening support
---

# DR-Sight — diabetic retinopathy screening prototype

Upload a colour fundus photo. The app

1. checks image quality (sharpness, exposure, contrast, burnt-out highlights, noise) and asks
   for a retake instead of guessing on a photo it cannot grade,
2. grades diabetic retinopathy on the five-level ICDR scale and flags referable disease
   (moderate or worse),
3. shows a Grad-CAM heat-map of the regions that supported the decision,
4. flags low-confidence cases for a human grader,
5. produces a one-page PDF summary, and can triage a batch of photos into a worklist.

On a held-out test set of 506 images: quadratic weighted kappa 0.88, referable-DR sensitivity
91% and specificity 92%.

Coursework prototype (BSc (Hons) Computer Science, NIBM). **Not a medical device and not a
diagnosis.**
