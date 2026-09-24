"""
make_toy_dataset.py - SMOKE TEST ONLY.

Draws 100 cartoon "fundus" images (orange disc, vessels, red/yellow dots whose
number grows with the grade) in the APTOS layout, so every script can be run
end-to-end on a laptop CPU in a few minutes to prove the code works.
These images are NEVER used for any result in the report.

    python tests/make_toy_dataset.py /tmp/toy
    export DR_RAW_DIR=/tmp/toy/train_images DR_LABELS_CSV=/tmp/toy/train.csv DR_WORK_DIR=/tmp/toy_work
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def fundus(grade: int, w=640, h=480) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    c, r = (w // 2, h // 2), int(h * rng.uniform(0.40, 0.48))
    base = tuple(int(v * rng.uniform(0.7, 1.2)) for v in (40, 90, 200))
    cv2.circle(img, c, r, base, -1)                                             # BGR orange retina
    side = 1 if rng.random() < 0.5 else -1                                      # left / right eye
    disc = (int(c[0] + side * r * rng.uniform(0.35, 0.6)), int(c[1] + rng.uniform(-0.2, 0.2) * r))
    cv2.circle(img, disc, r // 7, (150, 210, 250), -1)                          # optic disc
    for _ in range(int(rng.integers(6, 12))):                                   # vessels
        a = rng.uniform(0, 2 * np.pi)
        mid = (int(disc[0] + np.cos(a + 0.3) * r * 0.5), int(disc[1] + np.sin(a + 0.3) * r * 0.5))
        end = (int(disc[0] + np.cos(a) * r * 1.2), int(disc[1] + np.sin(a) * r * 1.2))
        cv2.line(img, disc, mid, (20, 30, 120), 3); cv2.line(img, mid, end, (20, 30, 120), 2)
    img[np.linalg.norm(np.mgrid[0:h, 0:w].transpose(1, 2, 0) - [c[1], c[0]], axis=2) > r] = 0
    for _ in range(grade * 12):                                                 # lesions
        p = (int(c[0] + rng.uniform(-0.7, 0.7) * r), int(c[1] + rng.uniform(-0.7, 0.7) * r))
        colour = (20, 20, 150) if rng.random() < 0.6 else (60, 220, 230)
        cv2.circle(img, p, int(rng.integers(2, 5)), colour, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/toy")
    (root / "train_images").mkdir(parents=True, exist_ok=True)
    grades = np.repeat(np.arange(5), [40, 15, 25, 10, 10])
    rows = []
    for i, g in enumerate(grades):
        name = f"toy{i:04d}"
        cv2.imwrite(str(root / "train_images" / f"{name}.png"), fundus(int(g)))
        rows.append({"id_code": name, "diagnosis": int(g)})
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    print(f"Wrote {len(rows)} toy images to {root}")
