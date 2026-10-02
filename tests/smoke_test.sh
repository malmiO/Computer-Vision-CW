#!/usr/bin/env bash
# smoke_test.sh - proves every script runs end-to-end in a few minutes on CPU.
# Uses 100 cartoon images and a randomly initialised network, so the numbers
# it prints are meaningless; it only checks that the code works.
#
#   bash tests/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TOY=/tmp/drsight_toy
export DR_RAW_DIR=$TOY/train_images DR_LABELS_CSV=$TOY/train.csv DR_WORK_DIR=/tmp/drsight_toy_work DR_CACHE_DIR=/tmp/drsight_toy_work/cache
export TF_CPP_MIN_LOG_LEVEL=3
rm -rf "$TOY" "$DR_WORK_DIR"

python tests/make_toy_dataset.py "$TOY"
cd src
python dataset_audit.py
python preprocess.py --variant P2_clahe_unsharp --size 224
python report_figures.py
python train.py --backbone efficientnet_b0 --epochs_head 1 --epochs_ft 1 --batch 8 \
                --no_pretrained --run_name smoke
python evaluate.py --run smoke
python robustness.py --run smoke --n 10
python - <<'EOF'
# The web app (phase 7) will call inference.py, so test that directly.
import os, numpy as np
from pathlib import Path
from PIL import Image
from inference import DRPredictor, pdf_report
out = Path(os.environ["DR_WORK_DIR"]) / "outputs"
p = DRPredictor(out / "runs/smoke/best.keras", out / "runs/smoke/meta.json", out / "quality_thresholds.json")
img = np.array(Image.open("/tmp/drsight_toy/train_images/toy0090.png").convert("RGB"))
res = p.predict(img, force=True)
assert res["status"] == "graded" and len(res["probs"]) == 5
pdf_report(res, img, "/tmp/drsight_smoke.pdf")
print("Inference + PDF OK")
EOF
cd ..
echo "SMOKE TEST PASSED"
