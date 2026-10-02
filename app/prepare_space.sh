#!/usr/bin/env bash
# prepare_space.sh - assemble everything the Hugging Face Space needs into ./space
#
#   bash app/prepare_space.sh <outputs_dir> <run_name> [examples_dir]
#   bash app/prepare_space.sh outputs bench_densenet121 my_test_photos
set -euo pipefail

OUT_DIR=${1:?outputs directory}
RUN=${2:?run name, e.g. bench_densenet121}
EXAMPLES=${3:-}
SPACE=space

rm -rf "$SPACE" && mkdir -p "$SPACE/model" "$SPACE/examples"

# 1. the app and the five modules it imports
cp app/streamlit_app.py app/project_journey.py app/README.md app/requirements.txt "$SPACE/"
mkdir -p "$SPACE/.streamlit" && cp app/.streamlit/config.toml "$SPACE/.streamlit/"
for f in config preprocess gradcam losses_metrics model inference; do
  cp "src/$f.py" "$SPACE/"
done

# 2. trained model + the settings validated in phase 6.
#    The training file carries Adam's optimizer state (~92 MB); re-saving it for
#    inference drops that to ~32 MB, which fits GitHub's 100 MB limit without Git
#    LFS - and Streamlit Community Cloud cannot read LFS files.
python - "$OUT_DIR/runs/$RUN/best.keras" "$SPACE/model/best.keras" <<'PY'
import sys, os
sys.path.insert(0, "src")
from model import load_for_inference
src, dst = sys.argv[1], sys.argv[2]
load_for_inference(src).save(dst)
print(f"   model: {os.path.getsize(src)/1e6:.0f} MB trained -> {os.path.getsize(dst)/1e6:.0f} MB for inference")
PY
cp "$OUT_DIR/runs/$RUN/meta.json" "$SPACE/model/"
cp "$OUT_DIR/quality_thresholds.json" "$SPACE/model/"

# 3. optional example photos (check the dataset licence before publishing any)
if [[ -n "$EXAMPLES" ]]; then cp "$EXAMPLES"/*.png "$SPACE/examples/" 2>/dev/null || true; fi

# 4. large files go through Git LFS on Hugging Face
printf '*.keras filter=lfs diff=lfs merge=lfs -text\n*.h5 filter=lfs diff=lfs merge=lfs -text\n' > "$SPACE/.gitattributes"

echo "Deployment folder ready in ./$SPACE  ($(du -sh "$SPACE" | cut -f1))"
echo "Deploy it to Streamlit Community Cloud (free) by pushing this folder to a public"
echo "GitHub repository, then selecting streamlit_app.py at share.streamlit.io."
echo "The same folder also works as a Hugging Face Space (sdk: streamlit)."
