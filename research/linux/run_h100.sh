#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m research.tools.preflight --batch-size 96
python -m research.tools.launch --gpus 0 --manifest "${WISER_MANIFEST:-research/data/frames.csv}" --workers 8
python -m research.tools.compare
python -m research.tools.spectral --manifest "${WISER_MANIFEST:-research/data/frames.csv}"
