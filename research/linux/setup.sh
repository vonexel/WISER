#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r research/archive/wiser-second-article-evidence-1.0.5/environment/requirements-lock-cu124.txt
python -m pip install pytest==8.3.5
python -m pip check
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest research/tests -q
python -m pip freeze > research/linux/installed-freeze.txt
echo 'Environment installed. Run the documented GPU and data preflight before full training.'
