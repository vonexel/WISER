#!/usr/bin/env bash


set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENTS=(
  "E00_baseline"
  "E01_rp_mphc_only"
  "E02_awb_sbi_only"
  "E03_cnd_only"
  "E04_wiser"
  "E05_no_rp_mphc"
  "E06_no_awb_sbi"
  "E07_no_cnd"
  "E08_no_calibration"
  "E09_full"
)

SEEDS=(0)

# $1 = single experiment, $2 = single seed
[[ $# -ge 1 ]] && EXPERIMENTS=("$1")
[[ $# -ge 2 ]] && SEEDS=("$2")

uv run python scripts/sanity_check.py

START=$(date +%s)
for EXP in "${EXPERIMENTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT_DIR="outputs/${EXP}/${SEED}"
    if [[ -f "${OUT_DIR}/metrics.json" ]]; then
      echo "[run_ablation] skip ${EXP}/${SEED} (already complete)"; continue
    fi
    echo "[run_ablation] === ${EXP} seed=${SEED} ==="
    if [[ "$EXP" == "E13_calibration_only" ]]; then
      SRC_CKPT="outputs/E01_full/${SEED}/ckpt/best.pt"
      if [[ ! -f "${SRC_CKPT}" ]]; then
        echo "[run_ablation] WARNING: ${SRC_CKPT} missing — skipping E13/${SEED}"
        continue
      fi
      uv run python scripts/calibrate_only.py \
        --src-ckpt "${SRC_CKPT}" \
        --out-dir  "${OUT_DIR}" \
        --seed     "${SEED}"
    else
      uv run python scripts/train.py \
        experiment="${EXP}" \
        seed="${SEED}" \
        hydra.run.dir="${OUT_DIR}"
    fi
  done
done

uv run python scripts/compile_results.py
uv run python scripts/make_visualizations.py
if [[ -f outputs/E01_full/0/ckpt/best.pt ]]; then
  uv run python scripts/run_robustness.py --ckpt outputs/E01_full/0/ckpt/best.pt
fi

END=$(date +%s); EL=$((END-START))
printf "[run_ablation] done in %dh%02dm%02ds\n" $((EL/3600)) $(((EL%3600)/60)) $((EL%60))