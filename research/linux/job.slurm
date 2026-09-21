#!/usr/bin/env bash
#SBATCH --job-name=wiser-paper2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=wiser-%j.log
set -euo pipefail
# Submit from the extracted project root; cluster partition/account are site-specific.
bash research/linux/run_h100.sh
