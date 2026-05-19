# WISER


## Quick start

```bash
# Base install
uv sync --extra dev

# (Optional, Add Mamba-ssm: H100 support-only) 
uv pip install --no-build-isolation causal-conv1d mamba-ssm
```


```bash
# Preprocess (idempotent, video-level, cached on disk)
uv run python scripts/preprocess_ffpp.py --raw_root dataset/ff_c23 --cache_root preprocessed
uv run python scripts/preprocess_celebdfpp.py --raw_root dataset/celebdfpp --cache_root preprocessed
```


```bash
# Train one experiment
uv run python scripts/train.py experiment=E04_wiser seed=0
```


```bash
# Run the full ablation sweep
bash scripts/run_ablation.sh
```