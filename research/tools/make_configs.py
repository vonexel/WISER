"""Generate immutable one-factor variants from the archived resolved E04 config."""
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from research.tools.common import ARCHIVE, ROOT, write_json

VARIANTS = {
    'E04': {},
    'E05': {'loss.mphc.rp_enabled': False},
    'E05b': {'loss.mphc.enabled': False},
    'E06': {'augment.use_awb_sbi': False},
    'F00_SBI_no_contrastive': {'augment.use_awb_sbi': False, 'loss.mphc.enabled': False},
    'M01_no_gradient': {'augment.sbi.energy_boost': 1.0},
    'M02_no_LL': {'augment.sbi.alpha_ll': 0.0},
    'M03_one_prototype': {'loss.mphc.K': 1},
    'M04_no_hard_real_weight': {'loss.mphc.hard_real_weight': 1.0},
}

def main():
    cfg = OmegaConf.load(ARCHIVE / 'experiments/E04/config_resolved.yaml')
    cfg.output_root = 'research/runs'
    cfg.data.paths.raw_root = 'datasets'; cfg.data.paths.cache_root = 'research/data'
    cfg.training.calibration.video_pool_calibrated = 'mean'
    cfg.training.calibration.video_pool_uncalibrated = 'mean'
    # Process memory/throughput choices kept identical across the complete matrix.
    cfg.data.num_workers = 8; cfg.data.prefetch_factor = 2
    out = ROOT / 'research/configs'; out.mkdir(exist_ok=True)
    plan = []
    for name, edits in VARIANTS.items():
        v = deepcopy(cfg); v.experiment_id = name; v.tags = ['prospective-v1']
        for key, value in edits.items():
            OmegaConf.update(v, key, value)
        OmegaConf.save(v, out / f'{name}.yaml')
        for seed in [0, 1, 2]:
            plan.append({'experiment': name, 'seed': seed,
                         'tier': 'confirmatory' if name in ['E04', 'E05', 'E05b', 'E06'] else 'mechanism',
                         'changes_from_E04': edits})
    write_json(out / 'matrix.json', plan)
    print(f'{len(plan)} planned runs (12 confirmatory; 15 optional mechanism/factorial)')

if __name__ == '__main__':
    main()
