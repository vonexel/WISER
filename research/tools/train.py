"""Controlled Linux run. No test-set model selection; no silent external-data fallback."""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
import platform
import subprocess
from pathlib import Path
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from omegaconf import OmegaConf
from research.tools.common import ROOT, sha256, code_manifest, write_json
from research.tools.dataset import ManifestDataset
from research.tools.evaluation import fit_calibration, reports
from wiser.models.wiser import build_from_cfg, count_parameters
from wiser.models.losses import CombinedLoss
from wiser.models import bissm
from wiser.data.ffpp_dataset import PairedRealFakeSampler
from wiser.training.utils import build_loader
from wiser.training.trainer import Trainer
from wiser.utils.repro import seed_everything
from wiser.utils import setup_logging
from wiser.utils.progress import progress

@torch.inference_mode()
def predict(model, loader, device, output):
    model.eval(); z, y, vids, frames = [], [], [], []
    embedding_sums = {}; embedding_counts = {}
    scope = output.stem.replace('_predictions', '')
    label = 'Cross-dataset testing (Celeb-DF++)' if scope == 'external' else f'Evaluation ({scope})'
    for batch in progress(loader, desc=label):
        out = model(batch['rgb'].to(device), batch['wavelet'].to(device), batch['defocus'].to(device))
        logits = out['logits'].float().cpu().numpy().reshape(-1)
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite logits')
        z.append(logits); y.append(batch['label'].numpy()); vids.extend(batch['video_id']); frames.extend(batch['frame_id'])
        embeddings = out['embeddings'].float().cpu().numpy()
        for video, embedding in zip(batch['video_id'], embeddings):
            if video not in embedding_sums:
                embedding_sums[video] = embedding.astype(np.float64)
                embedding_counts[video] = 1
            else:
                embedding_sums[video] += embedding
                embedding_counts[video] += 1
    data = {'logits': np.concatenate(z), 'labels': np.concatenate(y).astype(np.int8),
            'video_ids': np.asarray(vids, dtype=str), 'frame_ids': np.asarray(frames, dtype=str)}
    expected = [r['frame_id'] for r in loader.dataset.rows]
    if frames != expected or len(set(frames)) != len(frames):
        raise ValueError('Prediction coverage mismatch')
    np.savez_compressed(output, **data)
    embedding_ids = sorted(embedding_sums)
    np.savez_compressed(output.with_name(output.stem.replace('predictions','embeddings')+'.npz'),
                        video_ids=np.asarray(embedding_ids),
                        embeddings=np.stack([embedding_sums[v]/embedding_counts[v] for v in embedding_ids]).astype(np.float32))
    return data

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=[q.stem for q in (ROOT/'research/configs').glob('*.yaml')], required=True)
    p.add_argument('--seed', type=int, required=True, choices=[0, 1, 2])
    p.add_argument('--manifest', default='research/data/frames.csv')
    p.add_argument('--out-root', default='research/runs')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--smoke', action='store_true', help='Two-step functional run, never confirmatory')
    a = p.parse_args()
    if not torch.cuda.is_available() and not a.smoke:
        raise RuntimeError('CUDA required for full experiment')
    manifest = Path(a.manifest).resolve()
    freeze_path = manifest.parent / 'data_freeze.json'
    if not a.smoke:
        if (manifest.parent / 'SYNTHETIC_ONLY.json').exists():
            raise ValueError('Synthetic fixtures are permitted only with --smoke')
        freeze = json.loads(freeze_path.read_text())
        if not freeze['content_hashes_verified'] or freeze['manifest_sha256'] != sha256(manifest):
            raise ValueError('Manifest is not frozen; run audit_data')
        if freeze['videos_sha256'] != sha256(manifest.parent / 'videos.csv'):
            raise ValueError('Video inventory changed after freeze')
    cfg = OmegaConf.load(ROOT / 'research/configs' / f'{a.experiment}.yaml')
    cfg.seed = a.seed; cfg.data.num_workers = a.workers
    cfg.output_root = str(Path(a.out_root).resolve())
    cfg.experiment_id = a.experiment + ('_SMOKE' if a.smoke else '')
    out_dir = Path(cfg.output_root) / cfg.experiment_id / str(a.seed)
    if out_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {out_dir}; choose a fresh out-root')
    out_dir.mkdir(parents=True)
    write_json(out_dir / 'status.json', {'status': 'running', 'scientific_use_allowed': not a.smoke})
    try:
        if a.smoke:
            cfg.training.epochs = 1; cfg.training.sanity_steps = 2; cfg.training.amp_dtype = 'fp32'
            cfg.data.batch_size = 4; cfg.data.eval_batch_size = 4
        setup_logging('INFO'); seed_everything(a.seed, deterministic=True)
        torch.set_num_threads(4)
        # Historical checkpoint keys are LinearAttention2DAsToken, 270,733 detector parameters.
        bissm._BACKEND = 'linear'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg.training.calibration.enabled = True
        datasets = {s: ManifestDataset(manifest, cfg, s) for s in ['train', 'val', 'cal', 'test']}
        datasets['external'] = ManifestDataset(manifest, cfg, 'external', 'celebdfpp')
        if a.smoke:
            # Preserve both classes, rather than truncating to the first method.
            for ds in datasets.values():
                rows = [r for label in ['0', '1'] for r in [x for x in ds.rows if x['label'] == label][:8]]
                ds.rows = rows
                from types import SimpleNamespace
                ds.records = [SimpleNamespace(label=r['method'], video=r['video_id'].split('/',1)[1]) for r in rows]
        gen = torch.Generator().manual_seed(a.seed)
        sampler = PairedRealFakeSampler(datasets['train'], rounds_per_epoch=cfg.data.rounds_per_epoch, generator=gen)
        loaders = {}
        for split, ds in datasets.items():
            loaders[split] = build_loader(ds, batch_size=cfg.data.batch_size if split == 'train' else cfg.data.eval_batch_size,
                                         num_workers=a.workers, shuffle=False, sampler=sampler if split == 'train' else None,
                                         drop_last=split == 'train', seed=a.seed, prefetch_factor=2,
                                         persistent_workers=False)
        model = build_from_cfg(cfg.model)
        if count_parameters(model) != 270733:
            raise ValueError('Fixed WISER parameter count changed')
        loss = CombinedLoss(cfg.loss)
        OmegaConf.save(cfg, out_dir / 'config_resolved.yaml')
        write_json(out_dir / 'passport.json', {'protocol': 'prospective-all-local-v1', 'seed': a.seed,
                   'experiment': a.experiment, 'manifest_sha256': sha256(manifest), 'code': code_manifest(),
                   'backend': 'linear', 'model_parameters': count_parameters(model),
                   'loss_parameters': sum(x.numel() for x in loss.parameters()),
                   'python': platform.python_version(), 'platform': platform.platform(),
                   'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
                   'checkpoint_selection': 'maximum FF++ val EMA frame AUC; no test selection',
                   'data_freeze_sha256': sha256(freeze_path) if freeze_path.exists() else None,
                   'scientific_use_allowed': not a.smoke})
        (out_dir/'pip_freeze.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True))
        trainer = Trainer(model, loss, cfg, loaders['train'], loaders['val'], out_dir, device=device, seed=a.seed)
        initial = loss.mphc.prototypes.detach().clone() if loss.mphc is not None else None
        result = trainer.fit()
        if initial is not None and torch.equal(initial.to(device), loss.mphc.prototypes):
            raise ValueError('Prototypes did not update')
        ckpt = torch.load(out_dir/'ckpt/best.pt', map_location=device, weights_only=True)
        trainer.ema.load_state_dict(ckpt['ema'])
        cal_data = predict(trainer.ema.module, loaders['cal'], device, out_dir/'cal_predictions.npz')
        calibration = fit_calibration(cal_data['logits'], cal_data['labels'], cal_data['video_ids'])
        write_json(out_dir/'calibration.json', calibration)
        for split in ['val', 'test', 'external']:
            arrays = predict(trainer.ema.module, loaders[split], device, out_dir/f'{split}_predictions.npz')
            write_json(out_dir/f'{split}_metrics.json', reports(arrays, calibration))
        write_json(out_dir/'training_summary.json', dataclasses.asdict(result))
        from research.tools.plot_results import plot_run
        plot_run(out_dir)
        write_json(out_dir/'artifacts.sha256.json', {q.relative_to(out_dir).as_posix():sha256(q)
                   for q in out_dir.rglob('*') if q.is_file() and q.name not in ['status.json','artifacts.sha256.json']})
        write_json(out_dir/'status.json', {'status':'complete', 'scientific_use_allowed':not a.smoke,
                                         'checkpoint_sha256':sha256(out_dir/'ckpt/best.pt')})
    except Exception as exc:
        write_json(out_dir/'status.json', {'status':'failed', 'error':repr(exc), 'scientific_use_allowed':False})
        raise

if __name__ == '__main__':
    main()
