"""Paired, class-stratified video bootstrap with paired training seeds.

Do not treat the three seeds times N videos as independent observations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from research.tools.common import sha256, write_json
from research.tools.evaluation import pool, metrics, reports
from wiser.utils.progress import progress

def weighted_auc(y, p, weights):
    order = np.argsort(p, kind='stable'); s, y, w = p[order], y[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    pos = np.add.reduceat(w * y, starts); neg = np.add.reduceat(w * (1-y), starts)
    return float(np.sum(pos * (np.cumsum(neg) - .5 * neg)) / (pos.sum() * neg.sum()))

def paired_bootstrap(y, a, b, replicates=2000, seed=42):
    # Inputs [seed,video]; same videos sampled jointly for every seed and arm.
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == label) for label in [0, 1]]
    draws = np.empty((replicates, 2)); conditional = np.empty_like(draws)
    per_seed = [[metrics(y, x)['auc']-metrics(y, z)['auc'], metrics(y, x)['RR']-metrics(y, z)['RR']]
                for x, z in zip(a, b)]
    for i in progress(range(replicates), desc='Paired video bootstrap', unit='replicate'):
        ids = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        w = np.bincount(ids, minlength=len(y))
        differences = np.array([[weighted_auc(y, x, w)-weighted_auc(y, z, w),
                                 np.sum(w[y==0]*((x[y==0]<.5).astype(float)-(z[y==0]<.5)))/w[y==0].sum()]
                                for x,z in zip(a,b)])
        conditional[i] = differences.mean(axis=0)
        draws[i] = differences[rng.integers(0, len(a), size=len(a))].mean(axis=0)
    # Three contrasts x two co-primary endpoints = six comparisons.
    q = [100 * .05 / 12, 100 * (1-.05/12)]
    return {'seed_differences':per_seed, 'mean_difference':np.mean(per_seed,axis=0).tolist(),
            'seed_sd':np.std(per_seed,axis=0,ddof=1).tolist(),
            'endpoints':['video_auc','video_RR_at_0.5'],
            'hierarchical_95_ci':np.percentile(draws,[2.5,97.5],axis=0).T.tolist(),
            'hierarchical_familywise_95_ci':np.percentile(draws,q,axis=0).T.tolist(),
            'conditional_video_95_ci':np.percentile(conditional,[2.5,97.5],axis=0).T.tolist(),
            'replicates':replicates, 'bootstrap_seed':seed,
            'limitation':'Only 3 training seeds: hierarchical uncertainty is imprecise. Conditional CI fixes the three trained models.'}

def load_run(path):
    status = json.loads((path/'status.json').read_text())
    if status['status'] != 'complete' or not status['scientific_use_allowed']:
        raise ValueError(f'Not a completed scientific run: {path}')
    passport = json.loads((path/'passport.json').read_text())
    for filename,h in json.loads((path/'artifacts.sha256.json').read_text()).items():
        if sha256(path/filename) != h:
            raise ValueError(f'Artifact hash mismatch: {path/filename}')
    if sha256(path/'ckpt/best.pt') != status['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    with np.load(path/'external_predictions.npz',allow_pickle=False) as n:
        arrays = {k:n[k] for k in n.files}
    calibration = json.loads((path/'calibration.json').read_text())
    recalculated = reports(arrays, calibration)
    saved = json.loads((path/'external_metrics.json').read_text())
    for branch in ['raw_fixed','raw_source_threshold','calibrated']:
        for level in ['frame','video']:
            for metric in ['auc','RR','FR','bAcc','Brier','ECE']:
                if abs(saved[branch][level][metric]-recalculated[branch][level][metric]) > 1e-10:
                    raise ValueError('Independent metric mismatch')
    ids,y,scores = pool(arrays['labels'],expit(arrays['logits'].astype(float)),arrays['video_ids'])
    return ids,y,scores,passport,arrays['frame_ids']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',default='research/runs')
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--output',default='research/results/comparisons.json')
    a=p.parse_args(); root=Path(a.runs); loaded={}; reference=None
    summaries={}
    from omegaconf import OmegaConf
    from research.tools.make_configs import VARIANTS
    baseline_config = None
    for name in ['E04','E05','E05b','E06']:
        loaded[name]=[]
        for seed in [0,1,2]:
            cfg = OmegaConf.load(root/name/str(seed)/'config_resolved.yaml')
            cfg.seed=0;cfg.experiment_id='E04'
            for key in VARIANTS[name]:
                base=OmegaConf.load(root/'E04/0/config_resolved.yaml')
                OmegaConf.update(cfg,key,OmegaConf.select(base,key))
            normalized=OmegaConf.to_container(cfg,resolve=True)
            if baseline_config is None:baseline_config=normalized
            if normalized!=baseline_config:raise ValueError('Non-ablation config difference between runs')
            ids,y,scores,passport,frames=load_run(root/name/str(seed))
            fingerprint=(passport['manifest_sha256'],passport['code'])
            if reference is None:
                reference=(ids,y,frames,fingerprint)
            if not all(np.array_equal(x,z) for x,z in zip(reference[:3],(ids,y,frames))) or fingerprint!=reference[3]:
                raise ValueError('Runs differ in source, manifest, prediction coverage, or labels')
            loaded[name].append(scores)
        loaded[name]=np.stack(loaded[name])
        summary=[metrics(y,s) for s in loaded[name]]
        summaries[name]={k:{'mean':float(np.mean([v[k] for v in summary])),
                             'sd':float(np.std([v[k] for v in summary],ddof=1))}
                         for k in ['auc','RR','FR','bAcc','Brier','ECE']}
    output={'direction':'E04 minus comparator; positive AUC/RR favors E04',
            'primary_threshold':.5,'summaries':summaries,'contrasts':{}}
    for name in ['E05','E05b','E06']:
        result=paired_bootstrap(y,loaded['E04'],loaded[name],a.bootstrap)
        result['both_endpoints_positive_familywise']=all(v[0]>0 for v in result['hierarchical_familywise_95_ci'])
        output['contrasts']['E04-'+name]=result
    write_json(a.output,output)
    from research.tools.plot_results import plot_comparisons
    plot_comparisons(a.output)
    print(a.output)

if __name__=='__main__':
    main()
