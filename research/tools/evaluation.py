"""Independent metrics: fake=1, video mean probability, fixed 15-bin probability ECE."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve

def metrics(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Invalid prediction shapes')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Need finite probabilities and both binary classes')
    rr = float(np.mean(p[y == 0] < threshold)); fr = float(np.mean(p[y == 1] >= threshold))
    bins = np.minimum((p * 15).astype(int), 14)
    ece = sum(float(np.abs(p[bins == b].mean() - y[bins == b].mean())) * np.mean(bins == b)
              for b in range(15) if np.any(bins == b))
    return {'auc': float(roc_auc_score(y, p)), 'RR': rr, 'FR': fr,
            'bAcc': (rr + fr) / 2, 'Brier': float(np.mean((p - y)**2)), 'ECE': float(ece),
            'n': len(y), 'real': int(np.sum(y == 0)), 'fake': int(np.sum(y == 1)), 'threshold': float(threshold)}

def pool(y, p, video_ids):
    ids, first, inverse, counts = np.unique(video_ids, return_index=True, return_inverse=True, return_counts=True)
    labels = y[first]
    if not np.array_equal(labels[inverse], y):
        raise ValueError('Conflicting labels within a video')
    return ids, labels, np.bincount(inverse, weights=p) / counts

def threshold_on_source(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    idx = np.flatnonzero(valid)[np.argmax((tpr - fpr)[valid])]
    return float(thresholds[idx])

def fit_calibration(logits, labels, video_ids):
    # Fit video-balanced frame NLL on the separate source calibration partition.
    _, inverse, counts = np.unique(video_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]; weights /= weights.sum()
    def objective(x):
        z = logits.astype(np.float64) / np.exp(x[0]) - x[1]
        return float(np.sum(weights * (np.logaddexp(0, z) - labels * z)))
    fit = minimize(objective, [0., 0.], method='L-BFGS-B', bounds=[(-4., 4.), (-10., 10.)])
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError('Calibration optimization failed: ' + str(fit.message))
    t, b = float(np.exp(fit.x[0])), float(fit.x[1])
    _, y, raw = pool(labels, expit(logits), video_ids)
    _, _, calibrated = pool(labels, expit(logits / t - b), video_ids)
    return {'temperature': t, 'bias': b, 'raw_threshold': threshold_on_source(y, raw),
            'calibrated_threshold': threshold_on_source(y, calibrated),
            'fit': 'video-balanced frame NLL on FF++ cal; thresholds maximize video bAcc on cal',
            'nll': float(fit.fun)}

def reports(arrays, calibration):
    z, y, ids = arrays['logits'].astype(np.float64), arrays['labels'], arrays['video_ids']
    raw = expit(z.astype(np.float64)); cal = expit(z / calibration['temperature'] - calibration['bias'])
    result = {}
    for branch, scores, threshold in [('raw_fixed', raw, .5),
                                      ('raw_source_threshold', raw, calibration['raw_threshold']),
                                      ('calibrated', cal, calibration['calibrated_threshold'])]:
        vids, vy, vp = pool(y, scores, ids)
        result[branch] = {'frame': metrics(y, scores, threshold), 'video': metrics(vy, vp, threshold)}
    if abs(result['raw_fixed']['frame']['auc'] - result['calibrated']['frame']['auc']) > 1e-8:
        raise ValueError('Monotone calibration changed frame AUC beyond tolerance')
    vids, vy, vp = pool(y, raw, ids)
    methods = np.array([v.split('/')[0] for v in vids])
    method_results = {}
    for method in sorted(set(methods[vy == 1])):
        mask = (vy == 0) | (methods == method)
        method_results[method] = metrics(vy[mask], vp[mask])
    result['per_method_raw_video'] = method_results
    result['macro_method_auc'] = float(np.mean([r['auc'] for r in method_results.values()]))
    return result
