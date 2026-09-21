"""Recompute the historical E08r result without changing the evidence archive."""
import json
import numpy as np
from scipy.special import expit
from research.tools.common import ARCHIVE, ROOT, sha256, write_json
from research.tools.evaluation import pool, metrics

def main():
    base=ARCHIVE/'experiments/E04';path=base/'predictions_upstream.npz'
    cal=json.loads((base/'calib.json').read_text());out={'source_sha256':sha256(path),'scopes':{}}
    expected=json.loads((ARCHIVE/'experiments/E08r/metrics.json').read_text())
    mapping={'auc':'auc','RR':'real_recall','FR':'fake_recall','bAcc':'balanced_acc','Brier':'brier','ECE':'ece'}
    errors=[]
    with np.load(path,allow_pickle=False) as n:
        for scope in ['indomain','crossdomain']:
            z=n[scope+'_logits'].astype(float);y=n[scope+'_labels'];v=n[scope+'_video_ids']
            out['scopes'][scope]={}
            for name,scores,threshold in [('uncalibrated',expit(z),.5),
                                          ('calibrated',expit(z/cal['temperature']-cal['prior_bias']),cal['threshold'])]:
                ids,vy,vp=pool(y,scores,v)
                result={'frame':metrics(y,scores,threshold),'video':metrics(vy,vp,threshold)}
                out['scopes'][scope][name]=result
                # Independent float64 implementation; comparison targets inspected archive layout.
                published=expected['scopes'][scope]['branches'][name]
                for level in ['frame','video']:
                    for k,old in mapping.items():
                        # Archive stores video aggregations in a nested mapping.
                        target=published.get(level)
                        if level=='video' and target is not None:target=target['mean']
                        if target is None and level=='video':target=published.get('video_mean')
                        if target is not None and old in target:
                            errors.append(abs(result[level][k]-target[old]))
    out['maximum_checked_metric_error']=max(errors) if errors else None
    if not errors or max(errors)>1e-6:
        raise ValueError('Historical recalculation differs from archived metrics')
    out['note']='Independent float64 recalculation; historical calibration not refit; no new efficacy result.'
    write_json(ROOT/'research/review/historical_recalculation.json',out)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
