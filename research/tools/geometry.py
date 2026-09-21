"""Report prototype usage and real/fake margins from saved video embeddings."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from research.tools.common import write_json
from research.tools.evaluation import pool

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
    ckpt=torch.load(root/'ckpt/best.pt',map_location='cpu',weights_only=True)
    key='mphc.prototypes'
    if key not in ckpt['loss_fn']:raise ValueError('No prototypes: expected for E05b')
    proto=ckpt['loss_fn'][key].float().numpy();proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
    k=len(proto)//2;report={}
    for scope in ['val','test','external']:
        with np.load(root/f'{scope}_embeddings.npz',allow_pickle=False) as n:ids,z=n['video_ids'],n['embeddings']
        with np.load(root/f'{scope}_predictions.npz',allow_pickle=False) as n:
            vids,y,_=pool(n['labels'],n['logits'],n['video_ids'])
        if not np.array_equal(ids,vids):raise ValueError('Embedding/prediction video mismatch')
        z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);sim=z@proto.T
        real_margin=sim[:,:k].max(1)-sim[:,k:].max(1)
        report[scope]={}
        for label in [0,1]:
            m=y==label;own=sim[m,label*k:(label+1)*k];usage=np.bincount(own.argmax(1),minlength=k)
            report[scope][str(label)]={'videos':int(m.sum()),'prototype_assignment_counts':usage.tolist(),
                'real_minus_fake_margin_quantiles':np.quantile(real_margin[m],[.05,.25,.5,.75,.95]).tolist(),
                'mean_cosine_to_class_centroid':float(np.linalg.norm(z[m].mean(0))),
                'fraction_positive_real_margin':float((real_margin[m]>0).mean())}
    write_json(root/'geometry_diagnostics.json',{'unit':'mean frame embedding per video, then L2 normalization',
        'caution':'Diagnostic evidence only; improved geometry does not prove classifier generalization.',
        'scopes':report})
    from research.tools.plot_results import plot_geometry
    plot_geometry(root/'geometry_diagnostics.json')

if __name__=='__main__':main()
