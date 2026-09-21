"""Five-group, paired-source spectral mechanism analysis on frozen real crops."""
from __future__ import annotations
import argparse
import collections
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.stats import wasserstein_distance
from research.tools.common import sha256, write_json
from wiser.utils.progress import progress
from wiser.data.sbi import self_blend, SBIParams
from wiser.data.awb_sbi import awb_self_blend, AWBSBIParams, _haar_dwt2d_chw

def spectrum(rgb, bins=64):
    gray = np.tensordot(rgb.astype(float)/255., [.299,.587,.114], axes=([-1],[0]))
    centered=gray-gray.mean(); window=np.outer(np.hanning(gray.shape[0]),np.hanning(gray.shape[1]))
    fft=np.fft.fftshift(np.fft.fft2(centered*window))
    yy,xx=np.indices(gray.shape); radius=np.hypot(yy-gray.shape[0]//2,xx-gray.shape[1]//2)/(min(gray.shape)/2)
    valid=(radius<1); indices=np.minimum((radius*bins).astype(int),bins-1)
    power=np.abs(fft)**2/(gray.size**2)
    normalized_fft=fft/max(gray.std(),1e-8)
    profile=np.array([np.log1p(np.abs(normalized_fft))[valid & (indices==b)].mean() for b in range(bins)])
    energy=np.array([power[(radius>=lo)&(radius<hi)].sum() for lo,hi in [(0,1/3),(1/3,2/3),(2/3,1)]])
    return profile,energy

def distances(a,b,band):
    # L1 preserves amplitude; Wasserstein measures normalized mass on radius.
    x,y=a[band],b[band]; r=np.linspace(0,1,len(a))[band]
    wx=np.maximum(x,0)+1e-12; wy=np.maximum(y,0)+1e-12
    return np.array([np.mean(np.abs(x-y)),wasserstein_distance(r,r,wx,wy)])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='research/data/frames.csv')
    p.add_argument('--output',default='research/results/spectral')
    p.add_argument('--videos-per-method',type=int,default=20)
    p.add_argument('--real-videos',type=int,default=140)
    p.add_argument('--frames-per-video',type=int,default=5)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); manifest=Path(a.manifest); root=manifest.parent; out=Path(a.output)
    if a.smoke:
        if not (root/'SYNTHETIC_ONLY.json').exists():raise ValueError('Spectral smoke requires synthetic fixture')
    else:
        if (root/'SYNTHETIC_ONLY.json').exists():raise ValueError('Synthetic input cannot be scientific evidence')
        freeze=json.loads((root/'data_freeze.json').read_text())
        if freeze['manifest_sha256']!=sha256(manifest):raise ValueError('Unfrozen manifest')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    methods=collections.defaultdict(lambda:collections.defaultdict(list))
    with manifest.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['dataset']=='ffpp' and r['split'] not in ['val','cal']:continue
            if r['dataset']=='celebdfpp' and r['label']=='0':continue
            methods[(r['dataset'],r['method'])][r['video_id']].append(r)
    rng=np.random.default_rng(42); profiles=collections.defaultdict(list); energies=collections.defaultdict(list)
    selected=[]; ll=[]; keys=collections.defaultdict(list)
    for (ds,method),videos in progress(sorted(methods.items()), desc='Spectral groups', unit='method'):
        vids=sorted(videos); n=a.real_videos if method=='real' else a.videos_per_method
        vids=sorted(rng.choice(vids,min(n,len(vids)),replace=False).tolist())
        for vid in vids:
            rows=videos[vid]; idx=sorted(rng.choice(len(rows),min(a.frames_per_video,len(rows)),replace=False))
            per=collections.defaultdict(list); eng=collections.defaultdict(list); llv=[]
            for i in idx:
                r=rows[i]; selected.append(r)
                rgb=cv2.cvtColor(cv2.imread(str(root/r['crop'])),cv2.COLOR_BGR2RGB)
                group='Real' if method=='real' else ('Fake-FF++' if ds=='ffpp' else 'Fake-Celeb-DF++')
                images={group:rgb}
                if method=='real':
                    lm=np.load(root/r['landmarks'],allow_pickle=False)
                    seed=int.from_bytes(__import__('hashlib').sha256(r['frame_id'].encode()).digest()[:8],'little')
                    images['SBI']=self_blend(rgb,SBIParams(p=1),np.random.default_rng(seed),landmarks=lm)[0]
                    images['AWB-SBI']=awb_self_blend(rgb,AWBSBIParams(p=1),np.random.default_rng(seed),landmarks=lm)[0]
                    original_ll=_haar_dwt2d_chw(rgb.transpose(2,0,1).astype(float)/255)[0]
                    perturb=[]
                    for name in ['SBI','AWB-SBI']:
                        new_ll=_haar_dwt2d_chw(images[name].transpose(2,0,1).astype(float)/255)[0]
                        perturb.append(float(np.mean((new_ll-original_ll)**2)))
                    llv.append(perturb)
                for name,img in images.items():
                    profile,energy=spectrum(img);per[name].append(profile);eng[name].append(energy)
            for name in per:
                profiles[name].append(np.mean(per[name],axis=0));energies[name].append(np.mean(eng[name],axis=0));keys[name].append(vid)
            if llv:ll.append(np.mean(llv,axis=0))
    profiles={k:np.stack(v) for k,v in profiles.items()}; energies={k:np.stack(v) for k,v in energies.items()}
    required={'Real','SBI','AWB-SBI','Fake-FF++','Fake-Celeb-DF++'}
    if set(profiles)!=required:raise ValueError('Missing spectral group')
    np.savez_compressed(out/'per_video.npz',**{k:v for k,v in profiles.items()},
                        **{k+'__energy':v for k,v in energies.items()},ll_mse=np.array(ll),
                        **{k+'__videos':np.array(v) for k,v in keys.items()})
    bands={'low':slice(0,21),'mid':slice(21,43),'high':slice(43,64),'all':slice(None)}
    boot_means={k:[] for k in profiles}; diffs=collections.defaultdict(list); energy_delta=[];ll_delta=[]
    rng=np.random.default_rng(43)
    for _ in progress(range(a.bootstrap), desc='Spectral bootstrap', unit='replicate'):
        ix=rng.integers(0,len(profiles['Real']),len(profiles['Real']))
        means={k:profiles[k][ix].mean(0) for k in ['Real','SBI','AWB-SBI']}
        for k in ['Fake-FF++','Fake-Celeb-DF++']:
            # Equal-video empirical target; method composition fixed by prespecified sampling.
            j=rng.integers(0,len(profiles[k]),len(profiles[k]));means[k]=profiles[k][j].mean(0)
        for k in means:boot_means[k].append(means[k])
        for target in ['Fake-FF++','Fake-Celeb-DF++']:
            for name,band in bands.items():
                diffs[target+'/'+name].append(distances(means['AWB-SBI'],means[target],band)-distances(means['SBI'],means[target],band))
        energy_delta.append((energies['AWB-SBI'][ix]-energies['SBI'][ix]).mean(0))
        ll_delta.append((np.array(ll)[ix,1]-np.array(ll)[ix,0]).mean())
    result={'manifest_sha256':sha256(manifest),'bootstrap_unit':'video; Real/SBI/AWB paired by source',
            'scientific_use_allowed':not a.smoke,
            'scope':'mechanism analysis; not a classification efficacy test',
            'counts':{k:len(v) for k,v in profiles.items()},'distances_AWB_minus_SBI':{},
            'energy_delta_low_mid_high':{'estimate':(energies['AWB-SBI']-energies['SBI']).mean(0).tolist(),
                                         'ci95':np.percentile(energy_delta,[2.5,97.5],axis=0).T.tolist()},
            'LL_MSE_delta_AWB_minus_SBI':{'estimate':float(np.mean(np.array(ll)[:,1]-np.array(ll)[:,0])),
                                          'ci95':np.percentile(ll_delta,[2.5,97.5]).tolist()},
            'distance_columns':['mean_absolute_log_profile_difference','radial_Wasserstein'],
            'energy_definition':'unnormalized centered grayscale windowed FFT power; excludes corner radii >= Nyquist',
            'LL_definition':'post-clipping crop LL mean squared perturbation versus real; no claim of exact LL preservation'}
    for key,vals in diffs.items():
        target,band=key.split('/');means={k:v.mean(0) for k,v in profiles.items()}
        estimate=distances(means['AWB-SBI'],means[target],bands[band])-distances(means['SBI'],means[target],bands[band])
        result['distances_AWB_minus_SBI'][key]={'estimate':estimate.tolist(),'ci95':np.percentile(vals,[2.5,97.5],axis=0).T.tolist()}
    write_json(out/'results.json',result)
    with (out/'selected_frames.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for k in sorted(profiles):
        mean=profiles[k].mean(0);ci=np.percentile(boot_means[k],[2.5,97.5],axis=0)
        ax.plot(np.linspace(0,1,64),mean,label=k);ax.fill_between(np.linspace(0,1,64),ci[0],ci[1],alpha=.15)
    ax.set(xlabel='Radius / axial Nyquist',ylabel='Standardized log amplitude');ax.legend();fig.tight_layout()
    fig.savefig(out/'spectral_profiles.png',dpi=180);fig.savefig(out/'spectral_profiles.svg');plt.close(fig)
    print(out)

if __name__=='__main__':
    main()
