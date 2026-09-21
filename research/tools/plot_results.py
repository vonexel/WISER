"""Headless SVG plots from saved results. Never fits models or chooses thresholds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve
from research.tools.common import sha256, write_json
from research.tools.evaluation import pool

plt.rcParams.update({'svg.fonttype':'none', 'svg.hashsalt':'wiser-paper2', 'font.size':10})

def save(fig, out, name, smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if smoke:
        fig.suptitle('SMOKE TEST - NOT SCIENTIFIC EVIDENCE',color='darkred',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.95))
    else:
        fig.tight_layout()
    fig.savefig(out/f'{name}.svg',format='svg',metadata={'Date':None})
    plt.close(fig)

def plot_run(run, output=None):
    run=Path(run);out=Path(output) if output else run/'figures'
    passport=json.loads((run/'passport.json').read_text())
    smoke=not passport.get('scientific_use_allowed',False)
    cal=json.loads((run/'calibration.json').read_text());curve_data={};sources={}
    history_path=run/'epoch_history.json'
    if history_path.exists():
        h=json.loads(history_path.read_text());epochs=[v['epoch'] for v in h]
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(epochs,[v['train_loss'] for v in h],marker='o',markersize=3,label='Train')
        axes[0].plot(epochs,[v['val_loss'] for v in h],marker='o',markersize=3,label='Validation')
        axes[0].set(xlabel='Epoch (1-based)',ylabel='Loss',title='Optimization');axes[0].legend()
        axes[1].plot(epochs,[v['val_auc_ema'] for v in h],marker='o',markersize=3)
        axes[1].set(xlabel='Epoch (1-based)',ylabel='Frame AUC',title='Source validation, EMA',ylim=(0,1))
        from matplotlib.ticker import MaxNLocator
        for ax in axes:
            if len(epochs)==1:ax.set_xticks(epochs)
            else:ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        save(fig,out,'training_curves',smoke);sources['epoch_history.json']=sha256(history_path)
    for scope in ['val','test','external']:
        pred=run/f'{scope}_predictions.npz';metrics_path=run/f'{scope}_metrics.json'
        with np.load(pred,allow_pickle=False) as n:
            z=n['logits'].astype(float);y=n['labels'];ids=n['video_ids']
        _,vy,raw=pool(y,expit(z),ids)
        _,_,calibrated=pool(y,expit(z/cal['temperature']-cal['bias']),ids)
        values=json.loads(metrics_path.read_text())
        sources[pred.name]=sha256(pred);sources[metrics_path.name]=sha256(metrics_path)
        name='Celeb-DF++ external' if scope=='external' else f'FF++ {scope}'
        fig,ax=plt.subplots(figsize=(5.5,4.5));data={}
        for branch,score,metric_key in [('Raw',raw,'raw_fixed'),('Calibrated',calibrated,'calibrated')]:
            fpr,tpr,_=roc_curve(vy,score)
            auc=values[metric_key]['video']['auc']
            ax.plot(fpr,tpr,label=f'{branch} AUC={auc:.4f}')
            data[branch]={'fpr':fpr.tolist(),'tpr':tpr.tolist()}
        ax.plot([0,1],[0,1],'--',color='gray');ax.legend(loc='lower right')
        ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{name}: video ROC')
        save(fig,out,f'{scope}_roc',smoke)
        fig,ax=plt.subplots(figsize=(5.5,4.5))
        for branch,score in [('Raw',raw),('Calibrated',calibrated)]:
            bins=np.minimum((score*15).astype(int),14);points=[]
            for i in range(15):
                m=bins==i
                if m.any():points.append([float(score[m].mean()),float(vy[m].mean()),int(m.sum())])
            ax.plot([v[0] for v in points],[v[1] for v in points],marker='o',label=branch)
            data[branch]['reliability_bins_mean_probability_fraction_fake_count']=points
        ax.plot([0,1],[0,1],'--',color='gray');ax.legend()
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted fake probability',ylabel='Observed fake fraction',
               title=f'{name}: video calibration (15 bins)')
        save(fig,out,f'{scope}_calibration',smoke)
        method_metrics=values['per_method_raw_video'];methods=list(method_metrics)
        fig,ax=plt.subplots(figsize=(8,max(3,len(methods)*.28+1.5)))
        ax.barh(methods,[method_metrics[m]['auc'] for m in methods]);ax.axvline(.5,color='gray',linestyle='--')
        ax.set(xlim=(0,1),xlabel='Video AUC, raw scores; same real reference pool',title=f'{name}: manipulation methods')
        ax.invert_yaxis();save(fig,out,f'{scope}_per_method_auc',smoke)
        curve_data[scope]=data
    sources['calibration.json']=sha256(run/'calibration.json')
    write_json(out/'plot_data.json',{'scientific_use_allowed':not smoke,'curves':curve_data})
    write_json(out/'sources.json',{'source_files_sha256':sources,'pooling':'mean frame probability per video',
                                  'matplotlib':matplotlib.__version__,'scientific_use_allowed':not smoke})
    return out

def plot_comparisons(path, output=None):
    path=Path(path);data=json.loads(path.read_text());out=Path(output) if output else path.parent/'figures'
    smoke=not data.get('scientific_use_allowed', True)
    names=list(data['contrasts']);fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,(ax,title) in enumerate(zip(axes,['Video AUC difference','RR difference at threshold 0.5'])):
        for i,name in enumerate(names):
            r=data['contrasts'][name];low,high=np.array(r['hierarchical_familywise_95_ci'][j])*100
            ax.hlines(i,low,high,color='#245a78',linewidth=2)
            ax.plot(r['mean_difference'][j]*100,i,'o',color='#245a78')
        ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(range(len(names)),names)
        ax.set(xlabel='E04 minus comparator (percentage points)',title=title)
    fig.suptitle('Paired seed/video bootstrap; six-comparison adjusted intervals')
    save(fig,out,'hypothesis_contrasts',smoke)
    fig,axes=plt.subplots(2,3,figsize=(12,7));names=list(data['summaries'])
    for ax,key in zip(axes.flat,['auc','RR','FR','bAcc','Brier','ECE']):
        mean=[data['summaries'][n][key]['mean'] for n in names];sd=[data['summaries'][n][key]['sd'] for n in names]
        ax.bar(names,mean,yerr=sd,capsize=3,color='#245a78');ax.set(title=key,ylabel='Mean +/- seed SD')
        ax.tick_params(axis='x',rotation=20)
    save(fig,out,'ablation_metrics',smoke)
    write_json(out/'comparison_sources.json',{'source':path.name,'sha256':sha256(path),
                                            'error_bars':'Contrast: adjusted bootstrap CI. Metrics: seed SD, not CI.'})
    return out

def plot_geometry(path, output=None):
    path=Path(path);data=json.loads(path.read_text());out=Path(output) if output else path.parent/'figures'
    passport=path.parent/'passport.json'
    smoke=passport.exists() and not json.loads(passport.read_text()).get('scientific_use_allowed',False)
    scopes=list(data['scopes']);fig,axes=plt.subplots(2,len(scopes),figsize=(12,6),squeeze=False)
    for col,scope in enumerate(scopes):
        for label in [0,1]:
            r=data['scopes'][scope][str(label)];ax=axes[label,col]
            counts=r['prototype_assignment_counts'];ax.bar(range(1,len(counts)+1),counts,color='#245a78')
            ax.set(title=f'{scope}: {"real" if label==0 else "fake"}',xlabel='Own-class prototype',ylabel='Videos')
    save(fig,out,'prototype_usage',smoke)
    fig,ax=plt.subplots(figsize=(8,4));labels=[]
    for scope in scopes:
        for label in [0,1]:
            q=data['scopes'][scope][str(label)]['real_minus_fake_margin_quantiles'];i=len(labels)
            labels.append(f'{scope}: {"real" if label==0 else "fake"}')
            ax.hlines(i,q[0],q[4],color='#245a78');ax.hlines(i,q[1],q[3],color='#245a78',linewidth=5);ax.plot(q[2],i,'ko')
    ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(range(len(labels)),labels)
    ax.set(xlabel='Maximum real-prototype similarity minus maximum fake-prototype similarity',
           title='Video embedding margins: 5/25/50/75/95 percentiles (not confidence intervals)')
    save(fig,out,'prototype_margins',smoke)
    write_json(out/'geometry_sources.json',{'source':path.name,'sha256':sha256(path),'scientific_use_allowed':not smoke})
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run');g.add_argument('--comparisons');g.add_argument('--geometry')
    p.add_argument('--output',required=True,help='Use a separate folder to preserve a completed run’s checksum manifest')
    a=p.parse_args()
    if a.run:result=plot_run(a.run,a.output)
    elif a.comparisons:result=plot_comparisons(a.comparisons,a.output)
    else:result=plot_geometry(a.geometry,a.output)
    print(result)

if __name__=='__main__':main()
