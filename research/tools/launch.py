import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from research.tools.common import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='0')
    p.add_argument('--tier',choices=['confirmatory','mechanism'],default='confirmatory')
    p.add_argument('--manifest',default='research/data/frames.csv')
    p.add_argument('--out-root',default='research/runs')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();gpus=a.gpus.split(',')
    plan=[r for r in json.loads((ROOT/'research/configs/matrix.json').read_text()) if r['tier']==a.tier]
    def worker(gpu,jobs):
        for r in jobs:
            out=Path(a.out_root)/r['experiment']/str(r['seed'])
            if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['status']=='complete':
                print('Already complete:',out,flush=True);continue
            cmd=[sys.executable,'-m','research.tools.train','--experiment',r['experiment'],'--seed',str(r['seed']),
                 '--manifest',a.manifest,'--out-root',a.out_root,'--workers',str(a.workers)]
            print('GPU',gpu,subprocess.list2cmdline(cmd),flush=True)
            if a.dry_run:continue
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONHASHSEED':str(r['seed']),
                 'CUBLAS_WORKSPACE_CONFIG':':4096:8','NO_ALBUMENTATIONS_UPDATE':'1'}
            logroot=Path(a.out_root)/'logs';logroot.mkdir(parents=True,exist_ok=True)
            logpath=logroot/f'{r["experiment"]}_{r["seed"]}.log'
            with logpath.open('x',encoding='utf-8') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        jobs=[executor.submit(worker,gpu,plan[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for job in jobs:job.result()

if __name__=='__main__':main()
