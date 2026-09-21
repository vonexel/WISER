"""Run on target GPU before launching full training; never infer fit from parameter count."""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
import torch
from omegaconf import OmegaConf
from research.tools.common import ROOT, write_json
from wiser.models import bissm
from wiser.models.wiser import build_from_cfg, count_parameters
from wiser.models.losses import CombinedLoss
from wiser.training.optim import build_optimizer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size',type=int,default=96)
    p.add_argument('--output',default='research/results/preflight.json')
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable; do not silently change precision')
    bissm._BACKEND='linear';torch.set_num_threads(4)
    cfg=OmegaConf.load(ROOT/'research/configs/E04.yaml')
    model=build_from_cfg(cfg.model).cuda();loss=CombinedLoss(cfg.loss).cuda();opt=build_optimizer(model,cfg.training.optim,loss)
    assert count_parameters(model)==270733
    shape=a.batch_size
    rgb=torch.randn(shape,3,256,256,device='cuda');wav=torch.randn(shape,12,128,128,device='cuda');df=torch.randn(shape,1,128,128,device='cuda')
    labels=torch.arange(shape,device='cuda')%2
    torch.cuda.reset_peak_memory_stats();times=[]
    for i in range(6):
        opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(rgb,wav,df,return_features=True);out['method_label']=labels
            value,_=loss(out,labels)
        if not torch.isfinite(value):raise ValueError('Nonfinite smoke loss')
        value.backward();opt.step();torch.cuda.synchronize()
        if i>=2:times.append(time.perf_counter()-start)
    result={'synthetic_only':True,'gpu':torch.cuda.get_device_name(0),'batch_size':shape,
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
            'mean_train_step_seconds':sum(times)/len(times),'model_parameters':count_parameters(model),
            'backend':'linear','torch':str(torch.__version__),'cuda':torch.version.cuda,
            'python':platform.python_version(),
            'note':'Synthetic device-only timing excludes crop I/O/augmentation, EMA and full validation. Measure a real smoke too.'}
    write_json(a.output,result);print(json.dumps(result,indent=2))
    (Path(a.output).parent/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))

if __name__=='__main__':main()
