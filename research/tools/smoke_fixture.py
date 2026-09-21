"""Make tiny SYNTHETIC input fixtures for software checks only."""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from research.tools.common import sha256, write_json

def make_fixture(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True); rows=[]; videos=[]
    rng=np.random.default_rng(44)
    for sidx,split in enumerate(['train','val','cal','test','external']):
        ds='celebdfpp' if split=='external' else 'ffpp'
        for vid in range(4):
            source=f'{sidx*10+vid:03d}'
            for label in [0,1]:
                method=('Celeb-real' if label==0 else 'FS-Test') if ds=='celebdfpp' else ('real' if label==0 else 'Deepfakes')
                video=source if not label else source+'_'+source
                info={'dataset':ds,'method':method,'video_id':method+'/'+video,'label':str(label),
                      'split':split,'source_ids':source if ds=='ffpp' else '',
                      'group_id':source,'path':'SYNTHETIC','bytes':0}
                videos.append(info)
                for frame in [0,10]:
                    rgb=rng.integers(0,256,(256,256,3),dtype=np.uint8)
                    rel=Path('crops')/ds/method/video/f'frame_{frame:06d}.jpg'
                    path=out/rel;path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),rgb)
                    lm=''
                    if label==0 and ds=='ffpp':
                        lm=rel.with_suffix('.landmarks.npy').as_posix()
                        np.save(out/lm,np.array([[55,55],[200,55],[220,190],[128,230],[40,190]],dtype=np.float32))
                    rows.append({**info,'frame_idx':frame,'frame_id':f'{ds}/{method}/{video}/{frame:06d}',
                                 'crop':rel.as_posix(),'crop_sha256':sha256(path),'landmarks':lm,
                                 'landmarks_sha256':sha256(out/lm) if lm else ''})
    for name,items in [('frames.csv',rows),('videos.csv',videos)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    write_json(out/'SYNTHETIC_ONLY.json',{'scientific_use_allowed':False})
    return out/'frames.csv'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='research/tests/fixture')
    print(make_fixture(p.parse_args().out))
