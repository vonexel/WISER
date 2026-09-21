"""Package completed preprocessing separately, with Linux-friendly relative paths."""
import argparse
import json
from pathlib import Path
import tarfile
from research.tools.common import sha256
from wiser.utils.progress import progress

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='research/data')
    p.add_argument('--output',default='delivery/wiser-preprocessed.tar')
    a=p.parse_args();data=Path(a.data);out=Path(a.output)
    if (data/'SYNTHETIC_ONLY.json').exists():raise ValueError('Synthetic fixtures are not a scientific data bundle')
    freeze=json.loads((data/'data_freeze.json').read_text())
    if not freeze['content_hashes_verified'] or freeze['manifest_sha256']!=sha256(data/'frames.csv') or freeze['videos_sha256']!=sha256(data/'videos.csv'):
        raise ValueError('Data must be audited/frozen before packaging')
    if out.exists():raise FileExistsError(out)
    if out.resolve().is_relative_to(data.resolve()):raise ValueError('Output must be outside data directory')
    files=sorted(q for q in data.rglob('*') if q.is_file())
    out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(out,'w') as tar:
        for q in progress(files,desc='Packing preprocessed data',unit='file'):
            tar.add(q,arcname='research/data/'+q.relative_to(data).as_posix(),recursive=False)
    digest=sha256(out)
    out.with_suffix(out.suffix+'.sha256').write_text(f'{digest}  {out.name}\n',encoding='ascii',newline='\n')
    print(out)

if __name__=='__main__':main()
