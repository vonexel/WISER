"""Build a checksummed Linux transfer bundle; datasets remain separate."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from research.tools.common import ROOT, sha256, write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='delivery/wiser-paper2-linux.tar.gz');a=p.parse_args()
    files=[]
    folders=['wiser','conf','scripts','research/tools','research/configs','research/linux','research/tests',
             'research/review','research/archive','research/data']
    for folder in folders:
        for q in (ROOT/folder).rglob('*'):
            if q.is_file() and '__pycache__' not in q.parts and 'fixture' not in q.parts and q.suffix!='.pyc':
                # Do not package generated licensed crops or full manifests accidentally.
                if folder=='research/data' and q.name not in ['inventory.json','videos.csv']:continue
                files.append(q)
    for name in ['README.md','SECOND_README.md','pyproject.toml','LICENSE','research/__init__.py','research/REVIEW.md','research/RUNBOOK.md']:
        files.append(ROOT/name)
    files += [q for q in (ROOT/'research/source').iterdir() if q.suffix in ['.zip','.json'] or q.name=='manuscript.pdf']
    files=sorted(set(files))
    output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True)
    manifest=''.join(f'{sha256(q)}  {q.relative_to(ROOT).as_posix()}\n' for q in files)
    with tarfile.open(output,'w:gz') as tar:
        for q in files:
            tar.add(q,arcname=q.relative_to(ROOT).as_posix(),recursive=False)
        info=tarfile.TarInfo('SHA256SUMS');data=manifest.encode();info.size=len(data);tar.addfile(info,io.BytesIO(data))
    # Reopen and verify every archived byte against the embedded manifest.
    with tarfile.open(output,'r:gz') as tar:
        for line in manifest.splitlines():
            digest,name=line.split('  ',1)
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest()!=digest:raise ValueError(name)
    digest=sha256(output)
    output.with_suffix(output.suffix+'.sha256').write_text(f'{digest}  {output.name}\n',encoding='utf-8',newline='\n')
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'sha256':digest,'verified_files':len(files)}))

if __name__=='__main__':main()
