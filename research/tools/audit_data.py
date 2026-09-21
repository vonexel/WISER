"""Fail-closed data integrity checks and freeze the exact prospective frame manifest."""
import argparse
import collections
import csv
from pathlib import Path
from research.tools.common import sha256, write_json
from wiser.utils.progress import progress

def audit(manifest, verify_files=True, exclusion_reason=''):
    manifest = Path(manifest); root = manifest.parent
    if (root / 'SYNTHETIC_ONLY.json').exists():
        raise ValueError('Synthetic fixtures cannot be frozen for scientific training')
    if manifest.name != 'frames.csv':
        raise ValueError('Only the full frames.csv can be frozen')
    groups = {}; seen = set(); hashes = {}; observed = set(); counts = collections.Counter()
    videos = list(csv.DictReader((root / 'videos.csv').open(encoding='utf-8')))
    allowed = {(r['dataset'], r['video_id']): r for r in videos}
    frame_counts = collections.Counter()
    with manifest.open(encoding='utf-8') as f:
        for r in progress(csv.DictReader(f), desc='Auditing crop/landmark hashes', unit='frame'):
            key = (r['dataset'], r['video_id'])
            if key not in allowed:
                raise ValueError('Unexpected video ' + str(key))
            for field in ['label', 'split', 'source_ids', 'group_id', 'method']:
                if r[field] != allowed[key][field]:
                    raise ValueError(f'Provenance mismatch: {key}/{field}')
            if r['frame_id'] in seen:
                raise ValueError('Duplicate frame ID')
            seen.add(r['frame_id']); observed.add(key); frame_counts[key] += 1
            if int(r['frame_idx']) % 10:
                raise ValueError('Unexpected frame stride')
            for sid in filter(None, r['source_ids'].split('|')):
                g = (r['dataset'], sid)
                if g in groups and groups[g] != r['split']:
                    raise ValueError('Source ID leakage: ' + str(g))
                groups[g] = r['split']
            h = r['crop_sha256']
            # Exact duplicate content crossing partitions is invalid, even if file IDs differ.
            if h in hashes and hashes[h] != r['split']:
                raise ValueError('Exact crop duplicate across partitions')
            hashes[h] = r['split']
            for key_path, key_hash in [('crop', 'crop_sha256'), ('landmarks', 'landmarks_sha256')]:
                if not r[key_path]:
                    continue
                path = (root / r[key_path]).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError('Manifest path escapes data directory')
                if verify_files and (not path.is_file() or sha256(path) != r[key_hash]):
                    raise ValueError('Missing/corrupted input: ' + str(path))
            if r['dataset'] == 'ffpp' and r['label'] == '0' and not r['landmarks']:
                raise ValueError('Missing real landmarks')
            counts[r['dataset'] + '/' + r['split'] + '/' + r['label']] += 1
    if not seen or max(frame_counts.values()) > 100:
        raise ValueError('Invalid frame counts')
    for split in ['train', 'val', 'cal', 'test']:
        if any(counts[f'ffpp/{split}/{y}'] == 0 for y in [0, 1]):
            raise ValueError('Empty class in ' + split)
    if any(counts[f'celebdfpp/external/{y}'] == 0 for y in [0, 1]):
        raise ValueError('Empty external class')
    omitted = sorted(set(allowed) - observed)
    if omitted and not exclusion_reason:
        raise ValueError(f'{len(omitted)} videos have no accepted frames. Review crop_report.json; record an exclusion reason before freezing.')
    return {'manifest_sha256': sha256(manifest), 'videos_sha256': sha256(root / 'videos.csv'),
            'frames': len(seen), 'videos': len(observed), 'counts': dict(counts),
            'omitted_videos': omitted, 'exclusion_reason': exclusion_reason,
            'content_hashes_verified': verify_files, 'protocol': 'prospective-all-local-v1'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='research/data/frames.csv')
    p.add_argument('--exclusion-reason', default='')
    a = p.parse_args()
    result = audit(a.manifest, exclusion_reason=a.exclusion_reason)
    write_json(Path(a.manifest).parent / 'data_freeze.json', result)
    print(f'Frozen {result["frames"]} frames / {result["videos"]} videos')

if __name__ == '__main__':
    main()
