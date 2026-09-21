"""Inventory licensed local videos; construct component-disjoint splits; crop on Windows/Linux.

No downloads. Source-video IDs are provenance groups, not verified human identities.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import random
import platform
import importlib.metadata
from pathlib import Path
from research.tools.common import write_json, sha256
from wiser.utils.progress import progress

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter']
PREFIX = {'FaceReenact': 'FR', 'FaceSwap': 'FS', 'TalkingFace': 'TF'}

def component_split(real_ids, pairs, seed=42):
    parent = {v: v for v in sorted(real_ids)}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        if a not in parent or b not in parent:
            raise ValueError(f'Unknown source ID: {a}, {b}')
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)
    groups = collections.defaultdict(list)
    for v in sorted(parent):
        groups[find(v)].append(v)
    components = sorted(groups.values())
    random.Random(seed).shuffle(components)
    # Largest first; randomized tie order. Whole components never get split.
    components.sort(key=len, reverse=True)
    fractions = {'train': .72, 'val': .07, 'cal': .07, 'test': .14}
    counts = dict.fromkeys(fractions, 0)
    assignments = {}
    for comp in components:
        split = max(fractions, key=lambda s: fractions[s] * len(parent) - counts[s])
        counts[split] += len(comp)
        for v in comp:
            assignments[v] = (split, min(comp))
    if min(counts.values()) == 0:
        raise ValueError('Components cannot support four nonempty splits; review provenance graph')
    return assignments, {'source_counts': counts, 'components': len(components),
                         'largest_component': max(map(len, components))}

def inventory(raw_root, out, hash_videos=False):
    raw_root, out = Path(raw_root), Path(out)
    ff = raw_root / 'ff_c23/FaceForensics++_C23'
    real = sorted((ff / 'original').glob('*.mp4'))
    if len(real) != 1000:
        raise ValueError(f'Expected 1000 FF++ originals, found {len(real)}')
    fake = [(m, p) for m in METHODS for p in sorted((ff / m).glob('*.mp4'))]
    for m in METHODS:
        if not any(k == m for k, _ in fake):
            raise ValueError(f'Missing FF++ method {m}')
    pairs = []
    for _, p in fake:
        tokens = p.stem.split('_')
        if len(tokens) != 2:
            raise ValueError(f'Ambiguous FF++ provenance: {p.name}')
        pairs.append(tokens)
    assign, info = component_split([p.stem for p in real], pairs)
    rows = []
    def add(ds, method, p, split, source_ids, group):
        row = {'dataset': ds, 'method': method, 'video_id': method + '/' + p.stem,
               'label': int(method not in ['real', 'Celeb-real', 'YouTube-real']),
               'split': split, 'source_ids': '|'.join(source_ids), 'group_id': group,
               'path': p.relative_to(raw_root).as_posix(), 'bytes': p.stat().st_size}
        if hash_videos:
            row['sha256'] = sha256(p)
        rows.append(row)
    for p in real:
        split, group = assign[p.stem]
        add('ffpp', 'real', p, split, [p.stem], group)
    for (m, p), ids in zip(fake, pairs):
        split, group = assign[ids[0]]
        assert assign[ids[1]] == (split, group)
        add('ffpp', m, p, split, ids, group)
    cdf = raw_root / 'celebdfpp'
    for m in ['Celeb-real', 'YouTube-real']:
        for p in sorted((cdf / m).glob('*.mp4')):
            add('celebdfpp', m, p, 'external', [], m + '/' + p.stem)
    for cat, prefix in PREFIX.items():
        for d in sorted((cdf / 'Celeb-synthesis' / cat).glob('*')):
            if d.is_dir():
                for p in sorted(d.glob('*.mp4')):
                    add('celebdfpp', prefix + '-' + d.name, p, 'external', [], prefix + '-' + d.name + '/' + p.stem)
    if len({r['method'] for r in rows if r['dataset'] == 'celebdfpp' and r['label']}) != 22:
        raise ValueError('Expected 22 external manipulation methods')
    keys = [(r['dataset'], r['video_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate video IDs')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'videos.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    info.update({'seed': 42, 'protocol': 'prospective-all-local-v1',
                 'external_protocol': 'all locally available videos; NOT assumed official GFD-eval',
                 'video_counts': dict(collections.Counter(r['dataset'] + '/' + r['split'] for r in rows)),
                 'method_counts': dict(collections.Counter(r['dataset'] + '/' + r['method'] for r in rows)),
                 'raw_bytes': sum(r['bytes'] for r in rows), 'videos_sha256': sha256(out / 'videos.csv'),
                 'raw_content_hashed': hash_videos, 'excluded_ffpp_methods': ['DeepFakeDetection'],
                 'identity_limit': 'Connected source-video provenance, not biometric identity deduplication'})
    write_json(out / 'inventory.json', info)
    print(json.dumps(info, indent=2))

def crop(args):
    import cv2
    import numpy as np
    import torch
    import face_alignment
    from facenet_pytorch import MTCNN
    from PIL import Image
    raw, out = Path(args.raw_root), Path(args.out)
    rows = list(csv.DictReader((out / 'videos.csv').open(encoding='utf-8')))
    out_frames = out / ('frames_smoke.csv' if args.limit else 'frames.csv')
    if out_frames.exists():
        raise FileExistsError(f'{out_frames} exists; preserve it and use a fresh output directory')
    device = args.device
    preprocessing = {'platform':platform.platform(), 'python':platform.python_version(),
                     'device':device, 'torch':str(torch.__version__),
                     'opencv':cv2.__version__, 'script_sha256':sha256(__file__),
                     'packages':{p:importlib.metadata.version(p) for p in ['face-alignment','facenet-pytorch','numpy','Pillow']}}
    detector = MTCNN(image_size=256, margin=76, keep_all=False,
                     select_largest=True, post_process=False, device=device)
    landmarks = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                              flip_input=False, device=device)
    # No Haar/ellipse fallbacks: record failures and review exclusions before training.
    frames, failures = [], []
    selected_rows = rows[:args.limit] if args.limit else rows
    video_bar = progress(selected_rows, desc='Preprocessing videos', unit='video')
    for row in video_bar:
        video_bar.set_postfix(video=row['video_id'], accepted=len(frames), failures=len(failures), refresh=False)
        shard = out / 'crop_progress' / row['dataset'] / (row['video_id'] + '.json')
        if shard.exists():
            saved = json.loads(shard.read_text(encoding='utf-8'))
            if saved['source'] != row:
                raise ValueError('Video inventory changed since partial preprocessing')
            if saved.get('preprocessing') != preprocessing:
                raise ValueError('Preprocessing environment changed since partial crop; use the original environment or a fresh data directory')
            frames.extend(saved['frames']); failures.extend(saved['failures'])
            continue
        frame_start, failure_start = len(frames), len(failures)
        cap = cv2.VideoCapture(str(raw / row['path']))
        if not cap.isOpened():
            failures.append({**row, 'reason': 'video_open_failed'})
            write_json(shard, {'source':row,'frames':[],'failures':failures[failure_start:], 'preprocessing':preprocessing})
            continue
        frame_idx, selected = 0, 0
        while selected < 100:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_idx % 10 == 0:
                selected += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                crop_t = detector(Image.fromarray(rgb))
                if crop_t is None:
                    failures.append({**row, 'frame_idx': frame_idx, 'reason': 'no_face'})
                else:
                    arr = crop_t.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                    rel = Path('crops') / row['dataset'] / row['video_id'] / f'frame_{frame_idx:06d}.jpg'
                    path = out / rel; path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise IOError(path)
                    # Landmarks and every modality use the exact decoded stored crop.
                    decoded = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    pts = landmarks.get_landmarks(decoded) if row['dataset'] == 'ffpp' and row['label'] == '0' else None
                    lm_rel = ''
                    if row['dataset'] == 'ffpp' and row['label'] == '0':
                        if pts is None:
                            failures.append({**row, 'frame_idx': frame_idx, 'reason': 'no_landmarks'})
                            frame_idx += 1
                            continue
                        lm_rel = rel.with_suffix('.landmarks.npy').as_posix()
                        np.save(out / lm_rel, np.asarray(pts[0], dtype=np.float32))
                    frames.append({**row, 'frame_idx': frame_idx,
                                   'frame_id': row['dataset'] + '/' + row['video_id'] + f'/{frame_idx:06d}',
                                   'crop': rel.as_posix(), 'crop_sha256': sha256(path),
                                   'landmarks': lm_rel,
                                   'landmarks_sha256': sha256(out / lm_rel) if lm_rel else ''})
            frame_idx += 1
        cap.release()
        if selected == 0:
            failures.append({**row, 'reason': 'no_decoded_frames'})
        write_json(shard, {'source':row, 'frames':frames[frame_start:], 'failures':failures[failure_start:], 'preprocessing':preprocessing})
    if not frames:
        raise RuntimeError('No crops produced')
    with out_frames.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(frames[0])); w.writeheader(); w.writerows(frames)
    accepted = {(r['dataset'], r['video_id']) for r in frames}
    omitted = [r for r in selected_rows if (r['dataset'], r['video_id']) not in accepted]
    write_json(out / ('crop_report_smoke.json' if args.limit else 'crop_report.json'),
               {'frames': len(frames), 'failures': failures, 'omitted_videos': omitted,
                'manifest_sha256': sha256(out_frames), 'detector': 'MTCNN only, largest crop; no landmark alignment warp',
                'torch': str(torch.__version__), 'smoke_only': bool(args.limit),
                'preprocessing':preprocessing})
    print(f'Wrote {out_frames}. Audit omitted videos and freeze manifest before training.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inventory', 'crop'])
    p.add_argument('--raw-root', default='datasets')
    p.add_argument('--out', default='research/data')
    p.add_argument('--hash-videos', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=0, help='Crop smoke only; never creates frames.csv')
    a = p.parse_args()
    inventory(a.raw_root, a.out, a.hash_videos) if a.stage == 'inventory' else crop(a)

if __name__ == '__main__':
    main()
