from __future__ import annotations
import csv
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from wiser.data.augmentation import build_train_pipeline, build_eval_pipeline, denormalise, IMAGENET_MEAN, IMAGENET_STD
from wiser.data.awb_sbi import AWBSBIParams, awb_self_blend
from wiser.data.sbi import SBIParams, self_blend
from wiser.data.arca import arca_params_from_config, heavy_real_augment
from wiser.data.freq_mask import apply_freq_mask
from wiser.data.preprocessing import haar_dwt_stack, defocus_map
from wiser.models.disentangle import manip_to_method_label

class ManifestDataset(Dataset):
    def __init__(self, manifest, cfg, split, dataset='ffpp'):
        self.root = Path(manifest).parent
        with Path(manifest).open(encoding='utf-8') as f:
            self.rows = [r for r in csv.DictReader(f) if r['dataset'] == dataset and r['split'] == split]
        if not self.rows:
            raise ValueError(f'Empty {dataset}/{split}')
        self.records = [SimpleNamespace(label=r['method'], video=r['video_id'].split('/', 1)[1]) for r in self.rows]
        self.cfg, self.train = cfg, split == 'train'
        self.transform = build_train_pipeline(cfg.augment, 256) if self.train else build_eval_pipeline(256)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        arr = cv2.imread(str(self.root / r['crop']))
        if arr is None:
            raise IOError(r['crop'])
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        y = int(r['label']); target = float(y); pseudo = False; method = r['method']
        rng = np.random.default_rng(int(np.random.randint(0, 2**31)))
        aug = self.cfg.augment
        if self.train and y == 0:
            if aug.sbi.enabled:
                if not r['landmarks']:
                    raise ValueError('Training real crop is missing cached landmarks')
                lm = np.load(self.root / r['landmarks'], allow_pickle=False)
                if aug.use_awb_sbi:
                    params = AWBSBIParams(p=aug.sbi.p_sbi, alpha_ll=aug.sbi.alpha_ll,
                                          energy_boost=aug.sbi.energy_boost, soft_label=aug.sbi.soft_label,
                                          feather_sigma_min=aug.sbi.feather_sigma_min,
                                          feather_sigma_max=aug.sbi.feather_sigma_max)
                    rgb, pseudo, _, target_aug = awb_self_blend(rgb, params, rng, landmarks=lm)
                else:
                    params = SBIParams(p=aug.sbi.p_sbi, feather_sigma_min=aug.sbi.feather_sigma_min,
                                       feather_sigma_max=aug.sbi.feather_sigma_max)
                    rgb, pseudo, _ = self_blend(rgb, params, rng, landmarks=lm)
                    target_aug = float(aug.sbi.soft_label)
                if pseudo:
                    y, target, method = 1, target_aug, 'sbi_pseudo'
            if not pseudo and aug.arca.enabled and rng.random() < aug.arca.p_apply_real:
                params = arca_params_from_config(aug); params.p_apply = 1.0
                rgb = heavy_real_augment(rgb, params, rng)
        rgb_t = self.transform(rgb).float()
        if self.train and aug.freqmask.enabled:
            x = apply_freq_mask(denormalise(rgb_t), p=aug.freqmask.p_freqmask,
                                min_size_frac=aug.freqmask.min_size_frac,
                                max_size_frac=aug.freqmask.max_size_frac,
                                high_freq_bias=aug.freqmask.high_freq_bias, rng=rng)
            rgb_t = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
        # All geometric/image augmentations happen before recomputing the two forensic inputs.
        decoded = np.rint(denormalise(rgb_t).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        wavelet = haar_dwt_stack(decoded).astype(np.float32)
        defocus = defocus_map(decoded, target_size=128).astype(np.float32)[None]
        return {'rgb': rgb_t, 'wavelet': torch.from_numpy(wavelet), 'defocus': torch.from_numpy(defocus),
                'label': torch.tensor(y, dtype=torch.long), 'target_soft': torch.tensor(target),
                'is_soft': torch.tensor(pseudo), 'method_label': torch.tensor(manip_to_method_label(method)),
                'manip': method, 'sbi': pseudo, 'video_id': r['video_id'], 'frame_id': r['frame_id']}
