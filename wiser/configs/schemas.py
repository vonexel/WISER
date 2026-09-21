from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from hydra.core.config_store import ConfigStore


@dataclass
class CDCConfig:
    in_ch: int = 3
    out_ch: int = 32
    theta: float = 0.7


@dataclass
class WaveletStreamConfig:
    enabled: bool = True
    use_defocus: bool = True
    high_freq_gate: bool = True


@dataclass
class BiSSMConfig:
    enabled: bool = True
    d_state: int = 16
    d_conv: int = 4
    expand: int = 1
    bidirectional: bool = True


@dataclass
class SSCAConfig:
    enabled: bool = True
    heads: int = 1
    dim: int = 256


@dataclass
class CNDConfig:
    enabled: bool = False
    zc_dim: int = 192
    zn_dim: int = 64
    num_methods: int = 6
    grl_lambda_max: float = 0.3
    orth_warmup_epoch: int = 5


@dataclass
class ModelConfig:
    _target_: str = "wiser.models.wiser.WISER"
    embed_dim: int = 256
    num_classes: int = 1
    dropout: float = 0.3
    cdc: CDCConfig = field(default_factory=CDCConfig)
    wavelet: WaveletStreamConfig = field(default_factory=WaveletStreamConfig)
    bissm: BiSSMConfig = field(default_factory=BiSSMConfig)
    ssca: SSCAConfig = field(default_factory=SSCAConfig)
    cnd: CNDConfig = field(default_factory=CNDConfig)
    compile: bool = False


@dataclass
class FocalConfig:
    enabled: bool = True
    alpha: float = 0.25
    gamma: float = 2.0
    weight: float = 1.0


@dataclass
class MPHCConfig:
    enabled: bool = True
    K: int = 4
    embed_dim: int = 256
    margin: float = 0.35
    temperature: float = 0.07
    asymmetric: bool = True
    hard_negative_weight: float = 0.1
    weight: float = 0.5
    rp_enabled: bool = False
    lambda_real_margin: float = 0.0
    real_delta: float = 0.40
    hard_real_topk_frac: float = 0.25
    hard_real_weight: float = 2.0


@dataclass
class FreqConsistencyConfig:
    enabled: bool = True
    weight: float = 0.1


@dataclass
class BrierConfig:
    enabled: bool = False
    weight: float = 0.05


@dataclass
class CNDLossConfig:
    enabled: bool = False
    weight: float = 0.3
    aux_weight: float = 1.0
    grl_weight: float = 1.0
    orth_enabled: bool = False
    orth_weight: float = 0.05


@dataclass
class LossConfig:
    focal: FocalConfig = field(default_factory=FocalConfig)
    mphc: MPHCConfig = field(default_factory=MPHCConfig)
    freq: FreqConsistencyConfig = field(default_factory=FreqConsistencyConfig)
    brier: BrierConfig = field(default_factory=BrierConfig)
    cnd: CNDLossConfig = field(default_factory=CNDLossConfig)


@dataclass
class SBIConfig:
    enabled: bool = True
    p_sbi: float = 0.5
    feather_sigma_min: float = 1.0
    feather_sigma_max: float = 5.0
    alpha_ll: float = 0.15
    energy_boost: float = 1.5
    soft_label: float = 1.0


@dataclass
class FreqMaskConfig:
    enabled: bool = True
    p_freqmask: float = 0.3
    min_size_frac: float = 0.05
    max_size_frac: float = 0.25
    high_freq_bias: float = 0.7


@dataclass
class ARCAConfig:
    enabled: bool = False
    p_apply_real: float = 0.5
    jpeg_qf_min: int = 30
    jpeg_qf_max: int = 60
    noise_sigma_max: float = 0.0314
    median_blur_kernel: int = 3
    gamma_min: float = 0.7
    gamma_max: float = 1.3
    p_jpeg: float = 0.7
    p_noise: float = 0.5
    p_blur: float = 0.3
    p_gamma: float = 0.5


@dataclass
class RealMixStyleConfig:
    enabled: bool = False
    p_mix: float = 0.3


@dataclass
class AWBV2BandsConfig:
    perturb_ll: bool = False
    high_bands: list[str] = field(default_factory=lambda: ["LH", "HL", "HH"])
    band_temperature: float = 0.7


@dataclass
class AWBV2StrengthConfig:
    min_strength: float = 0.05
    max_strength: float = 0.35
    strength_step: float = 0.02
    init_strength: float = 0.20


@dataclass
class AWBV2HardnessConfig:
    enabled: bool = True
    ema_decay: float = 0.95
    hard_fake_threshold: float = 0.45
    false_real_threshold: float = 0.65


@dataclass
class AWBV2SoftLabelConfig:
    enabled: bool = True
    min_label: float = 0.55
    max_label: float = 0.95


@dataclass
class AWBV2RealGuardConfig:
    enabled: bool = True
    cooldown_steps: int = 100
    strength_multiplier: float = 0.75


@dataclass
class AWBV2LoggingConfig:
    log_band_probs: bool = True
    log_strength: bool = True
    log_soft_label_mean: bool = True


@dataclass
class AWBSBIv2Block:
    enabled: bool = False
    name: str = "awb_sbi_v2"
    p_apply: float = 0.5
    use_sbi_mask: bool = True
    operate_on_wavelet: bool = True
    bands: AWBV2BandsConfig = field(default_factory=AWBV2BandsConfig)
    strength: AWBV2StrengthConfig = field(default_factory=AWBV2StrengthConfig)
    hardness: AWBV2HardnessConfig = field(default_factory=AWBV2HardnessConfig)
    soft_labels: AWBV2SoftLabelConfig = field(default_factory=AWBV2SoftLabelConfig)
    real_preservation_guard: AWBV2RealGuardConfig = field(default_factory=AWBV2RealGuardConfig)
    logging: AWBV2LoggingConfig = field(default_factory=AWBV2LoggingConfig)


@dataclass
class AugmentConfig:
    sbi: SBIConfig = field(default_factory=SBIConfig)
    freqmask: FreqMaskConfig = field(default_factory=FreqMaskConfig)
    arca: ARCAConfig = field(default_factory=ARCAConfig)
    real_mixstyle: RealMixStyleConfig = field(default_factory=RealMixStyleConfig)
    awb_v2: AWBSBIv2Block = field(default_factory=AWBSBIv2Block)
    use_awb_sbi: bool = False
    horizontal_flip: float = 0.5
    color_jitter: float = 0.5
    jpeg_compression: float = 0.3
    gauss_noise: float = 0.2
    gauss_blur: float = 0.2
    brightness_contrast: float = 0.3
    shift_scale_rotate: float = 0.5


@dataclass
class DatasetPaths:
    raw_root: str = "${oc.env:WISER_RAW,${hydra:runtime.cwd}/dataset}"
    cache_root: str = "${oc.env:WISER_CACHE,${hydra:runtime.cwd}/preprocessed}"


@dataclass
class DataConfig:
    name: str = "ffpp"
    paths: DatasetPaths = field(default_factory=DatasetPaths)
    img_size: int = 256
    wavelet_size: int = 128
    frames_per_video: int = 100
    frame_stride: int = 10
    crop_margin: float = 0.30
    batch_size: int = 96
    eval_batch_size: int = 128
    num_workers: int = 12
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    val_fraction: float = 0.14
    test_fraction: float = 0.14
    split_seed: int = 42
    rounds_per_epoch: int = 16


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    epochs: int = 60
    warmup_frac: float = 0.05
    min_lr: float = 1e-6


@dataclass
class CalibrationConfig:
    enabled: bool = False
    target_per_class: int = 4000
    val_seed: int = 42
    video_pool_calibrated: str = "median"
    video_pool_uncalibrated: str = "mean"


@dataclass
class TrainingConfig:
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    epochs: int = 60
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    amp_dtype: str = "bf16"
    ema_decay: float = 0.9999
    early_stopping_patience: int = 10
    log_every: int = 50
    sanity_steps: int = 0
    eval_during_training: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 0
    experiment_id: str = "default"
    tags: list[str] = field(default_factory=list)
    output_root: str = "${hydra:runtime.cwd}/outputs"


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    cs.store(group="model", name="model_schema", node=ModelConfig)
    cs.store(group="data", name="data_schema", node=DataConfig)
    cs.store(group="loss", name="loss_schema", node=LossConfig)
    cs.store(group="augment", name="augment_schema", node=AugmentConfig)
    cs.store(group="training", name="training_schema", node=TrainingConfig)