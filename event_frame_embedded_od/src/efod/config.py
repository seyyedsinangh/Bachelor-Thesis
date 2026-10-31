from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Pipeline = Literal['event_snn', 'strict_ann', 'native_ann']

DSEC_CLASSES = (
    'pedestrian', 'rider', 'car', 'bus', 'truck',
    'bicycle', 'motorcycle', 'train'
)

# C is the channel count after the 4x4, stride-4 patchify stem.
# thesis_core_params are Table 5.2 reference counts (rounded in the thesis/repository).
MODEL_SPECS = {
    'vgg16':    {'family': 'stvgg', 'C': 16, 'display': '16-ST-VGG',       'thesis_core_params': 125_000},
    'vgg32':    {'family': 'stvgg', 'C': 32, 'display': '32-ST-VGG',       'thesis_core_params': 499_000},
    'vgg64':    {'family': 'stvgg', 'C': 64, 'display': '64-ST-VGG',       'thesis_core_params': 1_994_000},
    'rescat16': {'family': 'rescat','C': 16, 'display': 'ResCat-SNN-16',    'thesis_core_params': 216_000},
    'rescat32': {'family': 'rescat','C': 32, 'display': 'ResCat-SNN-32',    'thesis_core_params': 862_000},
    'rescat64': {'family': 'rescat','C': 64, 'display': 'ResCat-SNN-64',    'thesis_core_params': 3_442_000},
}

PIPELINES = ('event_snn', 'strict_ann', 'native_ann')
PIPELINE_DISPLAY = {
    'event_snn': 'A Event-SNN (BN-Conv-PLIF)',
    'strict_ann': 'B Strict RGB-ANN twin (BN-Conv-ReLU)',
    'native_ann': 'C ANN-native RGB (Conv-BN-ReLU)',
}

@dataclass
class ModelConfig:
    variant: str = 'vgg32'
    pipeline: Pipeline = 'event_snn'
    image_height: int = 480
    image_width: int = 640
    foreground_classes: int = len(DSEC_CLASSES)
    extra_channels: tuple[int, int, int] = (256, 128, 128)
    aspect_ratios: tuple[tuple[int, ...], ...] = (
        (2,), (2, 3), (2, 3), (2, 3), (2,), (2,)
    )
    min_ratio: float = 0.015
    max_ratio: float = 0.25
    score_thresh: float = 0.01
    nms_thresh: float = 0.45
    detections_per_img: int = 100
    plif_init_tau: float = 2.0
    plif_threshold: float = 1.0

    @property
    def input_channels(self) -> int:
        return 2 if self.pipeline == 'event_snn' else 3

    @property
    def spec(self) -> dict:
        if self.variant not in MODEL_SPECS:
            raise KeyError(f'Unknown variant {self.variant!r}; choose {sorted(MODEL_SPECS)}')
        return MODEL_SPECS[self.variant]
