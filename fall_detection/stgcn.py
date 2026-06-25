from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import STGCNConfig


COCO_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


MMACTION_INSTALL_HINT = (
    "MMAction2 is mandatory for ST-GCN inference. Install the OpenMMLab stack "
    "with: pip install -U openmim && mim install mmengine && mim install mmcv "
    "&& pip install mmaction2"
)


class STGCNRuntime:
    """Mandatory MMAction2 ST-GCN runtime for COCO-17 skeleton sequences."""

    def __init__(self, config: STGCNConfig):
        self.config = config
        self.classes = tuple(config.classes)
        self.ready = False
        self.load_error: str | None = None

        self.config_file = Path(config.config_file)
        self.checkpoint = Path(config.checkpoint)
        if not self.config_file.exists():
            raise FileNotFoundError(f"MMAction2 ST-GCN config not found: {self.config_file}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"MMAction2 ST-GCN checkpoint not found: {self.checkpoint}. "
                "Train/fine-tune the fall model before running inference."
            )

        try:
            from mmaction.apis import init_recognizer
            from mmengine.dataset import Compose, pseudo_collate
        except ImportError as error:
            raise RuntimeError(f"{MMACTION_INSTALL_HINT}. Original error: {error}") from error

        self._pseudo_collate = pseudo_collate
        self.model = init_recognizer(
            str(self.config_file),
            str(self.checkpoint),
            device=self._resolve_device(config.device),
        )
        self.model.eval()

        test_pipeline = getattr(self.model.cfg, "test_pipeline", None)
        if test_pipeline is None:
            raise ValueError(
                f"{self.config_file} does not define test_pipeline required by MMAction2"
            )
        self.pipeline = Compose(test_pipeline)
        self.ready = True

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
        return "cpu"

    @torch.inference_mode()
    def predict(self, sample: dict[str, Any] | None) -> dict[str, float]:
        if sample is None:
            return {}

        data = self.pipeline(dict(sample))
        batch = self._pseudo_collate([data])
        result = self.model.test_step(batch)[0]
        scores = _to_numpy(result.pred_score).reshape(-1)

        if len(scores) != len(self.classes):
            raise ValueError(
                f"MMAction2 checkpoint outputs {len(scores)} classes, "
                f"but config declares {len(self.classes)} classes: {self.classes}"
            )
        return {
            class_name: float(probability)
            for class_name, probability in zip(self.classes, scores)
        }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)
