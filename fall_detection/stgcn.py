from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

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


def _normalize_digraph(matrix: np.ndarray) -> np.ndarray:
    degree = np.sum(matrix, axis=0)
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    return matrix @ np.diag(inverse)


def coco_adjacency() -> np.ndarray:
    inward = np.zeros((17, 17), dtype=np.float32)
    for source, target in COCO_EDGES:
        inward[target, source] = 1.0
    outward = inward.T
    identity = np.eye(17, dtype=np.float32)
    return np.stack(
        [identity, _normalize_digraph(inward), _normalize_digraph(outward)], axis=0
    )


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.register_buffer("adjacency", adjacency.clone())
        partitions = adjacency.shape[0]
        self.graph_conv = nn.Conv2d(in_channels, out_channels * partitions, kernel_size=1)
        self.temporal_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = lambda value: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.residual(inputs)
        features = self.graph_conv(inputs)
        batch, _, frames, joints = features.shape
        features = features.view(
            batch, self.adjacency.shape[0], -1, frames, joints
        )
        features = torch.einsum("nkctv,kvw->nctw", features, self.adjacency)
        return self.activation(self.temporal_conv(features) + residual)


class STGCN(nn.Module):
    """Compact ST-GCN for one-person COCO-17 skeleton sequences."""

    def __init__(self, input_channels: int = 7, num_classes: int = 3):
        super().__init__()
        adjacency = torch.tensor(coco_adjacency(), dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(input_channels * 17)
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(input_channels, 64, adjacency, residual=False),
                STGCNBlock(64, 64, adjacency),
                STGCNBlock(64, 64, adjacency),
                STGCNBlock(64, 128, adjacency, stride=2, dropout=0.1),
                STGCNBlock(128, 128, adjacency, dropout=0.1),
                STGCNBlock(128, 256, adjacency, stride=2, dropout=0.2),
                STGCNBlock(256, 256, adjacency, dropout=0.2),
            ]
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # N,C,T,V,M -> N*M,C,T,V
        batch, channels, frames, joints, people = inputs.shape
        features = inputs.permute(0, 4, 3, 1, 2).contiguous()
        features = features.view(batch * people, joints * channels, frames)
        features = self.data_bn(features)
        features = features.view(batch * people, joints, channels, frames)
        features = features.permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            features = block(features)
        features = features.mean(dim=(2, 3)).view(batch, people, -1).mean(dim=1)
        return self.classifier(features)


class STGCNRuntime:
    def __init__(self, config: STGCNConfig):
        self.config = config
        self.classes = tuple(config.classes)
        self.device = self._resolve_device(config.device)
        self.model = STGCN(config.input_channels, len(self.classes)).to(self.device)
        self.ready = False
        self.load_error: str | None = None
        self._load_checkpoint(Path(config.checkpoint))

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            self.load_error = f"checkpoint not found: {path}"
            self.model.eval()
            return
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
            checkpoint_classes = tuple(checkpoint.get("classes", self.classes))
            checkpoint_channels = int(
                checkpoint.get("input_channels", self.config.input_channels)
            )
            if checkpoint_classes != self.classes:
                raise ValueError(
                    f"class mismatch: checkpoint={checkpoint_classes}, config={self.classes}"
                )
            if checkpoint_channels != self.config.input_channels:
                raise ValueError(
                    f"input channel mismatch: checkpoint={checkpoint_channels}, "
                    f"config={self.config.input_channels}"
                )
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.ready = True
        except Exception as error:
            self.load_error = str(error)
            self.ready = False
            self.model.eval()

    @torch.inference_mode()
    def predict(self, features: np.ndarray | None) -> dict[str, float]:
        if not self.ready or features is None:
            return {}
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        return {
            class_name: float(probability)
            for class_name, probability in zip(self.classes, probabilities)
        }
