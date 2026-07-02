from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import GCNConfig


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


class HPIGCNModel(nn.Module):
    """HPI-GCN model matching checkpoint structure (25 joints, 3 channels for NTU RGB+D)."""
    
    def __init__(self, num_classes: int = 3, input_channels: int = 3, num_nodes: int = 25) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_nodes = num_nodes
        
        # Data batch normalization - matches checkpoint shape [75] for 25 joints
        self.data_bn = nn.BatchNorm1d(input_channels * num_nodes)
        
        # Build blocks matching checkpoint structure (l2-l10)
        self.blocks = nn.ModuleList([
            self._make_block(input_channels, 64, stride=1),    # l2
            self._make_block(64, 64, stride=1),                  # l3
            self._make_block(64, 64, stride=1),                  # l4
            self._make_block(64, 128, stride=2),                 # l5
            self._make_block(128, 128, stride=1),                # l6
            self._make_block(128, 128, stride=1),                # l7
            self._make_block(128, 256, stride=2),                # l8
            self._make_block(256, 256, stride=1),                # l9
            self._make_block(256, 256, stride=1),                # l10
        ])
        
        # Final classifier
        self.classifier = nn.Linear(256, num_classes)
    
    def _make_block(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
        """Create a GCN block matching checkpoint structure."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=(stride, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input of shape (N, C, T, V), got {tuple(x.shape)}")
        
        batch_size, channels, time_steps, num_nodes = x.shape
        if channels != self.input_channels or num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.input_channels} channels and {self.num_nodes} joints, got "
                f"{channels} channels and {num_nodes} joints"
            )
        
        # Data normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, num_nodes * channels, time_steps)
        x = self.data_bn(x)
        x = x.view(batch_size, num_nodes, channels, time_steps).permute(0, 2, 3, 1).contiguous()
        
        # Pass through GCN blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=-1).mean(dim=-1)
        
        # Classification
        return self.classifier(x)


class GCNRuntime:
    """HPI-GCN runtime for skeleton sequences."""
    
    def __init__(self, config: GCNConfig):
        self.config = config
        self.classes = tuple(config.classes)
        self.ready = False
        self.load_error: str | None = None
        self.device = torch.device("cpu")  # Default device
        self.model = None
        
        self.checkpoint = Path(config.checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"HPI-GCN checkpoint not found: {self.checkpoint}. "
                "Train/fine-tune the fall model before running inference."
            )
        
        # Detect checkpoint structure
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        data_bn_shape = None
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                if 'data_bn.running_mean' in key:
                    data_bn_shape = checkpoint[key].shape[0]
                    break
        
        # Calculate expected joints from data_bn shape (assuming 3 channels)
        expected_joints = data_bn_shape // 3 if data_bn_shape else 50
        
        # Checkpoint expects different joint count than RTMPose provides
        # RTMPose provides 17 COCO joints, checkpoint expects {expected_joints} joints
        print(f"Warning: HPI-GCN disabled due to joint count mismatch")
        print(f"Checkpoint expects {expected_joints} joints, RTMPose provides 17 COCO joints")
        print("Running in rule-only mode. To enable GCN, use a checkpoint trained with COCO-17 joints.")
        self.ready = False
    
    @torch.inference_mode()
    def predict(self, sample: dict[str, Any] | None) -> dict[str, float]:
        # Return empty dict when GCN is disabled
        return {}


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)
