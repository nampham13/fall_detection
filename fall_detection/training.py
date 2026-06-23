from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .stgcn import STGCN


class SkeletonNPZDataset(Dataset):
    """Dataset of NPZ files containing x[C,T,17,1] and scalar y."""

    def __init__(self, root: str | Path):
        self.files = sorted(Path(root).rglob("*.npz"))
        if not self.files:
            raise ValueError(f"No NPZ files found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[index]) as sample:
            features = np.asarray(sample["x"], dtype=np.float32)
            label = int(sample["y"])
        if features.ndim != 4 or features.shape[2:] != (17, 1):
            raise ValueError(
                f"{self.files[index]} has shape {features.shape}; expected C,T,17,1"
            )
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

    def labels(self) -> np.ndarray:
        labels = []
        for path in self.files:
            with np.load(path) as sample:
                labels.append(int(sample["y"]))
        return np.asarray(labels, dtype=np.int64)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    training = SkeletonNPZDataset(args.train_data)
    validation = SkeletonNPZDataset(args.validation_data)
    train_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = tuple(args.classes.split(","))
    model = STGCN(args.input_channels, len(classes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    class_counts = np.bincount(training.labels(), minlength=len(classes))
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    best_balanced_accuracy = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * features.shape[0]

        metrics = _metrics(model, validation_loader, device, len(classes))
        mean_loss = total_loss / len(training)
        print(
            f"epoch={epoch:03d} loss={mean_loss:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} "
            f"val_balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )
        if metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = metrics["balanced_accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "input_channels": args.input_channels,
                    "validation_metrics": metrics,
                    "epoch": epoch,
                },
                output,
            )


@torch.inference_mode()
def _metrics(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> dict[str, float | list[float]]:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for features, labels in loader:
        predictions = model(features.to(device)).argmax(dim=1).cpu()
        for target, prediction in zip(labels.numpy(), predictions.numpy()):
            confusion[int(target), int(prediction)] += 1
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recall)),
        "recall_per_class": recall.astype(float).tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train compact COCO-17 ST-GCN")
    parser.add_argument(
        "--train-data",
        required=True,
        help="Subject/video-disjoint training NPZ directory",
    )
    parser.add_argument(
        "--validation-data",
        required=True,
        help="Subject/video-disjoint validation NPZ directory",
    )
    parser.add_argument("--output", default="models/stgcn_fall.pt")
    parser.add_argument("--classes", default="normal,falling,lying")
    parser.add_argument("--input-channels", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
