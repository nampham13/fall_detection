from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .stgcn import MMACTION_INSTALL_HINT


def train(args: argparse.Namespace) -> None:
    try:
        from mmaction.utils import register_all_modules
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as error:
        raise RuntimeError(f"{MMACTION_INSTALL_HINT}. Original error: {error}") from error

    cfg = Config.fromfile(args.config)
    if args.ann_file:
        _set_ann_file(cfg, args.ann_file)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.load_from:
        cfg.load_from = args.load_from
    if args.resume:
        cfg.resume = True

    register_all_modules(init_default_scope=True)
    runner = Runner.from_cfg(cfg)
    runner.train()

    if args.export_checkpoint:
        _export_checkpoint(Path(cfg.work_dir), Path(args.export_checkpoint))


def _set_ann_file(cfg: Any, ann_file: str) -> None:
    cfg.ann_file = ann_file
    for dataloader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        dataloader = cfg.get(dataloader_name)
        if dataloader is None:
            continue
        dataset = dataloader.get("dataset")
        if dataset is not None:
            dataset["ann_file"] = ann_file


def _export_checkpoint(work_dir: Path, destination: Path) -> None:
    candidates = sorted(
        work_dir.glob("best_*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            work_dir.glob("epoch_*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise RuntimeError(f"No MMAction2 checkpoint found under {work_dir}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], destination)
    print(f"Exported checkpoint: {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MMAction2 ST-GCN fall model")
    parser.add_argument("--config", default="configs/mmaction2/stgcn_fall.py")
    parser.add_argument(
        "--ann-file",
        default="data/mmaction/fall_skeleton.pkl",
        help="MMAction2 PoseDataset .pkl built by fall_detection.dataset",
    )
    parser.add_argument("--work-dir", default="work_dirs/stgcn_fall")
    parser.add_argument(
        "--load-from",
        help="Optional OpenMMLab/pretrained checkpoint to fine-tune from",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--export-checkpoint",
        default="models/mmaction2/stgcn_fall.pth",
        help="Copy the newest best_*.pth/epoch_*.pth here after training",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
