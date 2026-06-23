from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import FallDetectionPipeline


def _source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO26 + RTMPose + ByteTrack + ST-GCN fall detection"
    )
    parser.add_argument("--source", required=True, help="Video path, RTSP URL, or camera index")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--show",
        "--display",
        dest="show",
        action="store_true",
        help="Show the annotated video while processing; press q or Esc to stop",
    )
    parser.add_argument("--output-video")
    parser.add_argument("--event-log")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.show:
        config.runtime.display = True
    if args.output_video:
        config.runtime.output_video = args.output_video
    if args.event_log:
        config.runtime.event_log = args.event_log

    pipeline = FallDetectionPipeline(config)
    if not pipeline.stgcn.ready:
        print(
            "WARNING: ST-GCN checkpoint is not ready. The pipeline will emit only "
            f"rule-based suspected events. Detail: {pipeline.stgcn.load_error}"
        )
    pipeline.run(_source(args.source))


if __name__ == "__main__":
    main()
