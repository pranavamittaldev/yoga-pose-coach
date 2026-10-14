"""Yoga Pose Coach entry point.

    python main.py                   # use the webcam (config.CAMERA_INDEX)
    python main.py --source 1        # a different camera
    python main.py --source clip.mp4 # a video file instead of the webcam (looped)
    python main.py --program children  # preselect a program (children, men, women, all)
"""

import argparse
import sys

import config
from app.gui import App
from app.session import load_poses, load_programs


def parse_source(value: str):
    """Camera index if the value is a number, otherwise a video file path."""
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Coach a yoga sequence using your webcam.")
    parser.add_argument("--source", type=parse_source, default=config.CAMERA_INDEX,
                        help="camera index or path to a video file (default: %(default)s)")
    poses = load_poses()
    programs = load_programs(poses)
    parser.add_argument("--program", choices=[p.id for p in programs], default=programs[-1].id,
                        help="program selected on the start screen (default: %(default)s)")
    args = parser.parse_args()
    if sys.platform == "win32":
        try:  # crisp text on high-DPI screens instead of a blurry bitmap-scaled window
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    App(poses, source=args.source, programs=programs, program_id=args.program).run()


if __name__ == "__main__":
    main()
