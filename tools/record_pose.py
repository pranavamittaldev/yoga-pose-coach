"""Record your own target angles for a pose from the webcam.

Opens the camera and shows your skeleton with the pose's joint angles. Get
into the pose, press Space, and hold still for 5 seconds. Each joint angle is
averaged over the recording (frames where its landmarks are not visible are
ignored), printed, and written into poses/poses.json as that pose's new
``target_angle``, with ``"calibrated": true``. Nothing is saved except
poses.json; no video is written.

Usage:
    python tools/record_pose.py tree              # record Tree Pose from the webcam
    python tools/record_pose.py tree --source 1   # a different camera
    python tools/record_pose.py --list            # show the pose ids

Keys: Space = start recording, Q or Esc = quit without saving.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from app.camera import Camera, CameraError  # noqa: E402
from app.geometry import joint_angles  # noqa: E402
from app.gui import VIDEO_H, compose_view  # noqa: E402
from app.pose_detector import PoseDetector  # noqa: E402
from app.session import Pose, load_poses, write_poses_json  # noqa: E402

WINDOW = "Record reference pose"


def average_angles(frames: Iterable[dict[str, Optional[float]]], joint_names: Iterable[str]) -> dict[str, dict]:
    """Average each joint's angle over the frames where it was measurable.

    ``frames`` holds one ``{joint name: angle or None}`` dict per frame (None
    means not visible). Returns ``{joint: {"angle": mean or None, "frames": n}}``
    for every name in ``joint_names``; ``angle`` is None if the joint was never
    measured.
    """
    frames = list(frames)
    out = {}
    for name in joint_names:
        values = [f[name] for f in frames if f.get(name) is not None]
        out[name] = {"angle": sum(values) / len(values) if values else None, "frames": len(values)}
    return out


def apply_calibration(data: dict, pose_id: str, averages: dict[str, dict]) -> list[tuple[str, float, Optional[float]]]:
    """Write averaged angles into raw poses.json ``data`` (in place) and mark the pose calibrated.

    Joints with no measured angle keep their old target. Returns a list of
    ``(joint, old target, new target or None)`` for reporting. Raises
    ValueError if the pose does not exist or no joint was measured at all.
    """
    pose = next((p for p in data["poses"] if p["id"] == pose_id), None)
    if pose is None:
        raise ValueError(f"No pose with id {pose_id!r} in poses.json")
    if not any(averages.get(j["name"], {}).get("angle") is not None for j in pose["joints"]):
        raise ValueError("No joint was measured during the recording; poses.json was not changed")
    changes = []
    for joint in pose["joints"]:
        new = averages.get(joint["name"], {}).get("angle")
        old = joint["target_angle"]
        if new is not None:
            joint["target_angle"] = round(new, 1)
        changes.append((joint["name"], old, None if new is None else round(new, 1)))
    pose["calibrated"] = True
    return changes


def _banner(view_bgr, text: str, color=(255, 255, 255)) -> None:
    """Draw a dark strip with ``text`` near the bottom of the video."""
    y0, y1 = VIDEO_H - 78, VIDEO_H - 46
    view_bgr[y0:y1] = (view_bgr[y0:y1] * 0.3).astype(view_bgr.dtype)
    cv2.putText(view_bgr, text, (16, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def record(pose: Pose, source, seconds: float = config.RECORD_SECONDS) -> Optional[list[dict]]:
    """Show the live skeleton, wait for Space, then collect joint angles for ``seconds``.

    Returns one angle dict per frame of the recording, or None if the user quit.
    Raises CameraError if the source cannot be opened.
    """
    frames: list[dict] = []
    rec_start: Optional[float] = None
    fps, last_t = 0.0, None
    with Camera(source) as cam, PoseDetector() as detector:
        frame_period = 1.0 / (cam.file_fps() or config.TARGET_FPS) if cam.is_file else 0.0
        cv2.namedWindow(WINDOW)
        while True:
            started = time.monotonic()
            frame = cam.read()
            if frame is None:
                if cam.is_file:
                    cam.rewind()
                    continue
                raise CameraError("The camera stopped sending frames.")
            t = time.monotonic()
            landmarks = detector.detect(frame, int(t * 1000))
            inference_ms = (time.monotonic() - t) * 1000
            if last_t is not None and t > last_t:
                fps = 0.9 * fps + 0.1 / (t - last_t) if fps else 1.0 / (t - last_t)
            last_t = t
            angles = joint_angles(landmarks, pose.joints, config.VISIBILITY_THRESHOLD) if landmarks is not None else None

            if rec_start is not None:
                frames.append(angles or {})
                left = seconds - (t - rec_start)
                if left <= 0:
                    return frames
            view = cv2.cvtColor(compose_view(frame, landmarks, pose.joints, angles, fps, inference_ms),
                                cv2.COLOR_RGB2BGR)
            if rec_start is None:
                _banner(view, f"{pose.name}: get into the pose, then press SPACE")
            else:
                _banner(view, f"RECORDING... hold still  {left:.1f} s", (80, 80, 255))
            cv2.imshow(WINDOW, view)

            wait_ms = max(1, int((frame_period - (time.monotonic() - started)) * 1000)) if frame_period else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key == ord(" ") and rec_start is None:
                rec_start = time.monotonic()
                print(f"Recording {seconds:g} s...")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # window closed
                return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record target angles for a pose from the webcam.")
    parser.add_argument("pose", nargs="?", help="pose id from poses.json, e.g. tree")
    parser.add_argument("--source", default=str(config.CAMERA_INDEX),
                        help="camera index or video file (default: %(default)s)")
    parser.add_argument("--seconds", type=float, default=config.RECORD_SECONDS,
                        help="recording length (default: %(default)s)")
    parser.add_argument("--list", action="store_true", help="list the pose ids and exit")
    args = parser.parse_args(argv)

    poses = {p.id: p for p in load_poses()}
    if args.list or not args.pose:
        for p in poses.values():
            print(f"  {p.id:14s} {p.name}{'  (calibrated)' if p.calibrated else ''}")
        return 0 if args.list else 2
    if args.pose not in poses:
        print(f"Unknown pose {args.pose!r}. Choose one of: {', '.join(poses)}")
        return 2
    pose = poses[args.pose]
    source = int(args.source) if args.source.isdigit() else args.source

    try:
        frames = record(pose, source, args.seconds)
    except CameraError as e:
        print(f"Camera error: {e}")
        return 1
    finally:
        cv2.destroyAllWindows()
    if frames is None:
        print("Quit without saving.")
        return 1

    averages = average_angles(frames, [j.name for j in pose.joints])
    with open(config.POSES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    try:
        changes = apply_calibration(data, pose.id, averages)
    except ValueError as e:
        print(e)
        return 1

    print(f"\n{pose.name}: averaged over {len(frames)} frames")
    print(f"  {'joint':16s} {'old':>7s} {'new':>7s}  frames")
    for name, old, new in changes:
        n = averages[name]["frames"]
        new_text = f"{new:7.1f}" if new is not None else "  (kept, not visible)"
        print(f"  {name:16s} {old:7.1f} {new_text}  {n}")
    write_poses_json(data)
    print(f"Saved to {config.POSES_FILE.relative_to(config.PROJECT_ROOT)} with \"calibrated\": true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
