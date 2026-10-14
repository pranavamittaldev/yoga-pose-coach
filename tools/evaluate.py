"""Evaluate speed and scoring behaviour on this laptop.

Reports three things:

1. Speed: average FPS and per-frame inference latency of the full worker
   pipeline (read frame, detect landmarks, compute angles, draw the view),
   using the webcam, or the first clip if no camera is available.
2. Synthetic sequences (no camera needed): each stick-figure reference pose
   held steady with a little landmark jitter, scored against every pose,
   plus a user moving back and forth through each pose. The right pose
   should score high and hold; wrong poses and moving users should not.
3. Test clips: every video in eval_clips/ (or --clips DIR) scored against
   every pose. A clip's file name starts with the id of the pose it shows,
   e.g. ``tree_front.mp4`` or ``warrior2_me.mp4``; a name that starts with
   no pose id (e.g. ``wrong_slouch.mp4``) means no pose should accept it.

A pose is "accepted" when a steady hold was established during the attempt
(the same rule the app uses). A false positive is a wrong pose accepted.
Each attempt is run through the same Session code as the app, frame by
frame, for the pose's hold_seconds (short clips are looped).

Usage:
    python tools/evaluate.py                 # everything
    python tools/evaluate.py --clips DIR     # clips from another folder
    python tools/evaluate.py --no-speed      # skip the speed test
"""

from __future__ import annotations

import argparse
import ctypes
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from app.geometry import LANDMARK_INDEX, NUM_LANDMARKS, joint_angles  # noqa: E402
from app.session import Pose, PoseResult, Session, load_poses  # noqa: E402

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
from make_reference_images import FIGURES, W as FIGURE_W  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SYNTHETIC_FPS = 30
JITTER_PX = 1.5  # landmark noise in the synthetic sequences (torso is ~140 px)
MOVE_PERIOD_SECONDS = 3.0  # one full back-and-forth for the "moving user" sequences


# --- running an attempt ----------------------------------------------------------------

def run_attempt(pose: Pose, frames: Iterable[tuple[float, Optional[dict]]]) -> PoseResult:
    """Run one PERFORMING window through a Session and return its result.

    ``frames`` yields ``(t seconds from the start, angles or None)``; frames
    after the pose's ``hold_seconds`` are ignored.
    """
    session = Session([pose])
    session.start(0.0)
    session.skip(0.0)  # SHOW_REFERENCE -> COUNTDOWN
    session.skip(0.0)  # COUNTDOWN -> PERFORMING
    for t, angles in frames:
        if t >= pose.hold_seconds:
            break
        session.process_frame(t, angles)
    session.skip(pose.hold_seconds)  # -> SCORING -> FEEDBACK
    return session.last_result


# --- synthetic sequences --------------------------------------------------------------

def figure_landmarks(points: dict) -> np.ndarray:
    """(33, 3) landmark array from stick-figure keypoints, as the camera would see them.

    The figures are drawn mirrored, so x is flipped back. Landmarks the figure
    does not define get visibility 0.
    """
    lms = np.zeros((NUM_LANDMARKS, 3))
    for name, (x, y) in points.items():
        idx = LANDMARK_INDEX["nose" if name == "head" else name]
        lms[idx] = (FIGURE_W - 1 - x, y, 1.0)
    return lms


def steady_sequence(points: dict, seconds: float, rng: np.random.Generator):
    """Landmarks of a figure held still, with small random jitter, at SYNTHETIC_FPS."""
    base = figure_landmarks(points)
    for i in range(int(seconds * SYNTHETIC_FPS)):
        lms = base.copy()
        lms[:, :2] += rng.normal(0, JITTER_PX, (NUM_LANDMARKS, 2))
        yield i / SYNTHETIC_FPS, lms


def moving_sequence(start: dict, end: dict, seconds: float, rng: np.random.Generator):
    """Landmarks moving smoothly back and forth between two figures, reaching ``end`` once per period."""
    a, b = figure_landmarks(start), figure_landmarks(end)
    for i in range(int(seconds * SYNTHETIC_FPS)):
        t = i / SYNTHETIC_FPS
        s = 0.5 - 0.5 * math.cos(2 * math.pi * t / MOVE_PERIOD_SECONDS)
        lms = a + (b - a) * s
        lms[:, 2] = np.minimum(a[:, 2], b[:, 2])
        lms[:, :2] += rng.normal(0, JITTER_PX, (NUM_LANDMARKS, 2))
        yield t, lms


def angle_stream(pose: Pose, landmark_frames):
    """Turn ``(t, landmarks or None)`` into ``(t, angles or None)`` for ``pose``'s rules."""
    for t, lms in landmark_frames:
        yield t, None if lms is None else joint_angles(lms, pose.joints, config.VISIBILITY_THRESHOLD)


def synthetic_results(poses: list[Pose], seed: int = 0) -> tuple[dict, dict]:
    """Score every steady figure against every pose, and a moving user for every pose.

    Returns ``(steady, moving)``: ``steady[(figure_id, pose_id)]`` and
    ``moving[pose_id]`` are PoseResults. The moving user swings between
    Mountain and the pose (between Tree and Mountain for Mountain itself).
    """
    rng = np.random.default_rng(seed)
    steady, moving = {}, {}
    for figure_id in (p.id for p in poses if p.id in FIGURES):
        for pose in poses:
            frames = steady_sequence(FIGURES[figure_id], pose.hold_seconds, rng)
            steady[(figure_id, pose.id)] = run_attempt(pose, angle_stream(pose, frames))
    for pose in poses:
        if pose.id not in FIGURES:
            continue
        start = FIGURES["tree" if pose.id == "mountain" else "mountain"]
        frames = moving_sequence(start, FIGURES[pose.id], pose.hold_seconds, rng)
        moving[pose.id] = run_attempt(pose, angle_stream(pose, frames))
    return steady, moving


# --- clips ----------------------------------------------------------------------------

def clip_label(path: Path, pose_ids: Iterable[str]) -> Optional[str]:
    """The pose id a clip's file name starts with (longest match), or None."""
    stem = path.stem.lower()
    matches = [pid for pid in pose_ids if stem == pid or stem.startswith(pid + "_") or stem.startswith(pid + "-")]
    return max(matches, key=len) if matches else None


def clip_landmarks(path: Path, seconds: float, detector) -> list[tuple[float, Optional[np.ndarray]]]:
    """Detect landmarks on ``seconds`` worth of a video file (looped if shorter), timed by its own FPS."""
    from app.camera import Camera

    out = []
    with Camera(str(path)) as cam:
        fps = cam.file_fps() or config.TARGET_FPS
        for i in range(int(seconds * fps)):
            frame = cam.read()
            if frame is None:
                cam.rewind()
                frame = cam.read()
                if frame is None:
                    break
            t = i / fps
            out.append((t, detector.detect(frame, int(t * 1000))))
    return out


# --- speed ----------------------------------------------------------------------------

def power_state() -> str:
    """'plugged in', 'on battery (NN%)' or 'unknown' (Windows only)."""
    if sys.platform != "win32":
        return "unknown"

    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]

    s = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return "unknown"
    return {0: f"on battery ({s.percent}%)", 1: "plugged in"}.get(s.ac, "unknown")


def measure_speed(source, rules, frames: int = 150, warmup: int = 15) -> dict:
    """Time the worker pipeline (read, detect, angles, draw view) on ``frames`` frames.

    Video files are read as fast as possible (no pacing), so the FPS shows
    what the CPU can sustain; a webcam is limited by its own frame rate too.
    """
    from app.camera import Camera
    from app.gui import compose_view
    from app.pose_detector import PoseDetector

    inference, totals, found = [], [], 0
    with Camera(source) as cam, PoseDetector() as detector:
        n = 0
        start = None
        while n < warmup + frames:
            t0 = time.perf_counter()
            frame = cam.read()
            if frame is None:
                if cam.is_file:
                    cam.rewind()
                    continue
                raise RuntimeError("The camera stopped sending frames.")
            t1 = time.perf_counter()
            lms = detector.detect(frame, int(time.monotonic() * 1000))
            t2 = time.perf_counter()
            angles = joint_angles(lms, rules, config.VISIBILITY_THRESHOLD) if lms is not None else None
            compose_view(frame, lms, rules, angles, 0.0, 0.0)
            t3 = time.perf_counter()
            n += 1
            if n == warmup:
                start = t3
            elif n > warmup:
                inference.append((t2 - t1) * 1000)
                totals.append((t3 - t0) * 1000)
                found += lms is not None
        elapsed = time.perf_counter() - start
    return {
        "frames": frames,
        "fps": frames / elapsed,
        "inference_mean": statistics.fmean(inference),
        "inference_median": statistics.median(inference),
        "inference_p95": sorted(inference)[int(0.95 * (len(inference) - 1))],
        "pipeline_mean": statistics.fmean(totals),
        "person_found": found / frames,
    }


# --- report ---------------------------------------------------------------------------

def _cell(r: PoseResult) -> str:
    return f"{r.score:5.1f}{'*' if r.held else ' '}"


def print_matrix(title: str, rows: list[tuple[str, Optional[str], dict]], poses: list[Pose]) -> tuple[int, int, int, int]:
    """Print a score table (rows = inputs, columns = poses) and return
    (correct accepted, correct total, wrong accepted, wrong total)."""
    print(f"\n{title}")
    print("  score per pose; * = steady hold accepted; [ ] = the pose the input shows\n")
    width = max(len(name) for name, _, _ in rows) + 2
    print(" " * width + "".join(f"{p.id[:11]:>13s}" for p in poses))
    ok = ok_total = fp = fp_total = 0
    for name, label, results in rows:
        cells = []
        for p in poses:
            r = results[p.id]
            text = _cell(r)
            if p.id == label:
                ok_total += 1
                ok += r.held
                text = f"[{text}]"
            else:
                fp_total += 1
                fp += r.held
                text = f" {text} "
            cells.append(f"{text:>13s}")
        print(f"{name:<{width}s}" + "".join(cells))
    return ok, ok_total, fp, fp_total


def print_wrong_tips(rows: list[tuple[str, Optional[str], dict]], poses: list[Pose]) -> None:
    """How many tips each clearly wrong attempt received (the spec asks for at least 3)."""
    counts = [len(results[p.id].tips) for _, label, results in rows for p in poses if p.id != label]
    if counts:
        three = sum(c >= 3 for c in counts)
        print(f"  Wrong-pose attempts with 3 tips: {three}/{len(counts)} "
              f"(min {min(counts)}, mean {statistics.fmean(counts):.1f})")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report FPS, latency and scoring behaviour.")
    parser.add_argument("--source", default=str(config.CAMERA_INDEX),
                        help="camera index or video for the speed test (default: %(default)s; "
                             "falls back to the first clip)")
    parser.add_argument("--clips", type=Path, default=config.PROJECT_ROOT / "eval_clips",
                        help="folder of test clips (default: eval_clips/)")
    parser.add_argument("--frames", type=int, default=150, help="frames to time in the speed test")
    parser.add_argument("--no-speed", action="store_true", help="skip the speed test")
    args = parser.parse_args(argv)

    import mediapipe

    poses = load_poses()
    pose_ids = [p.id for p in poses]
    clips = sorted(p for p in args.clips.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS) \
        if args.clips.is_dir() else []

    print("Yoga Pose Coach evaluation")
    print(f"  CPU: {platform.processor() or platform.machine()}  |  power: {power_state()}")
    print(f"  Python {platform.python_version()}, MediaPipe {mediapipe.__version__}, "
          f"model {config.MODEL_PATH.name}")
    calibrated = [p.id for p in poses if p.calibrated]
    print(f"  Calibrated poses: {', '.join(calibrated) if calibrated else 'none (default targets)'}")

    # 1. Speed
    if not args.no_speed:
        from app.camera import CameraError

        source = int(args.source) if args.source.isdigit() else args.source
        print("\n1. Speed (read + detect + angles + draw, single thread)")
        try:
            stats = measure_speed(source, poses[0].joints, args.frames)
            used = f"camera {source}" if isinstance(source, int) else Path(source).name
        except CameraError:
            stats = None
            if clips:
                used = f"{clips[0].name} (no camera)"
                stats = measure_speed(str(clips[0]), poses[0].joints, args.frames)
        if stats is None:
            print("  skipped: no camera and no clips")
        else:
            print(f"  source: {used}, {stats['frames']} frames, person found in {stats['person_found']:.0%}")
            print(f"  average FPS:         {stats['fps']:6.1f}")
            print(f"  inference latency:   mean {stats['inference_mean']:.1f} ms, "
                  f"median {stats['inference_median']:.1f} ms, p95 {stats['inference_p95']:.1f} ms")
            print(f"  whole frame:         mean {stats['pipeline_mean']:.1f} ms")

    # 2. Synthetic
    steady, moving = synthetic_results(poses)
    rows = [(f"{fid} (steady)", fid, {p.id: steady[(fid, p.id)] for p in poses})
            for fid in pose_ids if fid in FIGURES]
    ok, ok_total, fp, fp_total = print_matrix("2. Synthetic sequences (stick figures + jitter)", rows, poses)
    print(f"\n  Correct pose accepted:        {ok}/{ok_total}")
    print(f"  Wrong pose accepted (FP):     {fp}/{fp_total}")
    print_wrong_tips(rows, poses)
    print(f"\n  Moving through the pose (every {MOVE_PERIOD_SECONDS:g} s), scored on that pose:")
    for pid, r in moving.items():
        print(f"    {pid:14s} score {r.score:5.1f}  hold accepted: {'YES' if r.held else 'no'}  "
              f"best hold {r.best_hold_seconds:.1f} s")
    moving_held = sum(r.held for r in moving.values())

    # 3. Clips
    if not clips:
        print(f"\n3. Test clips: none found in {args.clips}")
        clip_fp = 0
    else:
        from app.pose_detector import PoseDetector

        seconds = max(p.hold_seconds for p in poses)
        rows = []
        for clip in clips:
            with PoseDetector() as detector:
                lms_frames = clip_landmarks(clip, seconds, detector)
            results = {p.id: run_attempt(p, angle_stream(p, lms_frames)) for p in poses}
            rows.append((clip.name, clip_label(clip, pose_ids), results))
        ok, ok_total, clip_fp, fp_total = print_matrix(f"3. Test clips ({args.clips})", rows, poses)
        print(f"\n  Correct pose accepted:        {ok}/{ok_total}")
        print(f"  Wrong pose accepted (FP):     {clip_fp}/{fp_total}")
        print_wrong_tips(rows, poses)

    print(f"\nSummary: {fp} synthetic and {clip_fp} clip false positives; "
          f"{moving_held} of {len(moving)} moving users given a hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
