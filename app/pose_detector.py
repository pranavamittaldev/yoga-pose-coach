"""MediaPipe Pose wrapper (Tasks API) and skeleton drawing.

Inference runs locally on the CPU using the model file in ``models/``.
Run this module directly for a live skeleton preview:

    python -m app.pose_detector            # webcam
    python -m app.pose_detector photo.jpg  # still image
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config
from app.geometry import LANDMARK_INDEX, NUM_LANDMARKS

# Pairs of landmark names joined by a bone in the drawn skeleton (face omitted).
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"), ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
]
_DRAWN_POINTS = sorted({n for pair in SKELETON_CONNECTIONS for n in pair} | {"nose"})


class PoseDetector:
    """Detect the 33 pose landmarks of one person per frame.

    ``detect`` returns a (33, 3) array of (x_pixels, y_pixels, visibility),
    or None if no person was found.
    """

    def __init__(self, model_path=config.MODEL_PATH, video_mode: bool = True):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Download it from {config.MODEL_URL}"
            )
        self.video_mode = video_mode
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Optional[np.ndarray]:
        """Run pose detection on a BGR frame.

        In video mode, ``timestamp_ms`` must increase between calls; if it is
        omitted the current monotonic time is used.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video_mode:
            if timestamp_ms is None:
                timestamp_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
            self._last_ts = timestamp_ms
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        else:
            result = self._landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        out = np.zeros((NUM_LANDMARKS, 3), dtype=float)
        for i, lm in enumerate(lms[:NUM_LANDMARKS]):
            vis = lm.visibility if lm.visibility is not None else 0.0
            out[i] = (lm.x * w, lm.y * h, vis)
        return out

    def close(self) -> None:
        """Free the MediaPipe graph."""
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def mirror_landmarks(landmarks: np.ndarray, width: int) -> np.ndarray:
    """Flip landmark x coordinates to match a horizontally mirrored frame (for display only)."""
    out = landmarks.copy()
    out[:, 0] = width - 1 - out[:, 0]
    return out


def draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    visibility_threshold: float = config.VISIBILITY_THRESHOLD,
    color=(0, 220, 0),
) -> None:
    """Draw bones and joints on ``frame`` in place, skipping low-visibility landmarks."""
    def pt(name):
        x, y, v = landmarks[LANDMARK_INDEX[name]]
        return (int(round(x)), int(round(y))) if v >= visibility_threshold else None

    for a, b in SKELETON_CONNECTIONS:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)
    for name in _DRAWN_POINTS:
        p = pt(name)
        if p:
            cv2.circle(frame, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 5, color, 1, cv2.LINE_AA)


def _preview_image(path: str) -> None:
    """Detect and draw the skeleton on a still image."""
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Could not read image {path}")
    with PoseDetector(video_mode=False) as det:
        t0 = time.perf_counter()
        lms = det.detect(frame)
        ms = (time.perf_counter() - t0) * 1000
    print(f"Inference: {ms:.1f} ms; person found: {lms is not None}")
    if lms is not None:
        draw_skeleton(frame, lms)
    cv2.imshow("Pose preview (any key to close)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _preview_camera() -> None:
    """Live webcam preview with the skeleton and FPS. Press Q to quit."""
    from app.camera import Camera

    with Camera() as cam, PoseDetector() as det:
        fps = 0.0
        last = time.perf_counter()
        while True:
            frame = cam.read()
            if frame is None:
                print("Camera returned no frame.")
                break
            lms = det.detect(frame)
            view = cv2.flip(frame, 1)  # mirror for display only
            if lms is not None:
                draw_skeleton(view, mirror_landmarks(lms, frame.shape[1]))
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            cv2.putText(view, f"FPS {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Pose preview (Q to quit)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _preview_image(sys.argv[1])
    else:
        _preview_camera()
