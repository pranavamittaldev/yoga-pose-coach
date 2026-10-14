"""Angle computation and landmark normalization.

Landmarks are represented as a NumPy array of shape (33, 3): one row per
MediaPipe Pose landmark, holding (x, y, visibility). x and y must be in the
same units on both axes (pixels), so that angles are not distorted by the
frame's aspect ratio. This module has no OpenCV or MediaPipe dependency.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

# MediaPipe Pose landmark order (33 points).
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
LANDMARK_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}
NUM_LANDMARKS = len(LANDMARK_NAMES)


def angle_between(a, b, c) -> float:
    """Return the angle ABC at vertex ``b`` in degrees (0-180).

    ``a``, ``b`` and ``c`` are 2D (or 3D) points. Returns NaN if either arm
    of the angle has zero length, since the angle is then undefined.
    """
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def is_visible(landmarks: np.ndarray, name: str, threshold: float) -> bool:
    """Return True if landmark ``name`` has visibility at or above ``threshold``."""
    return bool(landmarks[LANDMARK_INDEX[name], 2] >= threshold)


def _visible_midpoint(landmarks: np.ndarray, names: Iterable[str], threshold: float) -> Optional[np.ndarray]:
    """Mean (x, y) of the visible landmarks among ``names``, or None if none are visible."""
    points = [landmarks[LANDMARK_INDEX[n], :2] for n in names if is_visible(landmarks, n, threshold)]
    if not points:
        return None
    return np.mean(points, axis=0)


def normalize_landmarks(landmarks: np.ndarray, visibility_threshold: float = 0.5) -> Optional[np.ndarray]:
    """Center landmarks on the hip midpoint and scale by torso length.

    Torso length is the distance from the shoulder midpoint to the hip
    midpoint. If only one hip or shoulder is visible (e.g. a side view), that
    one is used as the midpoint. Visibility values are copied unchanged.

    Returns a new (33, 3) array, or None if the hips or shoulders are not
    visible or the torso has zero length.
    """
    hip_mid = _visible_midpoint(landmarks, ("left_hip", "right_hip"), visibility_threshold)
    shoulder_mid = _visible_midpoint(landmarks, ("left_shoulder", "right_shoulder"), visibility_threshold)
    if hip_mid is None or shoulder_mid is None:
        return None
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso == 0:
        return None
    out = np.array(landmarks, dtype=float, copy=True)
    out[:, :2] = (out[:, :2] - hip_mid) / torso
    return out


def joint_angles(landmarks: np.ndarray, rules, visibility_threshold: float = 0.5) -> dict[str, Optional[float]]:
    """Compute the angle for each joint rule.

    ``rules`` is an iterable of objects with ``name`` and ``landmarks``
    (three landmark names, the middle one being the vertex). The landmarks are
    normalized first when possible; angles are unaffected by normalization,
    so the raw landmarks are used if it is not possible.

    Returns ``{rule name: angle in degrees}``, with None for a rule whose
    landmarks are not all visible ("not measurable" for this frame).
    """
    normalized = normalize_landmarks(landmarks, visibility_threshold)
    points = normalized if normalized is not None else landmarks
    angles: dict[str, Optional[float]] = {}
    for rule in rules:
        if not all(is_visible(landmarks, n, visibility_threshold) for n in rule.landmarks):
            angles[rule.name] = None
            continue
        a, b, c = (points[LANDMARK_INDEX[n], :2] for n in rule.landmarks)
        angle = angle_between(a, b, c)
        angles[rule.name] = None if math.isnan(angle) else angle
    return angles
