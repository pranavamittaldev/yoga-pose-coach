"""Rolling-window hold detection.

A frame counts as "in pose" only when, over the last ``window`` frames, the
average frame score is high enough AND every joint angle is steady (low
variance). This stops a user from scoring well by moving through the pose.

Time is passed in by the caller, so the tracker is pure and testable without
a camera or a real clock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from app.scorer import FrameScore


@dataclass
class HoldStatus:
    """Result of feeding one frame to the HoldTracker."""

    in_pose: bool  # window is good enough and steady
    hold_seconds: float  # current continuous hold time
    best_hold_seconds: float  # longest hold so far
    should_score: bool  # in pose and held long enough for this frame to count
    window_score: Optional[float]  # mean frame score over the window (None if nothing measurable)
    max_variance: Optional[float]  # largest per-joint angle variance in the window


def window_max_variance(frames: list[FrameScore]) -> Optional[float]:
    """Largest per-joint angle variance (deg^2) across ``frames``.

    Each joint uses only the frames where it was measurable, and needs at
    least two values. Returns None if no joint has enough values.
    """
    per_joint: dict[str, list[float]] = {}
    for f in frames:
        for name, angle in f.angles.items():
            if angle is not None:
                per_joint.setdefault(name, []).append(angle)
    variances = [float(np.var(v)) for v in per_joint.values() if len(v) >= 2]
    return max(variances) if variances else None


class HoldTracker:
    """Track whether the user is holding a pose and for how long."""

    def __init__(
        self,
        window: int = config.STABILITY_WINDOW,
        score_threshold: float = config.STABILITY_SCORE_THRESHOLD,
        variance_threshold: float = config.VARIANCE_THRESHOLD,
        grace_seconds: float = config.REENTRY_GRACE_SECONDS,
        min_hold_seconds: float = config.MIN_HOLD_BEFORE_SCORING_SECONDS,
    ):
        self.window = window
        self.score_threshold = score_threshold
        self.variance_threshold = variance_threshold
        self.grace_seconds = grace_seconds
        self.min_hold_seconds = min_hold_seconds
        self.reset()

    def reset(self) -> None:
        """Clear the window and all hold timers."""
        self._frames: deque[FrameScore] = deque(maxlen=self.window)
        self._run_start: Optional[float] = None  # start of the current continuous in-pose run
        self._hold_start: Optional[float] = None  # set once a run lasts min_hold_seconds
        self._last_in_pose: Optional[float] = None
        self.best_hold_seconds = 0.0

    def _window_in_pose(self) -> tuple[bool, Optional[float], Optional[float]]:
        """Check the current window. Returns (in_pose, mean score, max variance)."""
        frames = list(self._frames)
        scores = [f.score for f in frames if f.score is not None]
        mean_score = sum(scores) / len(scores) if scores else None
        max_var = window_max_variance(frames)
        in_pose = (
            len(frames) == self.window
            and all(f.score is not None for f in frames)
            and mean_score is not None
            and mean_score >= self.score_threshold
            and max_var is not None
            and max_var <= self.variance_threshold
        )
        return in_pose, mean_score, max_var

    def update(self, t: float, frame: FrameScore) -> HoldStatus:
        """Add the frame observed at time ``t`` (seconds) and return the hold status.

        A hold is established only after ``min_hold_seconds`` of *continuous*
        in-pose frames, so brief accidental entries (including a moving user
        passing through the pose) are ignored. Once established, short exits
        (up to ``grace_seconds``) do not reset the hold; longer ones do.
        Frames are marked ``should_score`` only while in pose during an
        established hold.
        """
        self._frames.append(frame)
        in_pose, mean_score, max_var = self._window_in_pose()

        if in_pose:
            if self._run_start is None:
                self._run_start = t
            self._last_in_pose = t
            if self._hold_start is None and t - self._run_start >= self.min_hold_seconds:
                self._hold_start = self._run_start
        else:
            self._run_start = None
            if self._hold_start is not None and t - self._last_in_pose > self.grace_seconds:
                self._hold_start = None

        if self._hold_start is not None:
            hold = self._last_in_pose - self._hold_start
            self.best_hold_seconds = max(self.best_hold_seconds, hold)
        elif self._run_start is not None:
            hold = t - self._run_start  # still building up to an established hold
        else:
            hold = 0.0

        return HoldStatus(
            in_pose=in_pose,
            hold_seconds=hold,
            best_hold_seconds=self.best_hold_seconds,
            should_score=in_pose and self._hold_start is not None,
            window_score=mean_score,
            max_variance=max_var,
        )
