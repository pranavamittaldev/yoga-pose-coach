"""Compare user joint angles with a reference pose and produce scores.

All functions here are pure: they take angles and rules and return numbers,
with no dependency on the camera, OpenCV or the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import config


@dataclass(frozen=True)
class JointRule:
    """One joint-angle rule from poses.json."""

    name: str
    landmarks: tuple[str, str, str]
    target_angle: float
    tolerance: float
    weight: float
    tip_low: str
    tip_high: str

    @classmethod
    def from_dict(cls, d: dict) -> "JointRule":
        """Build a rule from its JSON dictionary."""
        if len(d["landmarks"]) != 3:
            raise ValueError(f"Joint {d['name']!r} must have exactly 3 landmarks")
        return cls(
            name=d["name"],
            landmarks=tuple(d["landmarks"]),
            target_angle=float(d["target_angle"]),
            tolerance=float(d["tolerance"]),
            weight=float(d["weight"]),
            tip_low=d["tip_low"],
            tip_high=d["tip_high"],
        )


@dataclass
class FrameScore:
    """Scores for one frame.

    ``score`` is None when no joint was measurable. ``joint_scores`` and
    ``deviations`` only contain measurable joints; ``deviations`` are signed
    (user angle minus target: positive means too large).
    """

    score: Optional[float]
    angles: dict[str, Optional[float]] = field(default_factory=dict)
    joint_scores: dict[str, float] = field(default_factory=dict)
    deviations: dict[str, float] = field(default_factory=dict)


@dataclass
class JointSummary:
    """Per-joint averages over a set of frames."""

    avg_score: float
    avg_signed_deviation: float
    frames: int


def joint_score(user_angle: float, target_angle: float, tolerance: float, k: float = config.SCORE_K) -> float:
    """Score one joint from 0 to 100.

    Deviations within ``tolerance`` score 100; beyond it the score drops by
    ``k`` points per degree, never below 0.
    """
    deviation = abs(user_angle - target_angle)
    effective = max(0.0, deviation - tolerance)
    return max(0.0, 100.0 - effective * k)


def score_frame(angles: dict[str, Optional[float]], rules: Sequence[JointRule], k: float = config.SCORE_K) -> FrameScore:
    """Score one frame from its joint angles.

    The frame score is the weighted average of joint scores over measurable
    joints only (angle not None). Unmeasurable joints are excluded, not
    scored as 0. Returns a FrameScore with ``score=None`` if nothing was
    measurable.
    """
    joint_scores: dict[str, float] = {}
    deviations: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for rule in rules:
        angle = angles.get(rule.name)
        if angle is None:
            continue
        s = joint_score(angle, rule.target_angle, rule.tolerance, k)
        joint_scores[rule.name] = s
        deviations[rule.name] = angle - rule.target_angle
        weighted_sum += s * rule.weight
        total_weight += rule.weight
    score = weighted_sum / total_weight if total_weight > 0 else None
    return FrameScore(score=score, angles=dict(angles), joint_scores=joint_scores, deviations=deviations)


def pose_score(frames: Iterable[FrameScore]) -> Optional[float]:
    """Average frame score over frames that have a score, or None if there are none."""
    scores = [f.score for f in frames if f.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def summarize_joints(frames: Iterable[FrameScore]) -> dict[str, JointSummary]:
    """Per-joint average score and average signed deviation over ``frames``.

    Each joint is averaged only over the frames where it was measurable.
    A positive ``avg_signed_deviation`` means the angle was too large on
    average; negative means too small. Joints never measured are omitted.
    """
    score_sums: dict[str, float] = {}
    dev_sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for f in frames:
        for name, s in f.joint_scores.items():
            score_sums[name] = score_sums.get(name, 0.0) + s
            dev_sums[name] = dev_sums.get(name, 0.0) + f.deviations[name]
            counts[name] = counts.get(name, 0) + 1
    return {
        name: JointSummary(
            avg_score=score_sums[name] / n,
            avg_signed_deviation=dev_sums[name] / n,
            frames=n,
        )
        for name, n in counts.items()
    }
