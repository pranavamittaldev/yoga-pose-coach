import math
import random

from app.scorer import JointRule, score_frame
from app.stability import HoldTracker

FPS = 30.0
RULES = [
    JointRule("knee", ("a", "b", "c"), 90.0, 10.0, 1.0, "", ""),
    JointRule("elbow", ("a", "b", "c"), 180.0, 10.0, 1.0, "", ""),
]


def tracker():
    return HoldTracker(window=15, score_threshold=60, variance_threshold=16,
                       grace_seconds=1.0, min_hold_seconds=1.0)


def run(tr, angle_fn, seconds, start=0.0):
    """Feed ``seconds`` of frames; angle_fn(t) returns (knee, elbow). Returns statuses."""
    out = []
    n = int(seconds * FPS)
    for i in range(n):
        t = start + i / FPS
        knee, elbow = angle_fn(t)
        out.append(tr.update(t, score_frame({"knee": knee, "elbow": elbow}, RULES)))
    return out


def steady(t):
    rng = random.Random(int(t * 1000))
    return 90 + rng.uniform(-2, 2), 180 + rng.uniform(-2, 2)  # small jitter


def wrong(t):
    return 30.0, 100.0  # far from target, but steady


def test_steady_pose_registers_hold():
    tr = tracker()
    statuses = run(tr, steady, 5.0)
    assert statuses[-1].in_pose
    assert statuses[-1].hold_seconds > 4.0
    assert any(s.should_score for s in statuses)


def test_frames_not_scored_during_first_second_of_hold():
    tr = tracker()
    statuses = run(tr, steady, 5.0)
    for s in statuses:
        if s.should_score:
            assert s.hold_seconds >= 1.0
    # window fills after 15 frames (0.5 s), then 1 s of hold before scoring
    first_scored = next(i for i, s in enumerate(statuses) if s.should_score)
    assert first_scored / FPS >= 1.4


def test_moving_pose_does_not_register_hold():
    # Sweeps the knee through the target angle (90) every 2 seconds.
    tr = tracker()
    # The window can look steady for a moment at each turnaround, but never for a
    # continuous second, so no hold is established and nothing is scored.
    statuses = run(tr, lambda t: (90 + 40 * math.sin(2 * math.pi * t / 2), 180.0), 10.0)
    assert not any(s.should_score for s in statuses)
    assert tr.best_hold_seconds == 0


def test_slow_sweep_gets_no_long_hold_or_high_score():
    # A slow sweep (6 s period) nearly stops at each turnaround, 50 deg off target.
    # It may register a short hold there, but never a long one or a high score.
    tr = tracker()
    statuses = run(tr, lambda t: (90 + 50 * math.sin(2 * math.pi * t / 6), 180.0), 12.0)
    assert tr.best_hold_seconds < 1.5
    scored = [s.window_score for s in statuses if s.should_score]
    assert all(score < 75 for score in scored)


def test_steady_but_wrong_pose_does_not_register_hold():
    tr = tracker()
    statuses = run(tr, wrong, 5.0)
    assert not any(s.in_pose for s in statuses)


def test_leaving_pose_beyond_grace_resets_hold():
    tr = tracker()
    run(tr, steady, 4.0)
    held_before = tr.best_hold_seconds
    out = run(tr, wrong, 2.0, start=4.0)  # leave for 2 s (> 1 s grace)
    assert out[-1].hold_seconds == 0
    back = run(tr, steady, 2.0, start=6.0)
    assert 0 < back[-1].hold_seconds < held_before
    assert tr.best_hold_seconds == held_before


def test_brief_exit_within_grace_keeps_hold():
    tr = tracker()
    run(tr, steady, 3.0)
    run(tr, wrong, 0.3, start=3.0)  # leave for 0.3 s (< 1 s grace)
    back = run(tr, steady, 2.0, start=3.3)
    assert back[-1].hold_seconds > 4.0


def test_missing_body_is_not_in_pose():
    tr = tracker()
    statuses = run(tr, lambda t: (None, None), 2.0)
    assert not any(s.in_pose for s in statuses)
