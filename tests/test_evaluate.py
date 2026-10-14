"""End-to-end scoring on synthetic landmark sequences (no camera): the full Session path."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import config  # noqa: E402
from evaluate import (FIGURES, angle_stream, clip_label, moving_sequence, run_attempt,  # noqa: E402
                      steady_sequence)
from make_reference_images import figure_angles  # noqa: E402
from app.session import load_poses  # noqa: E402


def _default_targets(pose):
    """The pose with targets measured off its stick figure, so the tests don't depend on calibration."""
    angles = figure_angles(FIGURES[pose.id], [{"name": j.name, "landmarks": j.landmarks} for j in pose.joints])
    joints = [dataclasses.replace(j, target_angle=angles[j.name]) for j in pose.joints]
    return dataclasses.replace(pose, joints=joints)


POSES = {p.id: _default_targets(p) for p in load_poses() if p.id in FIGURES}  # poses with a stick figure


@pytest.mark.parametrize("pose_id", list(POSES))
def test_steady_correct_pose_is_held_and_scores_high(pose_id):
    pose = POSES[pose_id]
    frames = steady_sequence(FIGURES[pose_id], pose.hold_seconds, np.random.default_rng(1))
    result = run_attempt(pose, angle_stream(pose, frames))
    assert result.held
    assert result.score >= 90


@pytest.mark.parametrize("pose_id", list(POSES))
def test_user_moving_through_the_pose_gets_no_hold(pose_id):
    pose = POSES[pose_id]
    start = FIGURES["tree" if pose_id == "mountain" else "mountain"]
    frames = moving_sequence(start, FIGURES[pose_id], pose.hold_seconds, np.random.default_rng(1))
    result = run_attempt(pose, angle_stream(pose, frames))
    assert not result.held
    assert result.score <= config.NO_HOLD_SCORE_CAP


def test_clearly_wrong_pose_is_not_held_and_gets_three_tips():
    pose = POSES["warrior2"]
    frames = steady_sequence(FIGURES["mountain"], pose.hold_seconds, np.random.default_rng(1))
    result = run_attempt(pose, angle_stream(pose, frames))
    assert not result.held
    assert len(result.tips) == 3


def test_clip_label_uses_longest_pose_id_prefix():
    ids = ["tree", "downward_dog", "warrior2"]
    assert clip_label(Path("downward_dog_side.mp4"), ids) == "downward_dog"
    assert clip_label(Path("Tree-front.MP4"), ids) == "tree"
    assert clip_label(Path("treehouse.mp4"), ids) is None
    assert clip_label(Path("wrong_slouch.mp4"), ids) is None
