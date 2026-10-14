import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from app.session import ALL_POSES_ID, Session, load_poses, load_programs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from evaluate import FIGURES, angle_stream, run_attempt, steady_sequence  # noqa: E402
from make_reference_images import figure_angles  # noqa: E402

POSES = load_poses()


def _write(tmp_path, programs):
    path = tmp_path / "programs.json"
    path.write_text(json.dumps({"programs": programs}), encoding="utf-8")
    return path


def test_shipped_programs_load():
    programs = load_programs(POSES)
    assert [p.id for p in programs] == ["children", "men", "women", ALL_POSES_ID]
    for program in programs:
        assert program.name and len(program.poses) >= 4
        assert len({p.id for p in program.poses}) == len(program.poses)  # no pose twice
    assert [p.id for p in programs[-1].poses] == [p.id for p in POSES]


def test_children_get_short_holds_and_own_instructions():
    children = load_programs(POSES)[0]
    base = {p.id: p for p in POSES}
    for pose in children.poses:
        assert pose.hold_seconds == 6
        assert pose.instruction != base[pose.id].instruction
        for rule, base_rule in zip(pose.joints, base[pose.id].joints):
            assert rule == base_rule  # same rules: extra tolerance only added false positives


def test_overrides_do_not_change_the_base_poses(tmp_path):
    before = [(p.instruction, p.hold_seconds, [j.tolerance for j in p.joints]) for p in POSES]
    path = _write(tmp_path, [{"id": "x", "name": "X", "hold_seconds": 4, "extra_tolerance": 7,
                              "poses": ["tree", {"id": "mountain", "instruction": "Hi", "hold_seconds": 8}]}])
    program = load_programs(POSES, path)[0]
    assert [(p.id, p.hold_seconds) for p in program.poses] == [("tree", 4), ("mountain", 8)]
    assert program.poses[1].instruction == "Hi"
    tree = next(p for p in POSES if p.id == "tree")
    assert [j.tolerance for j in program.poses[0].joints] == [j.tolerance + 7 for j in tree.joints]
    assert [(p.instruction, p.hold_seconds, [j.tolerance for j in p.joints]) for p in POSES] == before


def test_unknown_pose_or_empty_program_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown pose"):
        load_programs(POSES, _write(tmp_path, [{"id": "x", "name": "X", "poses": ["lotus"]}]))
    with pytest.raises(ValueError, match="no poses"):
        load_programs(POSES, _write(tmp_path, [{"id": "x", "name": "X", "poses": []}]))


def test_session_log_records_the_program():
    program = load_programs(POSES)[1]
    s = Session(program.poses, wall_clock=lambda: datetime(2026, 1, 1), program_id=program.id)
    assert s.to_log()["program"] == "men"


@pytest.mark.parametrize("pose_id", [p.id for p in load_programs(POSES)[0].poses])
def test_children_can_hold_each_pose_within_the_short_window(pose_id):
    """6 s holds still leave time to establish a hold (1 s) and score it."""
    pose = next(p for p in load_programs(POSES)[0].poses if p.id == pose_id)
    # Targets from the stick figure, so the test holds after the user calibrates poses.json.
    angles = figure_angles(FIGURES[pose_id], [{"name": j.name, "landmarks": j.landmarks} for j in pose.joints])
    pose = dataclasses.replace(pose, joints=[dataclasses.replace(j, target_angle=angles[j.name]) for j in pose.joints])
    frames = steady_sequence(FIGURES[pose_id], pose.hold_seconds, np.random.default_rng(2))
    result = run_attempt(pose, angle_stream(pose, frames))
    assert result.held and result.score >= 90
    assert result.best_hold_seconds >= 4
