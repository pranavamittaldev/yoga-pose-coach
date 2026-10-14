import json
import math
from datetime import datetime

import config
from app.geometry import LANDMARK_INDEX
from app.scorer import JointRule
from app.session import (NO_BODY_TIP, Pose, PoseResult, Session, State, load_poses, mistake_of, save_log,
                         write_poses_json)

FPS = 30.0


def make_pose(pid="p", hold=10.0):
    joints = [
        JointRule("knee", ("left_hip", "left_knee", "left_ankle"), 90.0, 10.0, 1.0, "knee low", "knee high"),
        JointRule("elbow", ("left_shoulder", "left_elbow", "left_wrist"), 180.0, 10.0, 1.0, "elbow low", "elbow high"),
    ]
    return Pose(pid, f"Pose {pid}", "Do it.", "poses/images/x.png", hold, joints)


def session(n_poses=1):
    return Session([make_pose(str(i)) for i in range(n_poses)], wall_clock=lambda: datetime(2026, 1, 1, 12, 0, 0))


def to_performing(s, t=0.0):
    s.start(t)
    s.skip(t)  # reference -> countdown
    s.skip(t)  # countdown -> performing
    assert s.state == State.PERFORMING
    return t


def perform(s, angle_fn, seconds, start=0.0):
    """Feed frames for ``seconds`` then let the window time out."""
    for i in range(int(seconds * FPS)):
        t = start + i / FPS
        s.process_frame(t, angle_fn(t))
        s.tick(t)
    s.tick(start + seconds + 0.01)


def test_timed_transitions():
    s = session()
    assert s.state == State.IDLE
    s.start(0.0)
    assert s.state == State.SHOW_REFERENCE
    s.tick(config.REFERENCE_DISPLAY_SECONDS - 0.1)
    assert s.state == State.SHOW_REFERENCE
    s.tick(config.REFERENCE_DISPLAY_SECONDS + 0.01)
    assert s.state == State.COUNTDOWN
    s.tick(config.REFERENCE_DISPLAY_SECONDS + config.COUNTDOWN_SECONDS + 0.02)
    assert s.state == State.PERFORMING


def test_space_skips_each_stage():
    s = session()
    s.skip(0.0)
    assert s.state == State.SHOW_REFERENCE
    s.skip(0.1)
    assert s.state == State.COUNTDOWN
    s.skip(0.2)
    assert s.state == State.PERFORMING
    s.skip(0.3)
    assert s.state == State.FEEDBACK
    s.skip(0.4)
    assert s.state == State.SUMMARY


def test_good_hold_scores_high():
    s = session()
    to_performing(s)
    perform(s, lambda t: {"knee": 91.0 + math.sin(t * 7), "elbow": 178.0}, 10.0)
    assert s.state == State.FEEDBACK
    r = s.last_result
    assert r.held
    assert r.score > 95
    assert r.best_hold_seconds > 7
    assert r.tips == []


def test_wrong_pose_gets_low_score_and_tips():
    s = session()
    to_performing(s)
    perform(s, lambda t: {"knee": 150.0, "elbow": 120.0}, 10.0)
    r = s.last_result
    assert not r.held
    assert r.score < 20
    assert r.tips == ["knee high", "elbow low"] or r.tips == ["elbow low", "knee high"]


def test_moving_user_gets_no_hold():
    s = session()
    to_performing(s)
    perform(s, lambda t: {"knee": 90 + 40 * math.sin(2 * math.pi * t / 2), "elbow": 180.0}, 10.0)
    r = s.last_result
    assert not r.held
    assert r.best_hold_seconds == 0
    assert r.score <= config.NO_HOLD_SCORE_CAP


def test_no_body_scores_zero_with_tip():
    s = session()
    to_performing(s)
    perform(s, lambda t: None, 10.0)
    r = s.last_result
    assert r.score == 0
    assert r.tips == [NO_BODY_TIP]


def test_sequence_reaches_summary_and_counts_tips():
    s = session(n_poses=3)
    s.start(0.0)
    t = 0.0
    for _ in range(3):
        s.skip(t)
        s.skip(t)
        perform(s, lambda _t: {"knee": 150.0, "elbow": 178.0}, 10.0, start=t)
        t += 11
        s.skip(t)  # feedback -> next
    assert s.state == State.SUMMARY
    summary = s.summary()
    assert len(summary.pose_scores) == 3
    assert summary.common_tips[0] == ("knee high", 3)


def test_restart_clears_results():
    s = session()
    to_performing(s)
    s.skip(1.0)
    assert s.results
    s.restart()
    assert s.state == State.IDLE and not s.results


def test_save_log(tmp_path):
    s = session()
    to_performing(s)
    perform(s, lambda t: {"knee": 90.0, "elbow": 180.0}, 10.0)
    s.skip(20.0)
    path = save_log(s, tmp_path)
    assert path.name.startswith("session_") and path.suffix == ".json"
    data = json.loads(path.read_text())
    assert data["poses"][0]["pose_id"] == "0"
    assert "joint_averages" in data["poses"][0]
    assert "tips" in data["poses"][0]
    assert data["started_at"] and data["ended_at"]
    assert save_log(s, tmp_path) != path  # same second -> distinct file


def test_shipped_poses_are_valid():
    poses = load_poses()
    assert [p.id for p in poses] == ["mountain", "chair", "tree", "goddess", "warrior2", "triangle",
                                     "downward_dog", "childs_pose", "bridge"]
    for p in poses:
        assert 4 <= len(p.joints) <= 8  # Chair needs elbow rules to tell it apart from Goddess and Tree
        assert isinstance(p.calibrated, bool)  # true once calibrated with tools/record_pose.py
        assert (config.PROJECT_ROOT / p.image).exists()
        for j in p.joints:
            assert all(name in LANDMARK_INDEX for name in j.landmarks)
            assert 0 <= j.target_angle <= 180


def test_write_poses_json_round_trip(tmp_path):
    data = json.loads(config.POSES_FILE.read_text(encoding="utf-8"))
    out = tmp_path / "poses.json"
    write_poses_json(data, out)
    assert json.loads(out.read_text(encoding="utf-8")) == data


def test_mistake_is_first_sentence_of_tip():
    assert mistake_of("Left knee is bent. Straighten your back leg.") == "Left knee is bent."
    assert mistake_of("Arms up.") == "Arms up."


def test_summary_merges_tips_that_name_the_same_mistake():
    s = session(n_poses=2)

    def result(pid, tips):
        return PoseResult(pid, f"Pose {pid}", 70.0, True, 5.0, 10.0, 300, 200, {}, tips, "t0", "t1")

    s.results = [result("0", ["Left knee is bent. Straighten your left leg."]),
                 result("1", ["Left knee is bent. Straighten your back leg.", "Arms are low. Raise them."])]
    summary = s.summary()
    assert summary.common_mistakes[0] == ("Left knee is bent.", 2)
    assert len(summary.common_tips) == 3  # the full tips are still counted separately
    assert s.to_log()["common_mistakes"][0] == {"mistake": "Left knee is bent.", "count": 2}
