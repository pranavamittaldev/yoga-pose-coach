import pytest

from app.scorer import FrameScore, JointRule, joint_score, pose_score, score_frame, summarize_joints


def rule(name, target, tol=10.0, weight=1.0):
    return JointRule(name, ("a", "b", "c"), target, tol, weight, f"{name} low", f"{name} high")


RULES = [rule("knee", 180), rule("elbow", 90, weight=3.0)]


def test_perfect_match_scores_100():
    assert joint_score(180, 180, 10) == 100
    assert score_frame({"knee": 180.0, "elbow": 90.0}, RULES).score == pytest.approx(100)


def test_within_tolerance_scores_100():
    assert joint_score(171, 180, 10) == 100
    assert joint_score(190, 180, 10) == 100


def test_beyond_tolerance_decreases_linearly():
    k = 2.0
    assert joint_score(160, 180, 10, k) == pytest.approx(80)  # 10 deg over -> -20
    assert joint_score(150, 180, 10, k) == pytest.approx(60)  # 20 deg over -> -40
    assert joint_score(210, 180, 10, k) == pytest.approx(60)  # symmetric
    assert joint_score(10, 180, 10, k) == 0  # floored at 0


def test_weighted_average_of_joints():
    # knee 80 (weight 1), elbow 100 (weight 3) -> (80 + 300) / 4 = 95
    fs = score_frame({"knee": 160.0, "elbow": 90.0}, RULES)
    assert fs.score == pytest.approx(95)


def test_unmeasurable_joints_are_excluded_not_zeroed():
    fs = score_frame({"knee": None, "elbow": 90.0}, RULES)
    assert fs.score == pytest.approx(100)
    assert "knee" not in fs.joint_scores


def test_nothing_measurable_gives_none():
    assert score_frame({"knee": None}, RULES).score is None
    assert score_frame({}, RULES).score is None


def test_signed_deviation():
    fs = score_frame({"knee": 150.0, "elbow": 120.0}, RULES)
    assert fs.deviations["knee"] == pytest.approx(-30)  # too small
    assert fs.deviations["elbow"] == pytest.approx(30)  # too large


def test_pose_score_ignores_unscored_frames():
    frames = [FrameScore(80.0), FrameScore(None), FrameScore(100.0)]
    assert pose_score(frames) == pytest.approx(90)
    assert pose_score([FrameScore(None)]) is None


def test_summarize_joints_averages_only_measured_frames():
    frames = [
        score_frame({"knee": 150.0, "elbow": 90.0}, RULES),
        score_frame({"knee": 170.0, "elbow": None}, RULES),
    ]
    s = summarize_joints(frames)
    assert s["knee"].frames == 2
    assert s["knee"].avg_signed_deviation == pytest.approx(-20)
    assert s["knee"].avg_score == pytest.approx((60 + 100) / 2)
    assert s["elbow"].frames == 1


def test_rule_from_dict_requires_three_landmarks():
    with pytest.raises(ValueError):
        JointRule.from_dict({"name": "x", "landmarks": ["a", "b"], "target_angle": 1, "tolerance": 1,
                             "weight": 1, "tip_low": "", "tip_high": ""})
