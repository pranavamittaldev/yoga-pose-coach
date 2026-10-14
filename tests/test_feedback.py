from app.feedback import select_tips
from app.scorer import JointRule, JointSummary


def rule(name):
    return JointRule(name, ("a", "b", "c"), 90.0, 10.0, 1.0,
                     f"{name}: angle too small", f"{name}: angle too large")


RULES = [rule(n) for n in ("left_knee", "right_knee", "left_elbow", "right_elbow", "hip")]


def test_negative_deviation_uses_tip_low():
    tips = select_tips({"left_knee": JointSummary(40.0, -40.0, 30)}, RULES)
    assert [t.text for t in tips] == ["left_knee: angle too small"]


def test_positive_deviation_uses_tip_high():
    tips = select_tips({"left_knee": JointSummary(40.0, 40.0, 30)}, RULES)
    assert [t.text for t in tips] == ["left_knee: angle too large"]


def test_at_most_three_tips_ordered_by_severity():
    summaries = {
        "left_knee": JointSummary(50.0, -35.0, 30),
        "right_knee": JointSummary(10.0, 55.0, 30),
        "left_elbow": JointSummary(30.0, -45.0, 30),
        "right_elbow": JointSummary(70.0, 25.0, 30),
        "hip": JointSummary(20.0, 50.0, 30),
    }
    tips = select_tips(summaries, RULES)
    assert len(tips) == 3
    assert [t.joint for t in tips] == ["right_knee", "hip", "left_elbow"]


def test_small_deviations_and_perfect_joints_are_ignored():
    summaries = {
        "left_knee": JointSummary(95.0, 5.0, 30),  # below min deviation
        "right_knee": JointSummary(100.0, 12.0, 30),  # within tolerance, no points lost
        "hip": JointSummary(60.0, -30.0, 30),
    }
    tips = select_tips(summaries, RULES, min_deviation=10)
    assert [t.joint for t in tips] == ["hip"]


def test_no_tips_for_good_attempt():
    assert select_tips({"hip": JointSummary(100.0, 1.0, 30)}, RULES) == []


def test_identical_tip_text_is_shown_once():
    same = [JointRule(n, ("a", "b", "c"), 90.0, 10.0, 1.0, "Shoulders are hunched.", "x") for n in ("ls", "rs")]
    other = JointRule("hip", ("a", "b", "c"), 90.0, 10.0, 1.0, "Hips too low.", "y")
    summaries = {
        "ls": JointSummary(10.0, -50.0, 30),
        "rs": JointSummary(12.0, -48.0, 30),
        "hip": JointSummary(40.0, -30.0, 30),
    }
    tips = select_tips(summaries, same + [other])
    assert [t.text for t in tips] == ["Shoulders are hunched.", "Hips too low."]
