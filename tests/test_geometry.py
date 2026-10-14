from types import SimpleNamespace

import numpy as np
import pytest

from app.geometry import LANDMARK_INDEX, NUM_LANDMARKS, angle_between, joint_angles, normalize_landmarks


def make_landmarks(points: dict, visibility: float = 1.0) -> np.ndarray:
    """Build a (33, 3) landmark array; unspecified landmarks are invisible."""
    lms = np.zeros((NUM_LANDMARKS, 3))
    for name, (x, y) in points.items():
        lms[LANDMARK_INDEX[name]] = (x, y, visibility)
    return lms


BODY = {
    "left_shoulder": (110, 100), "right_shoulder": (90, 100),
    "left_hip": (108, 200), "right_hip": (92, 200),
    "left_knee": (108, 300), "left_ankle": (108, 400),
    "right_knee": (150, 250), "right_ankle": (110, 260),
    "left_elbow": (160, 100), "left_wrist": (160, 50),
}
RULES = [
    SimpleNamespace(name="left_knee", landmarks=("left_hip", "left_knee", "left_ankle")),
    SimpleNamespace(name="right_knee", landmarks=("right_hip", "right_knee", "right_ankle")),
    SimpleNamespace(name="left_elbow", landmarks=("left_shoulder", "left_elbow", "left_wrist")),
]


def test_right_angle_is_90():
    assert angle_between((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)


def test_straight_line_is_180():
    assert angle_between((-1, 0), (0, 0), (2, 0)) == pytest.approx(180.0)


def test_collinear_same_direction_is_0_and_no_nan():
    # Floating point can push the cosine slightly past 1; clamping avoids NaN.
    a = angle_between((0.1, 0.1), (0, 0), (0.3, 0.3))
    assert a == pytest.approx(0.0, abs=1e-5)


def test_degenerate_angle_is_nan():
    assert np.isnan(angle_between((0, 0), (0, 0), (1, 1)))


def test_normalize_centers_on_hips_and_scales_by_torso():
    norm = normalize_landmarks(make_landmarks(BODY))
    hip_mid = (norm[LANDMARK_INDEX["left_hip"], :2] + norm[LANDMARK_INDEX["right_hip"], :2]) / 2
    sh_mid = (norm[LANDMARK_INDEX["left_shoulder"], :2] + norm[LANDMARK_INDEX["right_shoulder"], :2]) / 2
    assert hip_mid == pytest.approx([0, 0])
    assert np.linalg.norm(sh_mid - hip_mid) == pytest.approx(1.0)


def test_normalization_is_scale_and_translation_invariant():
    small = make_landmarks(BODY)
    big = make_landmarks({k: (3 * x + 500, 3 * y - 40) for k, (x, y) in BODY.items()})
    n_small, n_big = normalize_landmarks(small), normalize_landmarks(big)
    placed = [LANDMARK_INDEX[k] for k in BODY]
    assert n_small[placed, :2] == pytest.approx(n_big[placed, :2])
    a_small, a_big = joint_angles(small, RULES), joint_angles(big, RULES)
    for name in a_small:
        assert a_small[name] == pytest.approx(a_big[name])


def test_joint_angles_values():
    angles = joint_angles(make_landmarks(BODY), RULES)
    assert angles["left_knee"] == pytest.approx(180.0)
    assert angles["left_elbow"] == pytest.approx(90.0)


def test_low_visibility_landmark_makes_rule_unmeasurable():
    lms = make_landmarks(BODY)
    lms[LANDMARK_INDEX["right_ankle"], 2] = 0.3  # below the 0.5 default
    angles = joint_angles(lms, RULES)
    assert angles["right_knee"] is None
    assert angles["left_knee"] == pytest.approx(180.0)


def test_missing_landmark_makes_rule_unmeasurable():
    body = {k: v for k, v in BODY.items() if k != "left_wrist"}
    angles = joint_angles(make_landmarks(body), RULES)
    assert angles["left_elbow"] is None


def test_normalize_returns_none_without_hips():
    body = {k: v for k, v in BODY.items() if "hip" not in k}
    assert normalize_landmarks(make_landmarks(body)) is None


def test_normalize_uses_single_side_when_other_hidden():
    lms = make_landmarks(BODY)
    lms[LANDMARK_INDEX["right_hip"], 2] = 0.1
    lms[LANDMARK_INDEX["right_shoulder"], 2] = 0.1
    norm = normalize_landmarks(lms)
    assert norm[LANDMARK_INDEX["left_hip"], :2] == pytest.approx([0, 0])


def test_angles_still_computed_when_normalization_impossible():
    lms = make_landmarks({"left_shoulder": (0, 0), "left_elbow": (10, 0), "left_wrist": (10, 10)})
    assert joint_angles(lms, RULES)["left_elbow"] == pytest.approx(90.0)
