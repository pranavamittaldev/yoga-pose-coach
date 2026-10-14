import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from record_pose import apply_calibration, average_angles  # noqa: E402


def _data():
    joints = [{"name": n, "landmarks": ["a", "b", "c"], "target_angle": 90, "tolerance": 10, "weight": 1.0,
               "tip_low": "low", "tip_high": "high"} for n in ("knee", "hip")]
    return {"poses": [{"id": "tree", "calibrated": False, "joints": joints}]}


def test_average_ignores_frames_where_joint_not_visible():
    frames = [{"knee": 170.0, "hip": None}, {"knee": 180.0, "hip": 100.0}, {}]
    avg = average_angles(frames, ["knee", "hip", "elbow"])
    assert avg["knee"] == {"angle": 175.0, "frames": 2}
    assert avg["hip"] == {"angle": 100.0, "frames": 1}
    assert avg["elbow"] == {"angle": None, "frames": 0}


def test_calibration_writes_targets_and_sets_flag():
    data = _data()
    changes = apply_calibration(data, "tree", {"knee": {"angle": 172.36, "frames": 50},
                                               "hip": {"angle": None, "frames": 0}})
    pose = data["poses"][0]
    assert pose["calibrated"] is True
    assert pose["joints"][0]["target_angle"] == 172.4
    assert pose["joints"][1]["target_angle"] == 90  # never visible: old target kept
    assert changes == [("knee", 90, 172.4), ("hip", 90, None)]


def test_calibration_rejects_unknown_pose_and_empty_recording():
    with pytest.raises(ValueError):
        apply_calibration(_data(), "nope", {"knee": {"angle": 170.0, "frames": 1}})
    data = _data()
    with pytest.raises(ValueError):
        apply_calibration(data, "tree", {"knee": {"angle": None, "frames": 0}})
    assert data["poses"][0]["calibrated"] is False
