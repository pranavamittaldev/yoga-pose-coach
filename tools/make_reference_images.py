"""Draw the stick-figure reference images in poses/images/.

Each figure is drawn from hand-placed 2D keypoints, as the user will see
themselves in the mirrored video (their right side on the right of the
screen). The script also prints the angle each joint rule in poses.json
makes in the figure, so the default target angles can match the picture.

Usage:
    python tools/make_reference_images.py          # draw images, print angles
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from app.geometry import angle_between  # noqa: E402

W, H = 400, 500
BG = (245, 241, 236)
LIMB = (90, 70, 50)
LEFT_ACCENT = (200, 120, 40)   # person's left limbs (blue-ish, BGR)
RIGHT_ACCENT = (60, 110, 220)  # person's right limbs (orange-ish, BGR)

# Keypoints in mirror view: the person's left side is on the viewer's left.
FIGURES: dict[str, dict] = {
    "mountain": {
        "head": (200, 70),
        "left_shoulder": (170, 120), "right_shoulder": (230, 120),
        "left_elbow": (160, 195), "right_elbow": (240, 195),
        "left_wrist": (156, 265), "right_wrist": (244, 265),
        "left_hip": (182, 260), "right_hip": (218, 260),
        "left_knee": (184, 360), "right_knee": (216, 360),
        "left_ankle": (186, 455), "right_ankle": (214, 455),
        "left_foot_index": (172, 468), "right_foot_index": (228, 468),
    },
    "tree": {
        "head": (200, 112),
        "left_shoulder": (170, 155), "right_shoulder": (230, 155),
        "left_elbow": (150, 85), "right_elbow": (250, 85),
        "left_wrist": (195, 28), "right_wrist": (205, 28),
        "left_hip": (182, 290), "right_hip": (218, 290),
        "left_knee": (108, 345), "right_knee": (218, 380),
        "left_ankle": (205, 345), "right_ankle": (218, 458),
        "left_foot_index": (212, 328), "right_foot_index": (232, 470),
    },
    "warrior2": {
        "head": (212, 105),
        "left_shoulder": (178, 150), "right_shoulder": (222, 150),
        "left_elbow": (110, 151), "right_elbow": (290, 151),
        "left_wrist": (42, 152), "right_wrist": (358, 152),
        "left_hip": (185, 285), "right_hip": (220, 285),
        "left_knee": (128, 370), "right_knee": (318, 312),
        "left_ankle": (72, 455), "right_ankle": (322, 455),
        "left_foot_index": (52, 462), "right_foot_index": (345, 468),
    },
    "downward_dog": {  # side view, head toward the viewer's left
        "head": (140, 385),
        "left_shoulder": (150, 330), "right_shoulder": (150, 330),
        "left_elbow": (110, 393), "right_elbow": (110, 393),
        "left_wrist": (70, 455), "right_wrist": (70, 455),
        "left_hip": (262, 185), "right_hip": (262, 185),
        "left_knee": (305, 315), "right_knee": (305, 315),
        "left_ankle": (352, 452), "right_ankle": (352, 452),
        "left_foot_index": (330, 468), "right_foot_index": (330, 468),
    },
    "triangle": {  # bending over the left leg, right arm up
        "head": (48, 286),
        "left_shoulder": (92, 308), "right_shoulder": (84, 270),
        "left_elbow": (93, 378), "right_elbow": (86, 200),
        "left_wrist": (96, 440), "right_wrist": (88, 128),
        "left_hip": (190, 290), "right_hip": (226, 294),
        "left_knee": (140, 372), "right_knee": (284, 374),
        "left_ankle": (92, 455), "right_ankle": (340, 455),
        "left_foot_index": (70, 462), "right_foot_index": (362, 465),
    },
    "chair": {  # side view, facing the viewer's left
        "head": (166, 167),
        "left_shoulder": (205, 195), "right_shoulder": (205, 195),
        "left_elbow": (175, 126), "right_elbow": (175, 126),
        "left_wrist": (145, 57), "right_wrist": (145, 57),
        "left_hip": (285, 310), "right_hip": (285, 310),
        "left_knee": (190, 375), "right_knee": (190, 375),
        "left_ankle": (240, 455), "right_ankle": (240, 455),
        "left_foot_index": (205, 468), "right_foot_index": (205, 468),
    },
    "goddess": {
        "head": (200, 140),
        "left_shoulder": (172, 185), "right_shoulder": (228, 185),
        "left_elbow": (105, 185), "right_elbow": (295, 185),
        "left_wrist": (105, 115), "right_wrist": (295, 115),
        "left_hip": (182, 320), "right_hip": (218, 320),
        "left_knee": (110, 340), "right_knee": (290, 340),
        "left_ankle": (112, 455), "right_ankle": (288, 455),
        "left_foot_index": (88, 466), "right_foot_index": (312, 466),
    },
    "childs_pose": {  # side view on the floor, head toward the viewer's left
        "head": (128, 446),
        "left_shoulder": (168, 432), "right_shoulder": (168, 432),
        "left_elbow": (95, 450), "right_elbow": (95, 450),
        "left_wrist": (25, 460), "right_wrist": (25, 460),
        "left_hip": (300, 405), "right_hip": (300, 405),
        "left_knee": (230, 455), "right_knee": (230, 455),
        "left_ankle": (330, 460), "right_ankle": (330, 460),
        "left_foot_index": (355, 466), "right_foot_index": (355, 466),
    },
    "bridge": {  # side view lying on the back, head toward the viewer's left
        "head": (64, 446),
        "left_shoulder": (110, 452), "right_shoulder": (110, 452),
        "left_elbow": (175, 458), "right_elbow": (175, 458),
        "left_wrist": (240, 462), "right_wrist": (240, 462),
        "left_hip": (225, 380), "right_hip": (225, 380),
        "left_knee": (315, 345), "right_knee": (315, 345),
        "left_ankle": (330, 458), "right_ankle": (330, 458),
        "left_foot_index": (355, 464), "right_foot_index": (355, 464),
    },
}

TORSO_BONES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
]
BONES = TORSO_BONES + [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot_index"),
]


def draw_figure(points: dict) -> np.ndarray:
    """Render one stick figure and return a BGR image."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    cv2.line(img, (20, 472), (W - 20, 472), (200, 195, 188), 2, cv2.LINE_AA)  # floor
    torso = np.array([points[k] for k in ("left_shoulder", "right_shoulder", "right_hip", "left_hip")], np.int32)
    cv2.fillPoly(img, [torso], (215, 205, 195), cv2.LINE_AA)
    for a, b in BONES:
        if (a, b) in TORSO_BONES:
            color = LIMB
        else:
            color = LEFT_ACCENT if a.startswith("left_") else RIGHT_ACCENT
        cv2.line(img, points[a], points[b], color, 12, cv2.LINE_AA)
    for name, p in points.items():
        if name != "head":
            cv2.circle(img, p, 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, points["head"], 24, LIMB, -1, cv2.LINE_AA)
    cv2.putText(img, "L", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LEFT_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(img, "R", (W - 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RIGHT_ACCENT, 2, cv2.LINE_AA)
    return img


def figure_angles(points: dict, joints: list[dict]) -> dict[str, float]:
    """Angle of each joint rule as drawn in the figure."""
    return {j["name"]: angle_between(*(points[n] for n in j["landmarks"])) for j in joints}


def main() -> None:
    with open(config.POSES_FILE, encoding="utf-8") as f:
        poses = json.load(f)["poses"]
    for pose in poses:
        points = FIGURES.get(pose["id"])
        if points is None:
            print(f"{pose['id']}: no figure defined, skipped")
            continue
        out = config.PROJECT_ROOT / pose["image"]
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), draw_figure(points))
        print(f"{pose['id']}: wrote {out.relative_to(config.PROJECT_ROOT)}")
        for name, angle in figure_angles(points, pose["joints"]).items():
            target = next(j["target_angle"] for j in pose["joints"] if j["name"] == name)
            flag = "" if abs(angle - target) <= 3 else "   <-- differs from target"
            print(f"    {name:16s} figure {angle:6.1f}   target {target:6.1f}{flag}")


if __name__ == "__main__":
    main()
