"""Tunable constants for Yoga Pose Coach. Change values here, not in the code."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# --- Pose detection ---
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
VISIBILITY_THRESHOLD = 0.5  # landmarks below this are treated as missing

# --- Scoring ---
SCORE_K = 2.0  # points lost per degree beyond tolerance

# --- Stability / hold detection ---
STABILITY_WINDOW = 15  # frames (~0.5 s at 30 FPS)
STABILITY_SCORE_THRESHOLD = 60.0  # min average joint score to count as "in pose"
VARIANCE_THRESHOLD = 16.0  # max per-joint angle variance in the window (deg^2, i.e. 4 deg std)
REENTRY_GRACE_SECONDS = 1.0  # leaving the pose longer than this resets the hold
MIN_HOLD_BEFORE_SCORING_SECONDS = 1.0  # frames are only scored after this much continuous hold
NO_HOLD_SCORE_CAP = 50.0  # max pose score when no steady hold was achieved (partial score)

# --- Session timing ---
COUNTDOWN_SECONDS = 3
REFERENCE_DISPLAY_SECONDS = 6
DEFAULT_HOLD_SECONDS = 10
FEEDBACK_DISPLAY_SECONDS = 5

# --- Feedback ---
MIN_TIP_DEVIATION_DEGREES = 10.0
MAX_TIPS = 3

# --- Calibration (tools/record_pose.py) ---
RECORD_SECONDS = 5  # length of a reference recording

# --- Files ---
POSES_FILE = PROJECT_ROOT / "poses" / "poses.json"
PROGRAMS_FILE = PROJECT_ROOT / "poses" / "programs.json"
LOGS_DIR = PROJECT_ROOT / "logs"
