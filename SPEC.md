> The original project spec, as given at the start (14 Oct 2026). Kept unchanged for reference.
> Later changes agreed with the user (extra poses, decisions) are in README.md and CLAUDE.md.

# Project: Yoga Pose Coach

Build a Python desktop application that uses the laptop webcam to coach the user through a yoga sequence. For each pose, the app shows the target pose, waits for the user to get into position, captures a timed hold, scores how closely the user's body matches the reference, and gives feedback on which joints are off. All processing must run locally. No video or images may be sent over the network.

## Goals
1. Show a reference pose (image plus name and a short instruction) before the user attempts it.
2. Run a preparation countdown, then a timed performance window (default 10 seconds).
3. Detect body landmarks in real time from the webcam.
4. Compare the user's pose to the reference using joint angles, not raw pixel or coordinate positions.
5. Produce an accuracy percentage (0-100) for each pose and an overall session score.
6. Identify the joints that deviated most and show a plain-language correction tip for each (e.g., "Left knee bent too much. Straighten it.").
7. Show a session summary at the end with per-pose scores, the hold time achieved, and the most common mistakes.

## Tech stack
- Python 3.10+
- OpenCV for webcam capture and drawing
- MediaPipe Pose (Tasks API or legacy solutions API, whichever installs cleanly) for 33 landmarks per frame
- NumPy for angle and vector math
- Tkinter for the GUI (a single window with a video panel, reference image panel, and status text)
- JSON for pose definitions and session logs
- pytest for unit tests

Do not use cloud APIs. Do not add heavy frameworks unless they are needed.

## Project structure
```
yoga-pose-coach/
├── README.md
├── requirements.txt
├── main.py                  # Entry point: launches the GUI
├── config.py                # Tunable constants (thresholds, durations, weights)
├── app/
│   ├── __init__.py
│   ├── camera.py            # Webcam capture wrapper
│   ├── pose_detector.py     # MediaPipe wrapper, returns landmarks per frame
│   ├── geometry.py          # Angle computation, normalization helpers
│   ├── scorer.py            # Compares user angles to reference, produces scores
│   ├── stability.py         # Rolling-window hold detection and variance check
│   ├── feedback.py          # Maps joint deviations to plain-language tips
│   ├── session.py           # Sequence state machine and session logging
│   └── gui.py               # Tkinter interface
├── poses/
│   ├── poses.json           # Reference pose definitions (see schema below)
│   └── images/              # Reference images, one per pose
├── tools/
│   └── record_pose.py       # Utility: record a reference pose from the webcam
├── tests/
│   ├── test_geometry.py
│   ├── test_scorer.py
│   ├── test_stability.py
│   └── test_feedback.py
└── logs/                    # Session JSON logs (gitignored)
```

## Pose definition schema (poses/poses.json)
Each pose is an entry with:
- `id`: short slug, e.g. `"tree"`
- `name`: display name, e.g. `"Tree Pose"`
- `instruction`: one or two sentences shown before the attempt
- `image`: path to the reference image
- `hold_seconds`: target hold duration (default 10)
- `joints`: a list of joint-angle rules, each with:
  - `name`: e.g. `"left_knee"`
  - `landmarks`: three MediaPipe landmark names forming the angle, e.g. `["left_hip", "left_knee", "left_ankle"]` (the middle one is the vertex)
  - `target_angle`: degrees
  - `tolerance`: degrees of acceptable deviation before penalty starts
  - `weight`: relative importance (float)
  - `tip_low` / `tip_high`: feedback strings for when the user's angle is too small or too large

Start with these poses, each with 4-6 joint rules: Mountain (Tadasana), Tree (Vrikshasana), Warrior II (Virabhadrasana II), Downward Dog (Adho Mukha Svanasana), and Triangle (Trikonasana). Use reasonable initial target angles and mark them clearly in the JSON with a `"calibrated": false` flag so they can be replaced later with recorded values.

## Core algorithms

### Geometry (app/geometry.py)
- `angle_between(a, b, c)`: returns the angle at vertex b in degrees, using `arccos` of the normalized dot product. Clamp the input to [-1, 1] to avoid NaN.
- `normalize_landmarks(landmarks)`: centers on the hip midpoint and scales by torso length (shoulder midpoint to hip midpoint). This makes angle-based comparison independent of distance from the camera.
- Ignore landmarks with visibility below a configurable threshold (default 0.5). If a joint required for a rule is not visible, mark that rule as "not measurable" for that frame and exclude it from scoring. Do not score it as 0.

### Scoring (app/scorer.py)
For each joint rule on each frame:
- `deviation = abs(user_angle - target_angle)`
- `effective_deviation = max(0, deviation - tolerance)`
- `joint_score = max(0, 100 - effective_deviation * k)`, where `k` is a constant in config (start with 2.0 points per degree).
- Frame score = weighted average of joint scores over measurable joints only.
- Pose score = average of frame scores across the hold window, computed only over frames that pass the stability check.

Expose a function that returns the per-joint average score and the per-joint average signed deviation (positive means too large, negative means too small). The signed value drives the feedback tip choice.

### Stability and hold detection (app/stability.py)
- Keep a rolling window of the last N frames (default 15, about 0.5 seconds at 30 FPS).
- A frame counts as "in pose" only if the average joint score is above a configurable threshold (default 60) AND the variance of key joint angles across the window is below a threshold. This prevents a moving user from scoring well by passing through the pose.
- Track the continuous hold time. If the user leaves the pose for more than 1 second, reset the hold timer and let them re-enter. Keep the best hold time in the session log.
- Do not score frames until the user has been in pose for at least 1 second, so accidental entries are ignored.

### Feedback (app/feedback.py)
- After a pose, take the 2-3 joints with the lowest average score (above a minimum deviation so tiny errors are ignored).
- For each, pick `tip_low` if the signed deviation is negative, `tip_high` if positive.
- Return a short list of tips, ordered by severity.
- Do not show more than 3 tips per pose.

### Session flow (app/session.py)
State machine with these states:
`IDLE → SHOW_REFERENCE → COUNTDOWN → PERFORMING → SCORING → FEEDBACK → (next pose or) SUMMARY`

- SHOW_REFERENCE: display the image and instruction for 6 seconds, or until the user presses Space.
- COUNTDOWN: 3-second countdown.
- PERFORMING: capture frames for `hold_seconds`. If the user has not achieved a stable hold by the end, record the best partial score and move on.
- SCORING and FEEDBACK: show the accuracy percentage, the tips, and a "Next" button or auto-advance after 5 seconds.
- SUMMARY: total score, per-pose scores, longest hold, and the three most frequent tips across the session.

Save each completed session to `logs/session_YYYYMMDD_HHMMSS.json` with timestamps, per-pose scores, joint averages, and tips shown.

## GUI requirements (app/gui.py)
- One window, roughly 1100x700. Left: live camera feed with the skeleton drawn over it, plus a small overlay of the current joint angles. Right: reference image, pose name, instruction, and current state.
- A large progress bar or timer for the hold. A live accuracy meter that updates during performance.
- Keyboard shortcuts: Space to start or skip a stage, Q to quit, R to restart the sequence.
- The UI must not freeze. Run camera capture and inference in a worker thread and pass frames to the GUI through a thread-safe queue. Keep the UI refresh at about 30 FPS.
- Show FPS in a corner of the video panel.

## Calibration tool (tools/record_pose.py)
Write a script that:
1. Opens the webcam and shows the skeleton.
2. Waits for the user to press Space to begin a 5-second recording.
3. Averages each joint angle over the recording (ignoring frames where landmarks are not visible).
4. Prints the averaged angles and writes them into `poses/poses.json` for the chosen pose, setting `"calibrated": true`.

This lets the user replace the default target angles with their own.

## Configuration (config.py)
Put all tunable values here, not scattered in code:
- `CAMERA_INDEX`, `FRAME_WIDTH`, `FRAME_HEIGHT`, `TARGET_FPS`
- `VISIBILITY_THRESHOLD` (0.5)
- `SCORE_K` (2.0 points per degree)
- `STABILITY_WINDOW` (15 frames)
- `STABILITY_SCORE_THRESHOLD` (60)
- `VARIANCE_THRESHOLD`
- `REENTRY_GRACE_SECONDS` (1.0)
- `COUNTDOWN_SECONDS` (3)
- `REFERENCE_DISPLAY_SECONDS` (6)
- `DEFAULT_HOLD_SECONDS` (10)
- `MIN_TIP_DEVIATION_DEGREES` (10)

## Testing requirements
- `test_geometry.py`: angle of a known right angle is 90°, angle of a straight line is 180°, normalization makes results scale-invariant, and missing or low-visibility landmarks are handled.
- `test_scorer.py`: a perfect match scores 100, a deviation within tolerance scores 100, a deviation beyond tolerance decreases the score linearly, and unmeasurable joints are excluded from the weighted average.
- `test_stability.py`: a steady pose registers a hold, a moving pose does not, and leaving the pose for more than the grace period resets the hold.
- `test_feedback.py`: the correct tip is chosen for negative and positive deviations, and at most 3 tips are returned.

Scoring and stability logic must be pure functions that do not depend on OpenCV or the GUI, so they can be tested without a camera.

## Deliverables and milestones
Work in this order and confirm each step runs before moving on:
1. Project skeleton, requirements.txt, and config.py.
2. Camera capture and MediaPipe landmark extraction, with the skeleton drawn on screen (no scoring yet).
3. geometry.py with tests passing.
4. scorer.py and stability.py with tests passing.
5. feedback.py with tests passing.
6. Session state machine and poses.json loaded for all five poses.
7. Tkinter GUI wired to the session, running without freezing.
8. Session logging and the summary screen.
9. tools/record_pose.py for calibration.
10. README with setup instructions, how to run, how to add a new pose, and how to calibrate.

After each milestone, summarize what was built and what tests pass. Do not claim a feature works unless you have run it or its tests.

## Evaluation (add to README)
Include a script or section that reports:
- Average FPS and per-frame inference latency on the laptop CPU.
- Scoring behavior on a small set of test clips or recordings (correct poses vs. clearly wrong poses), to show false positives are low.

## Acceptance criteria
- The app runs on a laptop with only a webcam, with no internet connection required after installation.
- Each pose displays its reference before the attempt.
- The accuracy percentage appears after each pose.
- At least 3 specific, correct joint tips appear for a visibly wrong attempt.
- A moving user does not receive a high hold score.
- No video is written to disk. Only the JSON session logs are saved.

## Constraints
- Keep the code simple and readable, with docstrings on public functions.
- Do not add features beyond this spec without asking first.
- Where the spec is ambiguous, choose the simplest option and note the decision in the README.
- Start by reading this spec, then propose the file layout and a short plan before writing code.
