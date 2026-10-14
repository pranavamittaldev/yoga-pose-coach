# Yoga Pose Coach

A desktop app that uses your laptop webcam to coach you through a short yoga sequence.
For each pose it shows a reference picture and instruction, counts you in, times your
hold, scores how closely your joint angles match the reference (0-100%), and tells you
which joints to fix. At the end you get a session summary.

Everything runs locally on the CPU. **No video or images leave your computer, and no
video is ever written to disk.** The only files the app saves are small JSON session logs.

### Programs

The app opens on a program picker. Press a number key or click a program:

| Key | Program | Poses | Holds | Length |
|---|---|---|---|---|
| 1 | **Children**: playful poses, short holds | Mountain, Tree, Chair, Warrior II, Downward Dog, Child's Pose | 6 s | ~2 min |
| 2 | **Men**: leg strength and flexibility | Mountain, Chair, Warrior II, Triangle, Downward Dog, Bridge, Child's Pose | 10 s | ~3 min |
| 3 | **Women**: balance and hip mobility | Mountain, Tree, Goddess, Warrior II, Triangle, Bridge, Child's Pose | 10 s | ~3 min |
| 4 | **All poses**: the full sequence | all 9 below | 10 s | ~4 min |

- **Children** uses kid-friendly instructions ("Be a tree!", "Sit in an invisible chair!")
  and short 6-second holds. It leaves out Triangle (hard to align), Bridge (a backbend)
  and Goddess (a deep, wide squat). Children should practise with an adult nearby; the first
  instruction says so. The scoring rules are the same as for adults.
- **Men** and **Women** differ in focus, not in what anyone is able to do. None of these poses is
  only for men or only for women, and anyone can pick any program. Men's programs commonly
  target leg strength and tight hamstrings and hips (Chair, Warrior II, Triangle,
  Downward Dog, Bridge). Women's programs commonly target balance and hip mobility (Tree,
  Goddess, Warrior II, Triangle, Bridge). Both start with Mountain and end with Child's Pose to rest.
- This is general fitness guidance, not medical advice. Anyone who is pregnant, injured or
  has a health condition should check with a doctor first.

Programs are defined in `poses/programs.json` (see "Changing the programs" below).
`python main.py --program children` (or `men`, `women`, `all`) preselects one, so Space
starts it straight away.

### Poses

The full sequence ("All poses") runs standing poses first, then floor poses:

| # | Pose | Face the camera or side on | Camera |
|---|---|---|---|
| 1 | Mountain (Tadasana) | facing | table height |
| 2 | Chair (Utkatasana) | side on | table height |
| 3 | Tree (Vrikshasana), standing on the right leg | facing | table height |
| 4 | Goddess (Utkata Konasana) | facing | table height |
| 5 | Warrior II (Virabhadrasana II), right knee bent | facing | table height |
| 6 | Triangle (Trikonasana), over the left leg | facing | table height |
| 7 | Downward Dog (Adho Mukha Svanasana) | side on | low |
| 8 | Child's Pose (Balasana) | side on | low, near the floor |
| 9 | Bridge (Setu Bandha Sarvangasana) | side on | low, near the floor |

Chair, Goddess, Child's Pose and Bridge were added after the first five, as easy
beginner poses. Side-on poses work facing either direction.

---

## Setup

You need Python 3.10 or newer and a webcam. Internet is only needed for these setup steps.

**Windows, quickest:** double-click **`setup.cmd`**. It creates `.venv`, installs the
packages, downloads the model if it is missing, and runs the tests. Then use `run.cmd`.

**Any OS, by hand:**

```bash
cd yoga-pose-coach
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Download the MediaPipe pose model (about 9 MB) into `models/`:

```bash
python -c "import config, urllib.request; config.MODEL_PATH.parent.mkdir(exist_ok=True); urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)"
```

After that the app works fully offline.

Tested with Python 3.13, MediaPipe 1.1.0 (Tasks API), OpenCV 5.0 and NumPy 2.5 on Windows 11.
If a newer package version breaks something, `pip install -r requirements-lock.txt` installs
the exact versions that were tested.

## Running

On Windows, double-click **`run.cmd`**. Or, from the activated environment:

```bash
python main.py                    # webcam (config.CAMERA_INDEX)
python main.py --source 1         # another camera
python main.py --source clip.mp4  # a video file instead of the webcam (looped)
```

Stand back so your **whole body** is in the frame, with even light and a plain background
if you can. The video is mirrored like a mirror; the reference figures are drawn the same
way, so copy the picture as you would a mirror image. For the floor poses, move the laptop
to the floor (or a low stool) about 2-3 m away, so your whole body fits in the frame lying down.

### Using the app

| Stage | What happens |
|---|---|
| Choose a program | Press 1-4 or click one. Space starts the highlighted one |
| Get ready | Reference figure, pose name and instruction for 6 s (Space to skip) |
| Countdown | 3 s to get into position |
| Hold the pose | 10 s window. The hold bar fills while you are steady in the pose; the accuracy bar shows your live score |
| Result | Your score and up to 3 correction tips. Next pose after 5 s (or Space) |
| Session complete | Overall score, per-pose scores and hold times, longest hold, the 3 most common mistakes |

Keys: **1-4** choose a program (on the start and summary screens), **Space** start / skip the
current stage, **R** stop and go back to the program picker (Space then repeats the same program),
**Q** quit.

The left panel shows the live skeleton, each joint's current angle against its target
(coloured by joint score: green 80+, amber 60+, red below), and FPS / inference time in the corner.

Each completed session is saved to `logs/session_YYYYMMDD_HHMMSS.json`, with timestamps,
per-pose scores, hold times, per-joint average scores and deviations, the tips shown, and
which program was run (`"program"`).

## How scoring works

1. MediaPipe finds 33 body landmarks per frame. Landmarks with visibility below 0.5 are
   treated as missing.
2. Each pose has 6 joint-angle rules (for example, the left knee angle between hip, knee
   and ankle). A rule whose landmarks aren't all visible is skipped for that frame. It is
   left out of the average, not scored as 0.
3. Joint score = `100 - 2 x (degrees beyond tolerance)`, clamped to 0-100. The frame score
   is the weighted average over the joints that could be measured.
4. A frame only counts as **in pose** when the average score over the last 15 frames is at
   least 60 **and** every joint angle is steady (variance at most 16 deg², roughly 4°
   standard deviation). You must stay in pose for 1 s continuously before frames are
   scored. Leaving the pose for more than 1 s resets the hold.
5. Pose score = average of the scored frames. If you never reach a steady hold, you get
   a partial score from all measured frames, **capped at 50%**.
6. Tips come from the 3 lowest-scoring joints that are at least 10° off. A joint's
   `tip_low` is used when its angle is too small and `tip_high` when it is too large.

All thresholds live in `config.py`.

## Calibrating a pose to your body

The default target angles were measured off the stick-figure pictures
(`"calibrated": false` in `poses.json`). To replace them with your own:

```bash
python tools/record_pose.py --list     # pose ids
python tools/record_pose.py tree       # or: tree --source 1 for another camera
```

1. A window shows your skeleton and each joint's current angle against its target.
2. Get into the pose (same side as the instruction says) and press **Space**.
3. Hold still for 5 seconds (`RECORD_SECONDS` in `config.py`).
4. The script prints the old and new angle for each joint and writes the averages into
   `poses/poses.json` for that pose, with `"calibrated": true`.

Frames where a joint's landmarks aren't visible are left out of that joint's average. If a
joint is never visible during the recording, its old target is kept and the script says
so. Press **Q** or **Esc** to quit without saving. Tolerances, weights and tips are not
changed. Keep a copy of `poses.json` if you may want the defaults back.

Record in good light with your whole body in view, and do the best version of the pose
you want to be coached towards: the app will then score you against that.

## Adding a new pose

1. Add an entry to `poses/poses.json`:

   ```json
   {
     "id": "staff",
     "name": "Staff Pose (Dandasana)",
     "instruction": "Turn sideways to the camera. Sit tall with both legs straight out in front of you.",
     "image": "poses/images/staff.png",
     "hold_seconds": 10,
     "calibrated": false,
     "joints": [
       {"name": "left_knee", "landmarks": ["left_hip", "left_knee", "left_ankle"],
        "target_angle": 180, "tolerance": 10, "weight": 1.0,
        "tip_low": "Knees are bent. Straighten your legs along the floor.",
        "tip_high": "Knees are locked back. Soften them slightly."}
     ]
   }
   ```

   - `landmarks` are three [MediaPipe Pose landmark names](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model)
     (`left_shoulder`, `right_hip`, `left_ankle`, ...). The middle one is the vertex
     of the angle. The full list is `LANDMARK_NAMES` in `app/geometry.py`.
   - `target_angle` is 0-180°. A straight limb is 180°.
   - `tip_low` is shown when the user's angle is **smaller** than the target, `tip_high`
     when it is larger.
   - 4-6 rules per pose works well. Use `weight` (default 1.0) to make important joints
     count more.
   - Choose joints that make the pose **different from the other poses**, not just
     joints the pose has. Scoring only sees angles, so a pose that differs from another by
     one joint (for example Star pose versus Warrior II, which differ only in the front
     knee) will be accepted for each other.
   - For a side-on pose, add the same rule for both sides with the **same** tip text.
     Both sides overlap from the side, and identical tips are shown once.
2. Put a reference image at the `image` path (PNG or JPG; it is scaled to fit). You can
   also add keypoints for the pose to `FIGURES` in `tools/make_reference_images.py` and
   run `python tools/make_reference_images.py` to draw a matching stick figure. It prints
   the angle each rule makes in the drawing, so you can use those as targets.
3. Run `python tools/evaluate.py --no-speed`. The synthetic table shows whether any other
   pose scores high on the new one, or the new one on others. Look for `*` outside the
   diagonal.
4. Run `python tools/record_pose.py staff` to calibrate the targets from your own body.
5. Add the id to the expected list in `test_shipped_poses_are_valid`
   (`tests/test_session.py`), run `pytest` (the end-to-end tests pick up every pose that
   has a stick figure), and try it with `python main.py`.

Poses run in the order they appear in the file.

## Changing the programs

Edit `poses/programs.json`. Each program has an `id`, a `name`, a one-line `focus`, and a
list of `poses` from `poses.json`, in the order they run:

```json
{
  "id": "children",
  "name": "Children",
  "focus": "Playful poses, short holds",
  "hold_seconds": 6,
  "poses": [
    {"id": "tree", "instruction": "Be a tree! ..."},
    "chair"
  ]
}
```

- A pose is either an id (`"chair"`) or an object that overrides its `instruction`
  and/or `hold_seconds` for this program only.
- `hold_seconds` at program level applies to every pose in it.
- Optional `extra_tolerance` adds that many degrees to every joint's tolerance, to make a
  program more forgiving. The Children program doesn't use it: in testing, +5° accepted no
  more correct poses but let more wrong ones through (5 wrong acceptances on the photo clips
  instead of 3).
- Targets come from `poses.json`, so calibrating a pose updates it in every program.
- "All poses" is added automatically, with every pose in `poses.json`.
- Keep the focus line short: the picker shows two lines per program.

## Tests

```bash
pytest
```

83 tests cover angles, normalization and visibility handling; joint, frame and pose
scoring; hold detection (steady, moving, leaving the pose); tip selection; the session
state machine and logging; the calibration math; and full attempts on synthetic landmark
sequences through the same `Session` code the app uses; and program loading (overrides
don't change the base poses, unknown poses are rejected, children can hold every pose
within the 6 s window). None of them need a camera.

## Evaluation

```bash
python tools/evaluate.py                # speed test + synthetic sequences + clips in eval_clips/
python tools/evaluate.py --clips DIR    # clips from another folder
python tools/evaluate.py --no-speed
```

It reports:

1. **Speed.** Average FPS and per-frame inference latency of the worker pipeline (read
   frame, detect, compute angles, draw the view). It uses the webcam, or the first clip
   when no camera is available. It also prints whether the laptop is on battery,
   because that changes the numbers a lot.
2. **Synthetic sequences.** Each stick-figure pose held steady with random landmark
   jitter, scored against all five poses, plus a user swinging back and forth through
   each pose every 3 s.
3. **Test clips.** Every video in `eval_clips/` scored against all five poses. Name each
   clip after the pose it shows, e.g. `tree_front.mp4` or `downward_dog_me.mp4`. A
   name that starts with no pose id (e.g. `wrong_slouch.mp4`) means no pose should
   accept it. `eval_clips/` is gitignored, and the script only reads the clips.

A pose counts as **accepted** when a steady hold was established. A **false positive**
is a wrong pose accepted.

### Results on the development laptop (14 Oct 2026)

Intel CPU (Family 6 Model 154), **on battery**, Python 3.13.5, MediaPipe 1.1.0, full model,
all 9 poses with default targets. No webcam was available, so the speed test and clips used
2-second looping videos made from seven openly licensed photos on Wikimedia Commons (one
per pose for Tree, Triangle, Warrior II, Chair, Goddess, Child's Pose and Bridge).

**Speed** (150 frames from a clip, single thread, on battery):

| | |
|---|---|
| Average FPS | 36.1 |
| Inference latency | mean 25.1 ms, median 25.0 ms, p95 27.4 ms |
| Whole frame (read + detect + angles + draw) | mean 27.7 ms |

Speed depends heavily on the power state. Earlier runs on battery measured 44 ms
(21 FPS) and, with the battery-saving CPU cap active, about 70 ms. In the app the video
runs in a worker thread, so a slow frame lowers the video FPS but never freezes the window.

**Synthetic sequences** (score per pose, `*` = hold accepted, bold = the pose shown):

| shown \ scored as | Mountain | Chair | Tree | Goddess | Warrior II | Triangle | Down Dog | Child's | Bridge |
|---|---|---|---|---|---|---|---|---|---|
| Mountain | **100\*** | 21 | 43 | 0 | 27 | 45 | 29 | 0 | 50 |
| Chair | 0 | **100\*** | 44 | 43 | 40 | 17 | 50 | 33 | 42 |
| Tree | 39 | 39 | **100\*** | 20 | 13 | 26 | 43 | 44 | 33 |
| Goddess | 0 | 50 | 21 | **100\*** | 74\* | 39 | 14 | 0 | 37 |
| Warrior II | 29 | 50 | 20 | 50 | **100\*** | 50 | 20 | 0 | 34 |
| Triangle | 44 | 32 | 29 | 29 | 50 | **100\*** | 50 | 15 | 14 |
| Downward Dog | 37 | 50 | 47 | 0 | 23 | 60\* | **100\*** | 50 | 0 |
| Child's Pose | 0 | 50 | 39 | 0 | 0 | 19 | 50 | **100\*** | 23 |
| Bridge | 50 | 50 | 33 | 36 | 32 | 11 | 0 | 20 | **100\*** |

- Correct pose accepted 9/9. Wrong pose accepted **2/72**: Goddess as Warrior II (74)
  and Downward Dog as Triangle (exactly 60). See known limitations.
- Moving through the pose: **0/9** given a hold, all at or below the 50% cap.
- Wrong-pose attempts with 3 tips: 59/72. The other 13 got 2; 12 of them were scored as a
  side-on pose (Downward Dog, Chair, Child's Pose, Bridge), which can show at most 3
  different tips (see known limitations).

The synthetic figures are the same drawings the default targets came from, so the 100s
on the diagonal are a sanity check of the pipeline, not a measure of accuracy. The
useful numbers are the wrong-pose and moving rows.

**Test clips** (photos of real people):

| clip | Mountain | Chair | Tree | Goddess | Warrior II | Triangle | Down Dog | Child's | Bridge |
|---|---|---|---|---|---|---|---|---|---|
| bridge | 23 | 60\* | 40 | 42 | 50 | 28 | 0 | 25 | **85\*** |
| chair | 0 | **100\*** | 24 | 28 | 50 | 10 | 50 | 50 | 43 |
| child's pose | 0 | 64\* | 39 | 0 | 0 | 3 | 64\* | **100\*** | 16 |
| goddess | 3 | 50 | 39 | **84\*** | 50 | 10 | 36 | 35 | 50 |
| tree | 48 | 49 | **100\*** | 13 | 16 | 31 | 50 | 31 | 28 |
| triangle | 48 | 29 | 34 | 28 | 50 | **100\*** | 50 | 18 | 18 |
| warrior2 | 27 | 50 | 23 | 50 | **99\*** | 50 | 14 | 0 | 31 |

Correct pose accepted **7/7**, wrong pose accepted **3/56**: Child's Pose as Chair (64)
and as Downward Dog (64), and Bridge as Chair (60, right at the threshold). 49/56 wrong
attempts got 3 tips. The Goddess photo is a variation with one arm raised overhead
instead of both elbows bent, so it scores 84 rather than 100.

Some photo variations score low on purpose, with the right correction. A chair photo with
the arms straight forward instead of up scored 57 ("Arms are too low"), and a child's
pose with the arms back by the feet scored 60 ("Arms are not reaching forward").

**Mirrored clips** (the same photos flipped, i.e. the pose done on the wrong side):

| clip | Mountain | Chair | Tree | Goddess | Warrior II | Triangle | Down Dog | Child's | Bridge |
|---|---|---|---|---|---|---|---|---|---|
| flipped tree | 49 | 49 | 42 | 12 | 50 | 41 | 50 | 31 | 29 |
| flipped triangle | 47 | 29 | 40 | 30 | 64\* | 60\* | 49 | 20 | 20 |
| flipped warrior2 | 28 | 49 | 50 | 50 | 50 | 50 | 15 | 0 | 28 |

The mirrored Tree and Warrior II are rejected everywhere. A Triangle done over the wrong
leg is still accepted as Triangle (60), and also as **Warrior II (64)**, which is a
false positive (see known limitations).

These clips are still photos, so they have no natural sway. Run `evaluate.py` on your own
recordings in `eval_clips/` for a fairer test.

## Design decisions

The spec left these open; this is what was chosen and why.

- **2D angles.** Angles use the image x/y in pixels. MediaPipe's depth estimate is
  noisy from a single webcam.
- **Fixed sides.** Each pose is done on one side (e.g. Tree stands on the right leg,
  Triangle bends over the left leg), as the instruction says. Mirrored attempts mostly
  score low, with one exception (see the mirrored clips above).
- **Mirrored display only.** The video is flipped for display. Landmarks and scoring use
  the unflipped frame, so "left" always means your anatomical left.
- **Reference images are stick figures** drawn by `tools/make_reference_images.py`, and
  the default targets are measured from the same drawings, so picture and rules agree.
  Real photos could not be bundled without licensing them.
- **A hold starts only after 1 s continuously in pose.** The spec's grace period alone let
  a slowly moving user build up a hold by touching the pose briefly and repeatedly.
- **No hold means a score of at most 50%** (`NO_HOLD_SCORE_CAP`). Without a cap, someone
  swinging one limb while the rest was perfect scored 84% with no hold.
- **Duplicate tips are shown once.** Left and right rules can share the same wording
  (e.g. Downward Dog), and the same sentence twice reads like a bug.
- **Warrior II checks the hips instead of the elbows.** With elbow rules, a Triangle pose
  scored 72% on Warrior II and registered a hold. With hip rules it scores 50.
- **Pillow** is used to hand frames to Tk. Tk's built-in image path took about 15 ms per
  frame. MediaPipe already installs Pillow (through matplotlib).
- **The worker thread does capture, detection and drawing.** The GUI only shows the
  newest frame from a one-slot queue, so it stays responsive at about 30 refreshes per second.
- **The window is DPI-aware on Windows**, so text is sharp on scaled displays.
- **Sessions are logged when they reach the summary.** Quitting halfway saves nothing.
- **`--source`** lets the app, the calibration tool and the evaluation take a video file
  instead of the webcam. Without it, nothing could be tested on a machine with no camera.
- **The four added poses were chosen to be different in joint angles.** Upward Salute
  was tried and removed: in angles it is Mountain with arms raised, Tree with the foot
  down, and Downward Dog unfolded at the hips (Dog scored 100% on it). Star pose was
  skipped for the same reason (it is Warrior II with a straight front knee).
- **Chair has 8 rules** (knees, hips, shoulders, elbows) instead of 4-6. The knee and hip
  angles match Goddess, and the arm angles match Tree, Child's Pose and Downward Dog;
  straight elbows plus raised arms is what sets it apart. The Chair figure was drawn
  deeper after real chair photos measured 70-77° at the knee (the target is 92° with a
  20° tolerance, so beginners and deep squats both count).
- **Programs reuse the 9 tested poses** rather than adding group-specific new ones. Each
  new pose needs its own confusion testing (see Upward Salute above). A program only
  selects, orders and re-words poses, so all the evaluation results still apply.
- **R goes back to the program picker** instead of restarting immediately; Space then
  repeats the same program. With several programs, one extra key press seemed better than
  a restart key that can't change the program.
- **The panel shows the English pose name** ("Bridge Pose"); the Sanskrit name is in
  poses.json and the logs. Two-line titles pushed the instruction text off the bottom
  of the panel at 150% display scaling.
- **The summary lists each common mistake once, by its first sentence**
  ("Left knee is bent."), merging the same mistake from different poses. The session log
  keeps the full tips (`common_tips`) as well as the merged list (`common_mistakes`).

## Known limitations

- **Angles can't see which way the body faces.** Scoring is angle-based, as specified, so
  poses with similar joint angles can look alike. The one false positive above is
  Downward Dog scored as Triangle: straight legs and a sharply folded hip satisfy half of
  Triangle's rules, which lands exactly on the 60% hold threshold. A Triangle done over
  the right leg instead of the left scored 63.8% on Warrior II and was accepted:
  both poses have wide straight-ish legs and one arm up and one down, and the angles
  can't tell a tipped torso from an upright one. Of the added poses, Goddess is accepted
  as Warrior II (74%, they differ only in the back knee and the elbows, which Warrior II
  doesn't check), and on real photos Child's Pose passed as Chair and Downward Dog (64%)
  and Bridge as Chair (60%). In side views the far leg is often hidden, which leaves the
  arms (which match) to carry the score.
- **A torso-tilt rule would fix most of these.** Chair leans the torso about 35°, Goddess
  and Warrior II keep it upright, Child's Pose is horizontal and Downward Dog points
  down. That is the angle between the torso and vertical, not a joint angle, so it is
  not in the scoring yet.
- **Side-on poses can show at most 3 different tips, often 2** (Downward Dog, Chair,
  Child's Pose, Bridge). Their left and right rules share wording, and duplicates are
  removed. Attempts that are wrong in only one or two ways (e.g. standing with arms
  overhead, which matches Dog's arm angle) get fewer than 3.
- **Tips can contradict each other on a completely different pose**, because each joint
  is judged on its own (e.g. "too upright" from one hip and "folding too far" from the other).
- **The steadiness window is counted in frames** (15), so at low FPS it covers more time
  and a hold takes slightly longer to register.
- **Live webcam use has been tried once, not measured.** The first development laptop had no
  camera, so the results above come from video files of still photos. A first live run on a
  second laptop (two full sessions) ran at about 26-27 FPS and scored correct poses 81-100%.
  It also showed that when the feet are out of frame, the knee rules can't be measured, and a
  pose can then be accepted on very few joints (Bridge scored 100% from the two shoulder rules
  alone). Floor poses depend on camera placement and lighting, which photos can't test. The
  calibration tool has not been used with a webcam yet.
- **The Children program has not been tried with children.** MediaPipe works on children's
  bodies, but kids move more, so 6-second holds may be hard to reach; the 1 s needed to start
  a hold still applies.
- **Programs run their poses in a fixed order;** there is no way to pick single poses in the app. The
  no-camera path shows "Camera not available" instead of crashing.
- **Calibration has no undo.** Back up `poses.json` before recording.

## Project layout

```
yoga-pose-coach/
├── main.py                 # entry point (GUI)
├── config.py               # all tunable constants
├── run.cmd                 # Windows launcher
├── setup.cmd               # Windows one-time setup
├── requirements-lock.txt   # exact package versions that were tested
├── SPEC.md                 # the original project spec
├── CLAUDE.md               # status and context for resuming work with Claude Code
├── app/
│   ├── camera.py           # webcam / video capture
│   ├── pose_detector.py    # MediaPipe wrapper and skeleton drawing
│   ├── geometry.py         # angles, normalization, visibility
│   ├── scorer.py           # joint, frame and pose scores
│   ├── stability.py        # hold detection
│   ├── feedback.py         # correction tips
│   ├── session.py          # state machine, pose loading, session logs
│   └── gui.py              # Tkinter window and capture worker thread
├── poses/
│   ├── poses.json          # pose definitions
│   ├── programs.json       # Children / Men / Women programs
│   └── images/             # reference figures
├── tools/
│   ├── record_pose.py      # calibrate a pose from the webcam
│   ├── evaluate.py         # speed and scoring evaluation
│   └── make_reference_images.py
├── models/                 # pose_landmarker_full.task (downloaded, gitignored)
├── eval_clips/             # test clips for tools/evaluate.py (gitignored), CREDITS.txt
├── tests/                  # pytest suite
└── logs/                   # session logs (gitignored)
```
