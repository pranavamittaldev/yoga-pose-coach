# Yoga Pose Coach: context for Claude

Read this first when resuming work, on any device. Last updated 2026-10-14 (end of session 4).

## What this is
A local Python desktop app (Tkinter + OpenCV + MediaPipe Tasks API) that coaches a yoga
sequence from the webcam: reference figure, countdown, 10 s timed hold, joint-angle score
0-100, up to 3 correction tips, and a session summary with a JSON log. The original spec is in
`SPEC.md`; everything that was decided since is in `README.md` ("Design decisions" and
"Known limitations"). Nothing leaves the machine, and no video is ever written to disk.

## Working rules from the user and the spec
- Don't add features beyond the spec without asking first. Offer them instead.
- Where the spec is ambiguous, choose the simplest option and note it in README.md.
- Don't claim something works unless it was run or tested. Report failures and false
  positives honestly; the README results tables come from real `tools/evaluate.py` runs.
- The user works in batches and likes a short summary at the end of each one.
- During automated GUI tests, keep windows off-screen (`root.geometry("1100x700+-4000+0")`);
  the user disliked test windows popping up. **The live demo is the exception**: the user
  wants to see the app on screen with their camera.

## Repository
Private GitHub repo: https://github.com/pranavamittaldev/yoga-pose-coach (branch `main`,
created 2026-10-14). Commits use the user's GitHub noreply address (repo-local git config).
Not in git, by `.gitignore`: `.venv/`, `models/` (`setup.cmd` downloads the model),
`eval_clips/` (copy the clips over separately, or `evaluate.py` reports no clips), and
`logs/` (personal session logs). Commit and push only when the user asks.

## Status
- All 10 spec milestones are done (3 batches), plus 4 extra poses the user asked for.
- 9 poses, in this order: mountain, chair, tree, goddess, warrior2, triangle, downward_dog,
  childs_pose, bridge. All have `"calibrated": false` (default targets measured off the
  stick figures in `tools/make_reference_images.py`).
- 83 tests pass (`pytest`). `python -m pyflakes app tools tests` is clean.
- **Programs (session 5, below):** the app opens on a picker: 1 Children, 2 Men, 3 Women,
  4 All poses. Defined in `poses/programs.json`.

## Session 5 (14 Oct 2026, back on the first laptop, no camera)
- The user copied the session-4 handoff back; it replaced `~/yoga-pose-coach` (code was identical).
- The user asked for "3 sections children men and women with exercises suitable for them".
  Built as **programs** that reuse the 9 tested poses (no new poses: each one needs confusion
  testing). `load_programs()` in `app/session.py` resolves them; `Session(program_id=...)`
  logs `"program"`. The GUI picker replaces the middle/instruction/detail rows while IDLE
  (`_set_chooser`). Keys: 1-4 pick from the picker or the summary, R goes back to the picker,
  Space starts or repeats the highlighted program. `main.py --program children|men|women|all`.
  `App(poses, ...)` without `programs` still works (demo-kit's `record_demo.py` relies on it).
  - Children: 6 poses, 6 s holds, kid-worded instructions (the first says to have an adult nearby).
    No extra tolerance: tested +3/+5°, which accepted no more correct poses but more wrong ones
    (photo clips: 3 -> 5 false accepts). The `extra_tolerance` field is still supported.
  - Men (leg strength and flexibility) and Women (balance and hip mobility) differ by focus only.
    The README says plainly that no pose is gender-specific and anyone can pick any program.
  - Verified: the picker and all 64 program screens fit at 150% DPI (off-screen geometry
    check); the full Children run, then 2 from the summary, then Men ran off-screen on
    `eval_clips/chair.mp4` with correct logs.
  - Not tested: real children, and the picker with a live camera.
- Full verification after the programs work: 83 tests, pyflakes, evaluate.py, record_pose.py on a clip
  (then poses.json restored), all 4 programs end to end off-screen with correct logs, keys, and no-camera.
  Fixed: `App.close()` now cancels the pending refresh timer (`_after_id`); before, a stale
  `after` could fire into a destroyed window ("invalid command name ..._update").

## Session 4 (14 Oct 2026, second Windows laptop with a webcam)
- Setup worked: `.venv` with Python 3.13.5, `pip install -r requirements.txt` (MediaPipe 1.1.0,
  OpenCV 5.0), 72 tests pass. Camera index 0 works (DirectShow); the frame looked dark.
- **First live webcam run.** The user did the whole sequence twice (logs in
  `logs/session_20261014_180048.json` and `logs/session_20261014_180429.json`):
  - Speed: ~26-27 FPS live (about 260-275 frames per 10 s hold). `tools/evaluate.py` speed
    test with the camera has NOT been run yet, so the README "Results" are still the old laptop's.
  - Good: Mountain 100/99, Goddess 84/93, Warrior II 81, Triangle 92, all with real holds.
    Clearly wrong attempts scored low with correct tips (Chair with arms down: 18).
  - **Knee/ankle rules were almost never measurable** (Mountain knees: 0 and 3 frames). The
    user's feet were probably out of frame: camera framing is the main real-world issue.
  - **False positive: Bridge 100%** in run 2 with only the 2 shoulder rules visible (15 frames),
    because arms-by-your-sides matches Bridge's shoulder angle. Down Dog in run 1 "held" on
    just 2 scored frames with 3 joints visible. Same root cause: scoring allows very few
    visible joints. The floor poses seem to have been done standing or skipped.
  - The user hasn't said yet how it felt (smoothness, tips, side-on poses, framing).
- **Demo video for friends** (the user asked for male models). Made from Wikimedia Commons photos
  fed to the real app; nothing from the user's webcam. Chair 98, Tree 100, Warrior II 96,
  Triangle 98, Bridge 86. Goddess was left out (no licensed male photo found).
  Published page: https://claude.ai/artifact/YNuwsCrVLBH4dWYMnuK8i1 (private until the user shares it).
  Everything to re-record it is in `../demo-kit/` next to this folder in the handoff zip (see
  its README). Not part of the app.
  - Off-screen Tk windows don't repaint on Windows, so `PrintWindow` captures are stale. The
    recorder keeps the window on screen but lowered behind other windows.
  - The detector runs in VIDEO (tracking) mode: use a fresh `PoseDetector` per clip, or
    tracking from the previous clip ruins the next one (Bridge dropped from 86 to 50).
  - The Warrior II result showed contradictory tips ("leaning toward your front leg" and
    "...back leg"), a known limitation the torso-tilt rule would fix.

## Next steps
1. Ask the user how the live run felt, if they haven't said.
2. **Offered, awaiting the user's answer (don't build unasked):** a minimum-visibility rule,
   so a frame only counts when enough of the pose's joints are visible (e.g. at least half the
   rule weight). Otherwise the app says "Can't see enough of your body, step back". It would
   have stopped the Bridge 100% and the 2-frame Dog hold. (Session 5 found that the live logs also
   show Goddess scored 84/93 with no knee ever visible, judged on the arms alone.) Add tests and
   re-run `evaluate.py`.
3. Also still awaiting an answer:
   - A **torso-tilt rule** (the angle of the shoulder-hip line against vertical) to fix most
     angle-only confusions: Goddess accepted as Warrior II (74%), Child's Pose as Chair or
     Downward Dog (64%), Bridge as Chair (60%), Downward Dog as Triangle (60%), a mirrored
     Triangle as Warrior II (64%), and the contradictory Warrior II tips.
   - A `--poses chair,goddess` option to run any subset. The programs now cover most of this
     need; only offer it again if the user wants single poses.
4. With the camera: try the program picker and the Children program live (are 6 s holds
   reachable?). Ask for feedback on the program contents.
5. Measure speed with the camera: `python tools/evaluate.py`, then update the README "Results"
   and the "Live webcam use is untested" limitation with real numbers.
6. Calibration, if the user wants it: `python tools/record_pose.py <pose_id>`, Space,
   hold 5 s. It writes `poses.json`, so back the file up first.
7. Coach the user on framing for the next live run: whole body including feet in view; for
   floor poses, put the laptop on the floor 2-3 m away.

## Things worth knowing
- MediaPipe 1.1.0 has no `mp.solutions`; this uses the Tasks API with
  `models/pose_landmarker_full.task` (included; `setup.cmd` downloads it if missing).
  Exact tested versions are in `requirements-lock.txt`.
- On the old laptop, inference took ~25 ms per frame plugged in and 44-70 ms on battery
  (CPU throttling). FPS depends heavily on the power state.
- The window is DPI-aware. At 150% scaling the right panel is tight. After any text change,
  check that nothing is clipped: compare each label's `winfo_reqheight()` against the space
  above the button bar; `winfo_height()` hides clipping.
- Before adding a pose, check confusion with `python tools/evaluate.py --no-speed`. Upward
  Salute was removed because Downward Dog scored 100% on it. See the README "Adding a new
  pose" section.
- `poses.json` is written by `app.session.write_poses_json` (one rule per line, LF endings).
- `logs/` and `eval_clips/` are gitignored; `eval_clips/` holds test clips only.
- To set up on a new machine, don't copy `.venv`: run `setup.cmd` again (it is machine-specific).
