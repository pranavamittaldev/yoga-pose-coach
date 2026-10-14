"""Pose sequence state machine, pose loading and session logging.

The Session is pure: every method takes the current time (seconds, any
monotonic clock) as an argument, so it can be tested without a camera or GUI.

States: IDLE -> SHOW_REFERENCE -> COUNTDOWN -> PERFORMING -> SCORING ->
FEEDBACK -> (SHOW_REFERENCE for the next pose, or SUMMARY).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import config
from app.feedback import select_tips
from app.scorer import FrameScore, JointRule, pose_score, score_frame, summarize_joints
from app.stability import HoldStatus, HoldTracker

NO_BODY_TIP = "We could not see your body. Step back so your whole body is in the frame."


class State(Enum):
    IDLE = "idle"
    SHOW_REFERENCE = "show_reference"
    COUNTDOWN = "countdown"
    PERFORMING = "performing"
    SCORING = "scoring"
    FEEDBACK = "feedback"
    SUMMARY = "summary"


@dataclass
class Pose:
    """A reference pose loaded from poses.json."""

    id: str
    name: str
    instruction: str
    image: str
    hold_seconds: float
    joints: list[JointRule]
    calibrated: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        """Build a pose from its JSON dictionary."""
        return cls(
            id=d["id"],
            name=d["name"],
            instruction=d["instruction"],
            image=d["image"],
            hold_seconds=float(d.get("hold_seconds", config.DEFAULT_HOLD_SECONDS)),
            joints=[JointRule.from_dict(j) for j in d["joints"]],
            calibrated=bool(d.get("calibrated", False)),
        )


def load_poses(path: Path = config.POSES_FILE) -> list[Pose]:
    """Load all pose definitions from a poses.json file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Pose.from_dict(d) for d in data["poses"]]


@dataclass
class Program:
    """A named sequence of poses, e.g. the Children program."""

    id: str
    name: str
    focus: str
    poses: list[Pose]

    def duration_seconds(self) -> float:
        """Approximate length of a full run: reference, countdown, hold and result for every pose."""
        per_stage = config.REFERENCE_DISPLAY_SECONDS + config.COUNTDOWN_SECONDS + config.FEEDBACK_DISPLAY_SECONDS
        return sum(per_stage + p.hold_seconds for p in self.poses)


ALL_POSES_ID = "all"


def all_poses_program(poses: list[Pose]) -> Program:
    """The program that runs every pose in poses.json, in file order."""
    return Program(ALL_POSES_ID, "All poses", "The full sequence", list(poses))


def load_programs(poses: list[Pose], path: Path = config.PROGRAMS_FILE) -> list[Program]:
    """Load programs.json, resolving pose ids against ``poses``.

    Each program's poses are copies of the base poses with the program's
    overrides applied (instruction, hold_seconds, extra_tolerance), so the base
    poses are never changed. Calibrated targets carry over automatically.
    An "All poses" program is appended at the end. Raises ValueError for an
    unknown pose id or an empty program.
    """
    by_id = {p.id: p for p in poses}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    programs = []
    for d in data["programs"]:
        extra = float(d.get("extra_tolerance", 0))
        resolved = []
        for item in d["poses"]:
            item = {"id": item} if isinstance(item, str) else item
            base = by_id.get(item["id"])
            if base is None:
                raise ValueError(f"Program {d['id']!r} uses unknown pose {item['id']!r}")
            joints = [replace(j, tolerance=j.tolerance + extra) for j in base.joints] if extra else base.joints
            hold = item.get("hold_seconds", d.get("hold_seconds", base.hold_seconds))
            resolved.append(replace(base, instruction=item.get("instruction", base.instruction),
                                    hold_seconds=float(hold), joints=joints))
        if not resolved:
            raise ValueError(f"Program {d['id']!r} has no poses")
        programs.append(Program(d["id"], d["name"], d.get("focus", ""), resolved))
    programs.append(all_poses_program(poses))
    return programs


def write_poses_json(data: dict, path: Path = config.POSES_FILE) -> None:
    """Write raw poses.json data, keeping one joint rule per line so the file stays readable."""
    lines = ["{"]
    for key, value in data.items():
        if key != "poses":
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},")
    lines.append('  "poses": [')
    for i, pose in enumerate(data["poses"]):
        lines.append("    {")
        fields = [(k, v) for k, v in pose.items() if k != "joints"]
        for k, v in fields:
            lines.append(f"      {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
        lines.append('      "joints": [')
        joints = pose["joints"]
        for j, joint in enumerate(joints):
            comma = "," if j < len(joints) - 1 else ""
            lines.append(f"        {json.dumps(joint, ensure_ascii=False)}{comma}")
        lines.append("      ]")
        lines.append("    }" + ("," if i < len(data["poses"]) - 1 else ""))
    lines.append("  ]")
    lines.append("}")
    text = "\n".join(lines) + "\n"
    json.loads(text)  # sanity check before overwriting
    path.write_text(text, encoding="utf-8", newline="\n")  # LF on every OS


@dataclass
class PoseResult:
    """Outcome of one pose attempt."""

    pose_id: str
    name: str
    score: float  # 0-100
    held: bool  # True if a stable hold was achieved (score uses held frames only)
    best_hold_seconds: float
    target_hold_seconds: float
    frames_measured: int
    frames_scored: int
    joint_averages: dict[str, dict]  # joint -> {avg_score, avg_signed_deviation, frames}
    tips: list[str]
    started_at: str
    ended_at: str


def mistake_of(tip: str) -> str:
    """The first sentence of a tip, which names the mistake.

    "Left knee is bent. Straighten it." -> "Left knee is bent."
    """
    head, sep, _ = tip.partition(". ")
    return head + "." if sep else tip


@dataclass
class Summary:
    """End-of-session summary."""

    total_score: float
    pose_scores: list[tuple[str, float, float]]  # (name, score, best hold seconds)
    longest_hold: tuple[str, float]  # (pose name, seconds)
    common_tips: list[tuple[str, int]]  # (tip, times shown)
    common_mistakes: list[tuple[str, int]] = field(default_factory=list)  # (first sentence of tips, count)


@dataclass
class _Attempt:
    """Frames collected during PERFORMING for the current pose."""

    tracker: HoldTracker
    started_at: str
    all_frames: list[FrameScore] = field(default_factory=list)
    scored_frames: list[FrameScore] = field(default_factory=list)


def evaluate_attempt(pose: Pose, all_frames: list[FrameScore], scored_frames: list[FrameScore],
                     best_hold: float, started_at: str, ended_at: str) -> PoseResult:
    """Turn the collected frames of one attempt into a PoseResult.

    With a stable hold, the score averages the held (scored) frames only.
    Without one, the partial score is the average over every measured frame
    in the window, capped at ``config.NO_HOLD_SCORE_CAP`` so a user who moves
    through the pose never gets a high score. With no measurable frames at
    all, the score is 0.
    """
    held = bool(scored_frames)
    frames = scored_frames if held else all_frames
    score = pose_score(frames)
    summaries = summarize_joints(frames)
    measured = sum(1 for f in all_frames if f.score is not None)
    if score is None:
        tips = [NO_BODY_TIP]
        score = 0.0
    else:
        tips = [t.text for t in select_tips(summaries, pose.joints)]
        if not held:
            score = min(score, config.NO_HOLD_SCORE_CAP)
    return PoseResult(
        pose_id=pose.id,
        name=pose.name,
        score=round(score, 1),
        held=held,
        best_hold_seconds=round(best_hold, 2),
        target_hold_seconds=pose.hold_seconds,
        frames_measured=measured,
        frames_scored=len(scored_frames),
        joint_averages={
            name: {"avg_score": round(s.avg_score, 1),
                   "avg_signed_deviation": round(s.avg_signed_deviation, 1),
                   "frames": s.frames}
            for name, s in summaries.items()
        },
        tips=tips,
        started_at=started_at,
        ended_at=ended_at,
    )


class Session:
    """Runs the pose sequence. Drive it with ``tick(now)``, ``skip(now)`` and ``process_frame``."""

    def __init__(self, poses: list[Pose], wall_clock: Callable[[], datetime] = datetime.now,
                 program_id: Optional[str] = None):
        if not poses:
            raise ValueError("A session needs at least one pose")
        self.poses = poses
        self.program_id = program_id
        self._wall_clock = wall_clock
        self.restart()

    # --- state ---------------------------------------------------------------

    def restart(self) -> None:
        """Return to IDLE and forget all results."""
        self.state = State.IDLE
        self.pose_index = 0
        self.state_started = 0.0
        self.results: list[PoseResult] = []
        self.session_started_at: Optional[str] = None
        self.session_ended_at: Optional[str] = None
        self._attempt: Optional[_Attempt] = None
        self.last_status: Optional[HoldStatus] = None
        self.last_frame: Optional[FrameScore] = None

    @property
    def current_pose(self) -> Pose:
        """The pose being shown or attempted (the last pose once in SUMMARY)."""
        return self.poses[min(self.pose_index, len(self.poses) - 1)]

    @property
    def last_result(self) -> Optional[PoseResult]:
        """Result of the most recently finished pose."""
        return self.results[-1] if self.results else None

    def _now_iso(self) -> str:
        return self._wall_clock().isoformat(timespec="seconds")

    def _enter(self, state: State, now: float) -> None:
        self.state = state
        self.state_started = now
        if state == State.PERFORMING:
            self._attempt = _Attempt(tracker=HoldTracker(), started_at=self._now_iso())
            self.last_status = None
            self.last_frame = None

    def state_duration(self) -> Optional[float]:
        """How long the current state lasts before auto-advancing (None = until a key press)."""
        return {
            State.SHOW_REFERENCE: config.REFERENCE_DISPLAY_SECONDS,
            State.COUNTDOWN: config.COUNTDOWN_SECONDS,
            State.PERFORMING: self.current_pose.hold_seconds,
            State.FEEDBACK: config.FEEDBACK_DISPLAY_SECONDS,
        }.get(self.state)

    def time_left(self, now: float) -> Optional[float]:
        """Seconds left in the current timed state, or None for untimed states."""
        duration = self.state_duration()
        if duration is None:
            return None
        return max(0.0, duration - (now - self.state_started))

    # --- transitions ---------------------------------------------------------

    def start(self, now: float) -> None:
        """Begin the sequence from IDLE."""
        if self.state == State.IDLE:
            self.session_started_at = self._now_iso()
            self.pose_index = 0
            self._enter(State.SHOW_REFERENCE, now)

    def skip(self, now: float) -> None:
        """Space key: start from IDLE, or end the current stage early."""
        if self.state == State.IDLE:
            self.start(now)
        elif self.state in (State.SHOW_REFERENCE, State.COUNTDOWN, State.PERFORMING, State.FEEDBACK):
            self._advance(now)

    def tick(self, now: float) -> None:
        """Advance the state when its time is up. Call this regularly."""
        left = self.time_left(now)
        if left is not None and left <= 0:
            self._advance(now)

    def _advance(self, now: float) -> None:
        if self.state == State.SHOW_REFERENCE:
            self._enter(State.COUNTDOWN, now)
        elif self.state == State.COUNTDOWN:
            self._enter(State.PERFORMING, now)
        elif self.state == State.PERFORMING:
            self._enter(State.SCORING, now)
            self._finish_attempt()
            self._enter(State.FEEDBACK, now)
        elif self.state == State.FEEDBACK:
            if self.pose_index + 1 < len(self.poses):
                self.pose_index += 1
                self._enter(State.SHOW_REFERENCE, now)
            else:
                self.session_ended_at = self._now_iso()
                self._enter(State.SUMMARY, now)

    def _finish_attempt(self) -> None:
        a = self._attempt
        self.results.append(evaluate_attempt(
            self.current_pose, a.all_frames, a.scored_frames,
            a.tracker.best_hold_seconds, a.started_at, self._now_iso(),
        ))
        self._attempt = None

    # --- frames --------------------------------------------------------------

    def process_frame(self, now: float, angles: Optional[dict[str, Optional[float]]]) -> Optional[FrameScore]:
        """Score a frame's joint angles while PERFORMING.

        ``angles`` is None when no person was detected. Returns the frame
        score, or None outside PERFORMING. Live values are kept in
        ``last_frame`` and ``last_status`` for the GUI.
        """
        if self.state != State.PERFORMING or self._attempt is None:
            return None
        if angles is None:
            angles = {j.name: None for j in self.current_pose.joints}
        frame = score_frame(angles, self.current_pose.joints)
        status = self._attempt.tracker.update(now, frame)
        self._attempt.all_frames.append(frame)
        if status.should_score:
            self._attempt.scored_frames.append(frame)
        self.last_frame = frame
        self.last_status = status
        return frame

    # --- summary and logging ------------------------------------------------

    def summary(self) -> Summary:
        """Totals over the poses completed so far."""
        scores = [r.score for r in self.results]
        total = sum(scores) / len(scores) if scores else 0.0
        longest = max(self.results, key=lambda r: r.best_hold_seconds, default=None)
        tips = Counter(t for r in self.results for t in r.tips if t != NO_BODY_TIP)
        mistakes = Counter()
        for tip, count in tips.items():  # e.g. "Left knee is bent." from Mountain and Warrior II count together
            mistakes[mistake_of(tip)] += count
        return Summary(
            total_score=round(total, 1),
            pose_scores=[(r.name, r.score, r.best_hold_seconds) for r in self.results],
            longest_hold=(longest.name, longest.best_hold_seconds) if longest else ("-", 0.0),
            common_tips=tips.most_common(3),
            common_mistakes=mistakes.most_common(3),
        )

    def to_log(self) -> dict:
        """JSON-serializable record of the session."""
        s = self.summary()
        return {
            "program": self.program_id,
            "started_at": self.session_started_at,
            "ended_at": self.session_ended_at,
            "total_score": s.total_score,
            "longest_hold": {"pose": s.longest_hold[0], "seconds": s.longest_hold[1]},
            "common_tips": [{"tip": t, "count": c} for t, c in s.common_tips],
            "common_mistakes": [{"mistake": m, "count": c} for m, c in s.common_mistakes],
            "poses": [asdict(r) for r in self.results],
        }


def save_log(session: Session, directory: Path = config.LOGS_DIR) -> Path:
    """Write the session to ``directory/session_YYYYMMDD_HHMMSS.json`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"session_{stamp}.json"
    n = 1
    while path.exists():  # two sessions in the same second
        path = directory / f"session_{stamp}_{n}.json"
        n += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session.to_log(), f, indent=2)
    return path
