"""Tkinter interface: live video with skeleton on the left, reference pose and status on the right.

Camera capture and pose inference run in a worker thread. The newest result
is handed to the GUI through a one-slot queue, and the GUI refreshes about
30 times per second, so the window never blocks on the camera or the model.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from app.camera import Camera, CameraError
from app.geometry import joint_angles
from app.pose_detector import PoseDetector, draw_skeleton, mirror_landmarks
from app.scorer import joint_score
from app.session import Pose, Program, Session, State, all_poses_program, save_log

VIDEO_W, VIDEO_H = 640, 480
REF_W, REF_H = 160, 200
UI_INTERVAL_MS = 1000 // 30

# Colors
BG = "#f4f1ec"
PANEL = "#ffffff"
INK = "#2b3440"
MUTED = "#6b7480"
GOOD = "#2e9d5b"
OK = "#e0a020"
BAD = "#d0453b"
ACCENT = "#3a6fd8"
TRACK = "#e4e0d9"
CHOICE = "#eef2fb"  # program buttons
CHOICE_SELECTED = "#d6e2fa"

STATE_LABELS = {
    State.IDLE: "READY",
    State.SHOW_REFERENCE: "GET READY",
    State.COUNTDOWN: "COUNTDOWN",
    State.PERFORMING: "HOLD THE POSE",
    State.SCORING: "SCORING",
    State.FEEDBACK: "RESULT",
    State.SUMMARY: "SESSION COMPLETE",
}
BUTTON_LABELS = {
    State.IDLE: "Start  [Space]",
    State.SHOW_REFERENCE: "I'm ready  [Space]",
    State.COUNTDOWN: "Skip  [Space]",
    State.PERFORMING: "Finish early  [Space]",
    State.SCORING: "...",
    State.FEEDBACK: "Next  [Space]",
    State.SUMMARY: "Programs  [R]",
}


def score_color(score: Optional[float]) -> str:
    """Color for a 0-100 score (hex string)."""
    if score is None:
        return MUTED
    return GOOD if score >= 80 else OK if score >= 60 else BAD


# Brighter BGR versions of the score colours for text drawn over video.
OVERLAY_COLORS = {GOOD: (120, 235, 140), OK: (60, 210, 255), BAD: (110, 110, 255), MUTED: (170, 170, 170)}


def short_name(name: str) -> str:
    """Pose name without the Sanskrit part, e.g. "Tree Pose (Vrikshasana)" -> "Tree Pose"."""
    return name.split(" (")[0]


@dataclass
class Packet:
    """One processed frame, sent from the worker to the GUI."""

    view: Optional[np.ndarray]  # RGB display image (VIDEO_H x VIDEO_W), mirrored, skeleton drawn
    angles: Optional[dict]  # joint angles for the rules the worker was given, None if no person
    t: float  # monotonic capture time (seconds)
    inference_ms: float
    fps: float
    error: Optional[str] = None


def compose_view(frame: np.ndarray, landmarks: Optional[np.ndarray], rules, angles: Optional[dict],
                 fps: float, inference_ms: float) -> np.ndarray:
    """Build the RGB display image: mirrored frame, skeleton, joint-angle table and FPS."""
    h, w = frame.shape[:2]
    scale = min(VIDEO_W / w, VIDEO_H / h)
    view = cv2.flip(frame, 1)  # mirror for display only; scoring uses the unflipped frame
    if scale != 1.0:
        view = cv2.resize(view, (int(w * scale), int(h * scale)))
    if landmarks is not None:
        lms = mirror_landmarks(landmarks, w)
        lms[:, :2] *= scale
        draw_skeleton(view, lms)
    if view.shape[:2] == (VIDEO_H, VIDEO_W):
        canvas = view
    else:
        canvas = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        y0, x0 = (VIDEO_H - view.shape[0]) // 2, (VIDEO_W - view.shape[1]) // 2
        canvas[y0:y0 + view.shape[0], x0:x0 + view.shape[1]] = view

    if rules:
        row_h = 22
        box = canvas[8:20 + row_h * len(rules), 8:258]
        box[:] = (box * 0.25).astype(np.uint8)  # darken behind the table
        for i, rule in enumerate(rules):
            angle = angles.get(rule.name) if angles else None
            if angle is None:
                value, color = f"--  / {rule.target_angle:.0f}", (150, 150, 150)
            else:
                js = joint_score(angle, rule.target_angle, rule.tolerance)
                value, color = f"{angle:.0f} / {rule.target_angle:.0f}", OVERLAY_COLORS[score_color(js)]
            y = 28 + i * row_h
            cv2.putText(canvas, rule.name.replace("_", " "), (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, value, (166, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    fps_text = f"{fps:4.1f} FPS | {inference_ms:3.0f} ms"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(canvas, (VIDEO_W - tw - 20, 8), (VIDEO_W - 8, 16 + th), (20, 20, 20), -1)
    cv2.putText(canvas, fps_text, (VIDEO_W - tw - 14, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                cv2.LINE_AA)
    if landmarks is None:
        cv2.putText(canvas, "No person detected", (20, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (80, 80, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


class CaptureWorker(threading.Thread):
    """Reads frames, runs pose detection and composes the display image off the GUI thread.

    The GUI sets ``rules`` to the current pose's joint rules; the worker
    computes those angles for every frame. Only the newest packet is kept in
    ``out``; older ones are dropped. Video files are looped and paced at
    their own frame rate.
    """

    def __init__(self, source: Union[int, str], out: "queue.Queue[Packet]"):
        super().__init__(daemon=True)
        self.source = source
        self.out = out
        self.rules: list = []
        self.stop_event = threading.Event()

    def _put(self, packet: Packet) -> None:
        try:
            self.out.get_nowait()
        except queue.Empty:
            pass
        self.out.put_nowait(packet)

    def _error(self, message: str) -> None:
        self._put(Packet(None, None, time.monotonic(), 0.0, 0.0, error=message))

    def run(self) -> None:
        try:
            cam = Camera(self.source)
        except CameraError as e:
            self._error(str(e))
            return
        try:
            detector = PoseDetector()
        except Exception as e:  # missing model file etc.
            cam.release()
            self._error(str(e))
            return
        frame_period = 1.0 / (cam.file_fps() or config.TARGET_FPS) if cam.is_file else 0.0
        failures = 0
        fps = 0.0
        last_t: Optional[float] = None
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                frame = cam.read()
                if frame is None:
                    if cam.is_file:
                        cam.rewind()
                        continue
                    failures += 1
                    if failures > 30:
                        self._error("The camera stopped sending frames.")
                        return
                    time.sleep(0.03)
                    continue
                failures = 0
                t = time.monotonic()
                landmarks = detector.detect(frame, int(t * 1000))
                inference_ms = (time.monotonic() - t) * 1000
                if last_t is not None and t > last_t:
                    fps = 0.9 * fps + 0.1 / (t - last_t) if fps else 1.0 / (t - last_t)
                last_t = t
                rules = self.rules
                angles = joint_angles(landmarks, rules, config.VISIBILITY_THRESHOLD) if landmarks is not None else None
                view = compose_view(frame, landmarks, rules, angles, fps, inference_ms)
                self._put(Packet(view, angles, t, inference_ms, fps))
                if frame_period:
                    time.sleep(max(0.0, frame_period - (time.monotonic() - started)))
        finally:
            detector.close()
            cam.release()


class Meter(tk.Canvas):
    """A horizontal bar with a caption and value text."""

    def __init__(self, master, caption: str, width: int = VIDEO_W, height: int = 34):
        super().__init__(master, width=width, height=height, bg=BG, highlightthickness=0)
        self.w, self.h = width, height
        self.caption = caption

    def set(self, fraction: float, text: str, color: str = ACCENT) -> None:
        """Fill ``fraction`` (0-1) of the bar and show ``text`` on the right."""
        fraction = min(1.0, max(0.0, fraction))
        self.delete("all")
        label_w = 118
        x0, x1 = label_w, self.w - 90
        self.create_text(0, self.h / 2, text=self.caption, anchor="w", fill=MUTED, font=("Segoe UI", 10, "bold"))
        self.create_rectangle(x0, 8, x1, self.h - 8, fill=TRACK, outline="")
        if fraction > 0:
            self.create_rectangle(x0, 8, x0 + (x1 - x0) * fraction, self.h - 8, fill=color, outline="")
        self.create_text(self.w, self.h / 2, text=text, anchor="e", fill=INK, font=("Segoe UI", 13, "bold"))


class App:
    """The main window. Create it, then call ``run()``."""

    def __init__(self, poses: list[Pose], source: Union[int, str] = config.CAMERA_INDEX,
                 root: Optional[tk.Tk] = None, programs: Optional[list[Program]] = None,
                 program_id: Optional[str] = None):
        # Without a programs list, the given poses form a single "All poses" program.
        self.programs = programs or [all_poses_program(poses)]
        self.program = next((p for p in self.programs if p.id == program_id), self.programs[-1])
        self.session = Session(self.program.poses, program_id=self.program.id)
        self.source = source
        self.root = root or tk.Tk()
        self.root.title("Yoga Pose Coach")
        self.root.geometry("1100x700")
        self.root.configure(bg=BG)
        self.root.minsize(1060, 660)

        self.queue: "queue.Queue[Packet]" = queue.Queue(maxsize=1)
        self.worker = CaptureWorker(source, self.queue)
        self.camera_error: Optional[str] = None
        self.last_packet: Optional[Packet] = None
        self.fps = 0.0
        self._shown_state: Optional[tuple] = None
        self.log_path: Optional[Path] = None
        self._ref_cache: dict[str, ImageTk.PhotoImage] = {}
        self._closed = False
        self._after_id: Optional[str] = None  # pending refresh, cancelled on close

        self._build()
        self.root.bind("<space>", lambda e: self.on_space())
        self.root.bind("<Key-q>", lambda e: self.close())
        self.root.bind("<Key-Q>", lambda e: self.close())
        self.root.bind("<Key-r>", lambda e: self.on_restart())
        self.root.bind("<Key-R>", lambda e: self.on_restart())
        for i in range(min(len(self.programs), 9)):
            self.root.bind(f"<Key-{i + 1}>", lambda e, i=i: self.on_program_key(i))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # --- layout --------------------------------------------------------------

    def _build(self) -> None:
        left = tk.Frame(self.root, bg=BG)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        right = tk.Frame(self.root, bg=PANEL, highlightbackground=TRACK, highlightthickness=1)
        right.pack(side="left", fill="both", expand=True, padx=(8, 16), pady=16)

        blank = Image.new("RGB", (VIDEO_W, VIDEO_H), (30, 34, 40))
        self.video_image = ImageTk.PhotoImage(blank)
        tk.Label(left, image=self.video_image, bg="black", bd=0).pack()
        self.hold_meter = Meter(left, "HOLD")
        self.hold_meter.pack(pady=(12, 0))
        self.accuracy_meter = Meter(left, "ACCURACY")
        self.accuracy_meter.pack(pady=(4, 0))
        self.timer_label = tk.Label(left, text="", bg=BG, fg=MUTED, font=("Segoe UI", 10),
                                    wraplength=VIDEO_W - 20, justify="left")
        self.timer_label.pack(anchor="w", pady=(6, 0))
        tk.Label(left, text=f"1-{min(len(self.programs), 9)}: program     Space: start / skip     "
                            "R: programs     Q: quit", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

        pad = {"padx": 20}
        bottom = tk.Frame(right, bg=PANEL)
        bottom.pack(side="bottom", fill="x", pady=14, **pad)
        self.button = tk.Button(bottom, text="", command=self.on_button, takefocus=0, bg=ACCENT, fg="white",
                                activebackground="#2c58b0", activeforeground="white", relief="flat",
                                font=("Segoe UI", 11, "bold"), padx=14, pady=4)
        self.button.pack(side="right")

        self.state_label = tk.Label(right, text="", bg=PANEL, fg=ACCENT, font=("Segoe UI", 10, "bold"))
        self.state_label.pack(anchor="w", pady=(16, 0), **pad)
        self.name_label = tk.Label(right, text="", bg=PANEL, fg=INK, font=("Segoe UI", 15, "bold"),
                                   wraplength=350, justify="left")
        self.name_label.pack(anchor="w", **pad)
        self.progress_label = tk.Label(right, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 10))
        self.progress_label.pack(anchor="w", **pad)

        self.middle = middle = tk.Frame(right, bg=PANEL)
        middle.pack(fill="x", pady=(8, 0), **pad)
        self.ref_label = tk.Label(middle, bg=PANEL, fg=INK, bd=0, font=("Segoe UI", 9), justify="left")
        self.ref_label.pack(side="left")
        self.big_label = tk.Label(middle, text="", bg=PANEL, fg=INK, font=("Segoe UI", 30, "bold"))
        self.big_label.pack(side="left", expand=True)

        self.instruction_label = tk.Label(right, text="", bg=PANEL, fg=INK, font=("Segoe UI", 10),
                                          wraplength=350, justify="left")
        self.instruction_label.pack(anchor="w", pady=(10, 0), **pad)
        self.detail_label = tk.Label(right, text="", bg=PANEL, fg=INK, font=("Segoe UI", 10),
                                     wraplength=350, justify="left")
        self.detail_label.pack(anchor="w", pady=(6, 0), **pad)

        # Program chooser, shown instead of the middle, instruction and detail rows while IDLE.
        self.chooser = tk.Frame(right, bg=PANEL)
        self.program_buttons: dict[str, tk.Button] = {}
        for i, prog in enumerate(self.programs[:9], start=1):
            minutes = max(1, round(prog.duration_seconds() / 60))
            text = f"{i}   {prog.name}  ·  {len(prog.poses)} poses  ·  {minutes} min\n{prog.focus}"
            button = tk.Button(self.chooser, text=text, command=lambda p=prog: self._start_program(p),
                               takefocus=0, justify="left", anchor="w", bg=CHOICE, fg=INK,
                               activebackground=CHOICE_SELECTED, relief="flat", font=("Segoe UI", 10),
                               padx=12, pady=3, wraplength=340)
            button.pack(fill="x", pady=(0, 6))
            self.program_buttons[prog.id] = button
        self.chooser_note = tk.Label(self.chooser, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
                                     wraplength=350, justify="left")
        self.chooser_note.pack(anchor="w", pady=(2, 0))
        self._chooser_visible = False

    def _set_chooser(self, visible: bool) -> None:
        """Swap the program chooser in or out of the right panel, keeping the widget order."""
        if visible == self._chooser_visible:
            return
        self._chooser_visible = visible
        pad = {"padx": 20}
        if visible:
            for widget in (self.middle, self.instruction_label, self.detail_label):
                widget.pack_forget()
            self.chooser.pack(after=self.progress_label, fill="x", pady=(12, 0), **pad)
        else:
            self.chooser.pack_forget()
            self.middle.pack(after=self.progress_label, fill="x", pady=(8, 0), **pad)
            self.instruction_label.pack(after=self.middle, anchor="w", pady=(10, 0), **pad)
            self.detail_label.pack(after=self.instruction_label, anchor="w", pady=(6, 0), **pad)


    def _reference_photo(self, pose: Pose) -> Optional[ImageTk.PhotoImage]:
        """Load (and cache) the pose's reference image scaled to the panel."""
        if pose.id not in self._ref_cache:
            path = config.PROJECT_ROOT / pose.image
            img = cv2.imread(str(path))
            if img is None:
                return None
            scale = min(REF_W / img.shape[1], REF_H / img.shape[0])
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            self._ref_cache[pose.id] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        return self._ref_cache[pose.id]

    # --- input ---------------------------------------------------------------

    def on_space(self) -> None:
        """Space: start the selected program, or skip the current stage."""
        if self.session.state == State.SUMMARY:
            return
        if self.session.state == State.IDLE:
            self._start_program(self.program)
            return
        self.session.skip(time.monotonic())

    def on_restart(self) -> None:
        """R: stop and go back to the program choice. Space then repeats the same program."""
        self.session.restart()
        self.log_path = None

    def on_program_key(self, index: int) -> None:
        """Number keys: start that program from the chooser or the summary."""
        if index < len(self.programs):
            self._start_program(self.programs[index])

    def _start_program(self, program: Program) -> None:
        """Begin ``program`` from its first pose. Ignored while a program is running."""
        if self.session.state not in (State.IDLE, State.SUMMARY):
            return
        self.program = program
        self.session = Session(program.poses, program_id=program.id)
        self.log_path = None
        self._shown_state = None
        self.session.start(time.monotonic())

    def on_button(self) -> None:
        if self.session.state == State.SUMMARY:
            self.on_restart()
        else:
            self.on_space()

    # --- main loop -----------------------------------------------------------

    def run(self) -> None:
        """Start the worker thread and the Tk main loop."""
        self.worker.start()
        self._after_id = self.root.after(UI_INTERVAL_MS, self._update)
        self.root.mainloop()

    def close(self) -> None:
        """Stop the worker and close the window."""
        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.worker.stop_event.set()
        self.worker.join(timeout=2.0)
        self.root.destroy()

    def _update(self) -> None:
        if self._closed:
            return
        now = time.monotonic()
        try:
            self._update_once(now)
        finally:
            spent_ms = int((time.monotonic() - now) * 1000)
            if not self._closed:
                self._after_id = self.root.after(max(1, UI_INTERVAL_MS - spent_ms), self._update)

    def _update_once(self, now: float) -> None:
        """Take the newest packet, advance the session, and redraw."""
        running = self.session.state not in (State.IDLE, State.SUMMARY)
        self.worker.rules = self.session.current_pose.joints if running else []
        packet = None
        try:
            packet = self.queue.get_nowait()
        except queue.Empty:
            pass

        if packet is not None:
            if packet.error:
                self.camera_error = packet.error
            else:
                self.fps = packet.fps
                self.last_packet = packet
                self.session.process_frame(packet.t, packet.angles)

        self.session.tick(now)
        if self.session.state == State.SUMMARY and self.log_path is None:
            self.log_path = save_log(self.session, config.LOGS_DIR)

        if packet is not None and packet.view is not None:
            self._show_view(packet.view, now)
        elif self.camera_error and self.last_packet is None:
            self._draw_message(self.camera_error)
        self._refresh_panel(now)

    # --- drawing -------------------------------------------------------------

    def _draw_message(self, text: str) -> None:
        img = np.full((VIDEO_H, VIDEO_W, 3), (30, 34, 40), dtype=np.uint8)
        cv2.putText(img, "Camera not available", (30, VIDEO_H // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text[:70], (30, VIDEO_H // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)
        self.video_image.paste(Image.fromarray(img))

    def _show_view(self, view: np.ndarray, now: float) -> None:
        """Display a composed RGB frame, adding the countdown number if needed."""
        if self.session.state == State.COUNTDOWN:
            n = str(int(np.ceil(self.session.time_left(now) or 0)))
            size = cv2.getTextSize(n, cv2.FONT_HERSHEY_DUPLEX, 5, 8)[0]
            org = ((VIDEO_W - size[0]) // 2, (VIDEO_H + size[1]) // 2)
            cv2.putText(view, n, org, cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 0), 14, cv2.LINE_AA)
            cv2.putText(view, n, org, cv2.FONT_HERSHEY_DUPLEX, 5, (255, 255, 255), 8, cv2.LINE_AA)
        self.video_image.paste(Image.fromarray(view))

    def _refresh_panel(self, now: float) -> None:
        s = self.session
        pose = s.current_pose
        left = s.time_left(now)

        # Meters update every frame.
        if s.state == State.PERFORMING:
            status = s.last_status
            hold = status.hold_seconds if status else 0.0
            self.hold_meter.set(hold / pose.hold_seconds, f"{hold:4.1f} s", GOOD if status and status.should_score else OK)
            live = (status.window_score if status and status.window_score is not None
                    else s.last_frame.score if s.last_frame else None)
            self.accuracy_meter.set((live or 0) / 100, f"{live:3.0f}%" if live is not None else "--",
                                    score_color(live))
            self.timer_label.config(text=f"Time left: {left:4.1f} s   Best hold: {status.best_hold_seconds if status else 0:.1f} s")
        elif s.state == State.FEEDBACK and s.last_result:
            r = s.last_result
            self.hold_meter.set(r.best_hold_seconds / r.target_hold_seconds, f"{r.best_hold_seconds:4.1f} s", GOOD)
            self.accuracy_meter.set(r.score / 100, f"{r.score:3.0f}%", score_color(r.score))
            self.timer_label.config(text=f"Next pose in {left:.0f} s" if left is not None else "")
        elif s.state != State.SUMMARY:
            self.hold_meter.set(0, "")
            self.accuracy_meter.set(0, "")
            self.timer_label.config(text=f"{left:.0f} s" if left is not None and s.state == State.SHOW_REFERENCE else "")

        if s.state == State.COUNTDOWN:
            self.big_label.config(text=str(int(np.ceil(left or 0))), fg=ACCENT)
        elif s.state == State.PERFORMING:
            self.big_label.config(text=f"{left:.0f}", fg=INK)

        # The rest only changes when the state or pose changes.
        key = (s.state, s.pose_index, self.log_path, self.camera_error, self.program.id)
        if key == self._shown_state:
            return
        self._shown_state = key
        self.state_label.config(text=STATE_LABELS[s.state])
        self.button.config(text=BUTTON_LABELS[s.state])
        self.instruction_label.config(fg=INK)
        self._set_chooser(s.state == State.IDLE)

        if s.state == State.SUMMARY:
            self._show_summary()
            return
        if s.state == State.IDLE:
            self._show_chooser()
            return

        # On the result screen the score and tips need the room, so the reference figure is hidden.
        photo = self._reference_photo(pose) if s.state != State.FEEDBACK else None
        missing = s.state != State.FEEDBACK and photo is None
        self.ref_label.config(image=photo or "", text="(no image)" if missing else "")
        self.name_label.config(text=short_name(pose.name))  # one line, so the panel never overflows
        self.progress_label.config(text=f"Pose {s.pose_index + 1} of {len(s.poses)}"
                                        + ("" if pose.calibrated else "  ·  default targets"))
        self.instruction_label.config(text=pose.instruction)
        self.detail_label.config(text="", fg=INK)
        if s.state == State.SHOW_REFERENCE:
            self.big_label.config(text="")
            self.detail_label.config(text="Get into position. Press Space when you are ready.", fg=MUTED)
        elif s.state == State.PERFORMING:
            self.detail_label.config(text="Hold still in the pose. The bar turns green once your hold counts.",
                                     fg=MUTED)
        elif s.state == State.FEEDBACK and s.last_result:
            r = s.last_result
            self.big_label.config(text=f"{r.score:.0f}%", fg=score_color(r.score))
            lines = [f"Held {r.best_hold_seconds:.1f} s of {r.target_hold_seconds:.0f} s"]
            if not r.held:
                lines[0] += f"  (no steady hold, so the score is capped at {config.NO_HOLD_SCORE_CAP:.0f}%)"
            lines += [f"•  {tip}" for tip in r.tips] or ["•  Nice work, no major corrections."]
            self.instruction_label.config(text="\n".join(lines))

    def _show_chooser(self) -> None:
        """The IDLE screen: one button per program, the selected one highlighted."""
        self.name_label.config(text="Choose a program")
        self.progress_label.config(text=f"Press 1-{min(len(self.programs), 9)} or click one")
        self.button.config(text=f"Start {self.program.name}  [Space]")
        for pid, button in self.program_buttons.items():
            button.config(bg=CHOICE_SELECTED if pid == self.program.id else CHOICE)
        self.chooser_note.config(text=f"Camera problem: {self.camera_error}" if self.camera_error else "")

    def _show_summary(self) -> None:
        summary = self.session.summary()
        self.name_label.config(text=f"Summary: {self.program.name}")
        longest_name, longest_seconds = summary.longest_hold
        self.progress_label.config(text=f"{len(summary.pose_scores)} poses  ·  longest hold: "
                                        f"{short_name(longest_name)}, {longest_seconds:.1f} s")
        self.big_label.config(text=f"{summary.total_score:.0f}%", fg=score_color(summary.total_score))
        # Per-pose rows sit beside the total score, so a long sequence still fits the panel.
        rows = [f"{short_name(name)}:  {score:.0f}%  ({hold:.1f} s)" for name, score, hold in summary.pose_scores]
        self.ref_label.config(image="", text="\n".join(rows))
        self.instruction_label.config(text="")
        tips = [f"•  {tip}" + (f"  (x{count})" if count > 1 else "") for tip, count in summary.common_mistakes]
        text = "Most common mistakes:\n" + "\n".join(tips) if tips else "No corrections needed. Well done!"
        self.detail_label.config(text=text, fg=INK)
        self.hold_meter.set(0, "")
        self.accuracy_meter.set(summary.total_score / 100, f"{summary.total_score:.0f}%", score_color(summary.total_score))
        saved = f"Saved to logs/{self.log_path.name}.  " if self.log_path else ""
        self.timer_label.config(text=saved + "Press R to choose a program or Q to quit.")
