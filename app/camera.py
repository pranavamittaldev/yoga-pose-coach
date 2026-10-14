"""Webcam capture wrapper around cv2.VideoCapture."""

from __future__ import annotations

import sys
from typing import Optional, Union

import cv2
import numpy as np

import config


class CameraError(RuntimeError):
    """Raised when the camera (or video file) cannot be opened."""


class Camera:
    """Open a webcam by index (or a video file by path) and read BGR frames.

    Frames are only held in memory; nothing is written to disk.
    Use as a context manager or call ``release()`` when done.
    """

    def __init__(
        self,
        source: Union[int, str] = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        fps: int = config.TARGET_FPS,
    ):
        self.source = source
        if isinstance(source, int) and sys.platform == "win32":
            # DirectShow opens much faster than the default MSMF backend on Windows.
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(source)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open camera/video source {source!r}")
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)

    @property
    def is_file(self) -> bool:
        """True if reading from a video file rather than a live camera."""
        return not isinstance(self.source, int)

    def file_fps(self) -> float:
        """Frame rate reported by the source (useful for video files)."""
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if no frame is available."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def rewind(self) -> None:
        """Restart a video file from the first frame (no effect on a live camera)."""
        if self.is_file:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        """Release the underlying capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
