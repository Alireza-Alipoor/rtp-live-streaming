"""Read and decode video frames from a local media file using PyAV.

This module is responsible only for file input. It does not resize, encode,
packetize, or transmit frames. The returned ``av.VideoFrame`` objects can be
passed directly to the project's ``VideoEncoder``.

PyAV is imported lazily so importing this module does not require PyAV to be
installed. Constructing :class:`VideoFileCapture` does require it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class VideoCaptureError(RuntimeError):
    """Raised when a video file cannot be opened or decoded."""


class VideoFileCapture:
    """Decode video frames from a file.

    Args:
        source: Path to a video file supported by FFmpeg.
        stream_index: Zero-based index among the file's video streams.
        realtime: If true, pace returned frames according to their media
            timestamps. If false, decode as quickly as possible.
        loop: If true, reopen and decode the file again after reaching EOF.

    ``read()`` returns the next decoded ``av.VideoFrame`` or ``None`` at EOF.
    ``frames()`` provides the same operation as an iterator. The capture owns
    its PyAV container and should be closed explicitly or used as a context
    manager.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        stream_index: int = 0,
        realtime: bool = False,
        loop: bool = False,
    ) -> None:
        if not isinstance(source, (str, Path)):
            raise TypeError("source must be a string or pathlib.Path")
        if not isinstance(stream_index, int) or isinstance(stream_index, bool):
            raise TypeError("stream_index must be an integer")
        if stream_index < 0:
            raise ValueError("stream_index cannot be negative")
        if not isinstance(realtime, bool):
            raise TypeError("realtime must be a bool")
        if not isinstance(loop, bool):
            raise TypeError("loop must be a bool")

        self._source = Path(source).expanduser()
        if not self._source.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self._source}")

        self._stream_index = stream_index
        self._realtime = realtime
        self._loop = loop
        self._av = self._import_av()
        self._container: Any | None = None
        self._stream: Any | None = None
        self._decoded_frames: Iterator[Any] | None = None
        self._closed = False
        self._playback_started: float | None = None
        self._first_media_time: float | None = None
        self._frame_index = 0

        self._open()

    @staticmethod
    def _import_av() -> Any:
        try:
            import av
        except ImportError as exc:
            raise VideoCaptureError(
                "PyAV is required for video capture. Install it with `uv add av`."
            ) from exc
        return av

    def _open(self) -> None:
        """Open the file and prepare its selected video stream."""
        container = None
        try:
            container = self._av.open(str(self._source), mode="r")
            video_streams = container.streams.video
            if self._stream_index >= len(video_streams):
                raise VideoCaptureError(
                    f"Video stream index {self._stream_index} is unavailable; "
                    f"the file contains {len(video_streams)} video stream(s)"
                )

            stream = video_streams[self._stream_index]
            stream.thread_type = "AUTO"
            self._container = container
            self._stream = stream
            self._decoded_frames = iter(container.decode(stream))
            self._reset_playback_clock()
        except VideoCaptureError:
            if container is not None:
                container.close()
            raise
        except Exception as exc:
            if container is not None:
                container.close()
            raise VideoCaptureError(
                f"Could not open video file {self._source}: {exc}"
            ) from exc

    def _reset_playback_clock(self) -> None:
        self._playback_started = None
        self._first_media_time = None
        self._frame_index = 0

    @property
    def source(self) -> Path:
        """Path of the input file."""
        return self._source

    @property
    def width(self) -> int:
        """Coded width of the selected video stream."""
        self._ensure_open()
        return int(self._stream.codec_context.width)

    @property
    def height(self) -> int:
        """Coded height of the selected video stream."""
        self._ensure_open()
        return int(self._stream.codec_context.height)

    @property
    def average_rate(self) -> Any | None:
        """Average frame rate reported by the container, if available."""
        self._ensure_open()
        return self._stream.average_rate

    @property
    def duration(self) -> float | None:
        """Duration of the selected stream in seconds, if known."""
        self._ensure_open()
        if self._stream.duration is None or self._stream.time_base is None:
            return None
        return float(self._stream.duration * self._stream.time_base)

    @property
    def is_closed(self) -> bool:
        """Whether the capture has been closed."""
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed or self._container is None:
            raise VideoCaptureError("video capture is closed")

    def _media_time(self, frame: Any) -> float:
        """Get a frame's media timestamp, with an FPS-based fallback."""
        if frame.pts is not None and frame.time_base is not None:
            return float(frame.pts * frame.time_base)

        rate = self._stream.average_rate
        if rate is not None and float(rate) > 0:
            return self._frame_index / float(rate)

        # Without timestamps or a usable frame rate, do not invent a delay.
        return 0.0

    def _pace(self, frame: Any) -> None:
        if not self._realtime:
            return

        media_time = self._media_time(frame)
        now = time.monotonic()
        if self._playback_started is None:
            self._playback_started = now
            self._first_media_time = media_time
            return

        assert self._first_media_time is not None
        target = self._playback_started + max(0.0, media_time - self._first_media_time)
        delay = target - now
        if delay > 0:
            time.sleep(delay)

    def read(self) -> Any | None:
        """Return the next decoded video frame, or ``None`` at end-of-file."""
        self._ensure_open()
        assert self._decoded_frames is not None

        try:
            frame = next(self._decoded_frames)
        except StopIteration:
            if not self._loop:
                return None

            self._close_container()
            self._open()
            assert self._decoded_frames is not None
            try:
                frame = next(self._decoded_frames)
            except StopIteration:
                # An empty video stream must not cause an infinite loop.
                return None
            except Exception as exc:
                raise VideoCaptureError(f"Failed to decode video frame: {exc}") from exc
        except Exception as exc:
            raise VideoCaptureError(f"Failed to decode video frame: {exc}") from exc

        self._pace(frame)
        self._frame_index += 1
        return frame

    def frames(self) -> Iterator[Any]:
        """Yield decoded frames until EOF or until the capture is closed."""
        while not self._closed:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def _close_container(self) -> None:
        container = self._container
        self._container = None
        self._stream = None
        self._decoded_frames = None
        if container is not None:
            try:
                container.close()
            except Exception as exc:
                raise VideoCaptureError(f"Could not close video file: {exc}") from exc

    def close(self) -> None:
        """Close the input container. Safe to call more than once."""
        if self._closed:
            return
        try:
            self._close_container()
        finally:
            self._closed = True

    def __iter__(self) -> Iterator[Any]:
        return self.frames()

    def __enter__(self) -> VideoFileCapture:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
