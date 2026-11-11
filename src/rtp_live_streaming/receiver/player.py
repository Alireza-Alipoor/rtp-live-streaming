"""Receive, depacketize, decode, and optionally display an RTP video stream.

The player composes the project's UDP :class:`RTPReceiver`, H.264
:class:`H264Depacketizer`, and :class:`VideoDecoder`. Decoded frames are sent
to an optional ``on_frame`` callback. A simple OpenCV window can be enabled
with ``display=True``; OpenCV remains an optional dependency.

At present, the receiver-side video decoding pipeline supports H.264 only.
The project has a VP8 payload type, but a VP8 depacketizer/decoder is not part
of the current implementation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rtp_live_streaming.common.payload_types import Codec
from rtp_live_streaming.common.rtp_packet import RTPPacket
from rtp_live_streaming.receiver.decoding.video_decoder import (
    VideoDecoder,
    VideoDecoderError,
)
from rtp_live_streaming.receiver.depacketization.h264 import (
    H264DepacketizationError,
    H264Depacketizer,
)
from rtp_live_streaming.receiver.rtp_receiver import (
    RTPReceiver,
    RTPReceiverError,
)


class VideoPlayerError(RuntimeError):
    """Raised when the video player cannot start or operate."""


class UnsupportedVideoCodecError(VideoPlayerError):
    """Raised when a received payload type is not currently decodable video."""


FrameCallback = Callable[[Any], None]
ErrorCallback = Callable[[Exception], None]


@dataclass(frozen=True, slots=True)
class VideoPlayerStats:
    """Snapshot of the player-level processing counters."""

    packets_received: int
    access_units_completed: int
    frames_decoded: int
    decode_errors: int
    callback_errors: int


@dataclass(slots=True)
class _H264Source:
    """Depacketizer and decoder state for one RTP synchronization source."""

    depacketizer: H264Depacketizer
    decoder: VideoDecoder


class VideoPlayer:
    """Receive an RTP/H.264 stream and deliver decoded video frames.

    Args:
        bind_host: Local interface to bind the RTP UDP socket to.
        bind_port: Local UDP port to receive video RTP packets on.
        jitter_buffer_ms: Receiver playout/reordering delay in milliseconds.
        on_frame: Optional callback invoked with each decoded PyAV VideoFrame.
            The callback executes on the RTP receiver's background thread.
        on_error: Optional callback for receive, depacketization, decoding,
            display, or frame-callback errors.
        display: If true, display frames in an OpenCV window. Requires
            ``opencv-python`` (or a compatible OpenCV installation).
        window_name: Title for the optional OpenCV window.
        decoder_options: Optional FFmpeg H.264 decoder options passed to
            :class:`VideoDecoder`.
        payload_type_map: Optional RTP payload type overrides passed to
            :class:`RTPReceiver`.

    Call :meth:`start` to run reception in the background, or :meth:`run` to
    block until interrupted. The class owns its RTP receiver and decoder
    contexts. Use :meth:`close` or a context manager to release them.

    Notes:
        - The current decoding path supports H.264 only.
        - ``on_frame`` runs on the receiver thread and should return quickly.
        - RTP timestamps are passed unchanged to ``VideoDecoder`` (90 kHz).
    """

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        bind_port: int = 5004,
        *,
        jitter_buffer_ms: float = 50,
        on_frame: FrameCallback | None = None,
        on_error: ErrorCallback | None = None,
        display: bool = False,
        window_name: str = "RTP Video",
        decoder_options: dict[str, str] | None = None,
        payload_type_map: dict[int, Codec | str] | None = None,
        max_datagram_size: int = 65507,
        socket_buffer_size: int = 1_048_576,
        receive_timeout: float = 0.02,
    ) -> None:
        if on_frame is not None and not callable(on_frame):
            raise TypeError("on_frame must be callable or None")
        if on_error is not None and not callable(on_error):
            raise TypeError("on_error must be callable or None")
        if not isinstance(display, bool):
            raise TypeError("display must be a bool")
        if not isinstance(window_name, str) or not window_name:
            raise ValueError("window_name must be a non-empty string")
        if decoder_options is not None and not isinstance(decoder_options, dict):
            raise TypeError("decoder_options must be a dictionary or None")

        self.on_frame = on_frame
        self.on_error = on_error
        self.display = display
        self.window_name = window_name
        self._decoder_options = dict(decoder_options or {})
        self._cv2: Any | None = None
        self._window_created = False
        self._closed = False
        self._run_stop_event = threading.Event()
        self._lock = threading.RLock()
        self._sources: dict[int, _H264Source] = {}

        self._packets_received = 0
        self._access_units_completed = 0
        self._frames_decoded = 0
        self._decode_errors = 0
        self._callback_errors = 0

        if self.display:
            self._cv2 = self._import_cv2()

        try:
            self._receiver = RTPReceiver(
                bind_host,
                bind_port,
                jitter_buffer_ms=jitter_buffer_ms,
                max_datagram_size=max_datagram_size,
                socket_buffer_size=socket_buffer_size,
                receive_timeout=receive_timeout,
                on_packet=self._handle_packet,
                on_error=self._report_error,
                payload_type_map=payload_type_map,
            )
        except (OSError, ValueError, TypeError, RTPReceiverError) as exc:
            raise VideoPlayerError(
                f"Could not initialize RTP video receiver: {exc}"
            ) from exc

    @staticmethod
    def _import_cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise VideoPlayerError(
                "OpenCV is required when display=True. "
                "Install it with `uv add opencv-python`."
            ) from exc
        return cv2

    @property
    def address(self) -> tuple[str, int]:
        """Bound local (host, port), including the assigned port if port is 0."""
        return self._receiver.address

    @property
    def is_closed(self) -> bool:
        """Whether the player has been closed."""
        return self._closed

    @property
    def stats(self) -> VideoPlayerStats:
        """Return a thread-safe snapshot of player processing counters."""
        with self._lock:
            return VideoPlayerStats(
                packets_received=self._packets_received,
                access_units_completed=self._access_units_completed,
                frames_decoded=self._frames_decoded,
                decode_errors=self._decode_errors,
                callback_errors=self._callback_errors,
            )

    def start(self, *, daemon: bool = True) -> None:
        """Start receiving RTP packets on a background thread."""
        if self._closed:
            raise VideoPlayerError("Cannot start a closed video player")
        self._run_stop_event.clear()
        try:
            self._receiver.start(daemon=daemon)
        except RTPReceiverError as exc:
            raise VideoPlayerError(
                f"Could not start RTP video receiver: {exc}"
            ) from exc

    def run(self) -> None:
        """Run until Ctrl+C, then stop and close the player."""
        self.start()
        try:
            while not self._run_stop_event.wait(0.25):
                if self._closed:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def _get_source(self, packet: RTPPacket) -> _H264Source:
        codec = self._receiver.codec_for_payload_type(packet.payload_type)
        if codec is not Codec.H264:
            raise UnsupportedVideoCodecError(
                f"Received {codec.value} payload type {packet.payload_type}; "
                "the current VideoPlayer supports H.264 only"
            )

        source = self._sources.get(packet.ssrc)
        if source is None:
            try:
                source = _H264Source(
                    depacketizer=H264Depacketizer(),
                    decoder=VideoDecoder(options=self._decoder_options),
                )
            except VideoDecoderError:
                raise
            self._sources[packet.ssrc] = source
        return source

    def _handle_packet(self, packet: RTPPacket) -> None:
        """Depacketize and decode one packet released by the jitter buffer."""
        with self._lock:
            self._packets_received += 1

        try:
            source = self._get_source(packet)
            access_unit = source.depacketizer.push(
                packet.payload,
                marker=packet.marker,
            )
            if access_unit is None:
                return

            with self._lock:
                self._access_units_completed += 1
            frames = source.decoder.decode_access_unit(
                access_unit,
                timestamp=packet.timestamp,
            )
        except (
            H264DepacketizationError,
            VideoDecoderError,
            UnsupportedVideoCodecError,
            ValueError,
        ) as exc:
            with self._lock:
                self._decode_errors += 1
            self._report_error(exc)
            # Discard a partial access unit after a malformed or undecodable
            # packet so fragments from separate access units cannot combine.
            source = self._sources.get(packet.ssrc)
            if source is not None:
                source.depacketizer.reset()
            return
        except Exception as exc:
            with self._lock:
                self._decode_errors += 1
            self._report_error(
                VideoPlayerError(f"Unexpected video processing error: {exc}")
            )
            return

        for frame in frames:
            with self._lock:
                self._frames_decoded += 1
            self._deliver_frame(frame)

    def _deliver_frame(self, frame: Any) -> None:
        if self.on_frame is not None:
            try:
                self.on_frame(frame)
            except Exception as exc:
                with self._lock:
                    self._callback_errors += 1
                self._report_error(VideoPlayerError(f"on_frame callback failed: {exc}"))

        if self.display:
            try:
                image = frame.to_ndarray(format="bgr24")
                self._cv2.imshow(self.window_name, image)
                self._window_created = True
                self._cv2.waitKey(1)
            except Exception as exc:
                with self._lock:
                    self._callback_errors += 1
                self._report_error(
                    VideoPlayerError(f"Could not display video frame: {exc}")
                )

    def _report_error(self, error: Exception) -> None:
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception:
                # An error callback must never terminate the RTP receive loop.
                pass

    def stop(self, *, flush: bool = False) -> None:
        """Stop receiving. Optionally process packets still in the jitter buffer."""
        if self._closed:
            return
        self._run_stop_event.set()
        try:
            self._receiver.stop(flush=flush)
        except Exception as exc:
            self._report_error(VideoPlayerError(f"Could not stop RTP receiver: {exc}"))

    def close(self) -> None:
        """Stop reception, drain decoder delay, and release all resources."""
        if self._closed:
            return

        self._run_stop_event.set()
        try:
            self._receiver.stop(flush=True)
        except Exception as exc:
            self._report_error(VideoPlayerError(f"Could not stop RTP receiver: {exc}"))

        for source in tuple(self._sources.values()):
            try:
                for frame in source.decoder.flush():
                    with self._lock:
                        self._frames_decoded += 1
                    self._deliver_frame(frame)
                source.decoder.close()
            except VideoDecoderError as exc:
                self._report_error(exc)

        self._sources.clear()
        self._receiver.close()

        if self._window_created and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self._closed = True

    def __enter__(self) -> VideoPlayer:
        if self._closed:
            raise VideoPlayerError("Cannot enter a closed video player")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "VideoPlayer",
    "VideoPlayerError",
    "VideoPlayerStats",
    "UnsupportedVideoCodecError",
]
