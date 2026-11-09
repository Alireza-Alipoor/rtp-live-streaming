"""H.264 video decoding for the RTP streaming receiver.

``H264Depacketizer.push()`` returns a tuple of complete H.264 NAL units when
an RTP access unit is complete. Pass that tuple directly to
:meth:`VideoDecoder.decode_access_unit`.

The depacketizer deliberately returns NAL units *without* Annex-B start
codes. This decoder adds the four-byte start code required by the configured
FFmpeg H.264 byte-stream decoder, wraps the access unit in an ``av.Packet``,
and returns the decoded ``av.VideoFrame`` objects.

PyAV is imported lazily, so importing this module does not require PyAV.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Any

VIDEO_CLOCK_RATE = 90_000
ANNEX_B_START_CODE = b"\x00\x00\x00\x01"


class VideoDecoderError(RuntimeError):
    """Raised when the H.264 decoder cannot be created or used."""


class VideoDecoder:
    """Decode H.264 access units produced by ``H264Depacketizer``.

    Args:
        options: Optional FFmpeg decoder options.

    Example::

        depacketizer = H264Depacketizer()
        decoder = VideoDecoder()

        access_unit = depacketizer.push(rtp_payload, marker=rtp_marker)
        if access_unit is not None:
            frames = decoder.decode_access_unit(access_unit, timestamp=rtp_ts)

    ``timestamp`` is the RTP video timestamp (90 kHz clock). It is attached
    to the compressed packet as its PTS/time base; FFmpeg may reorder frames,
    so the frames returned by a call are not guaranteed to correspond one to
    one with the access unit supplied in that call.
    """

    def __init__(self, *, options: dict[str, str] | None = None) -> None:
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary")
        self._options = dict(options or {})
        self._av = self._import_av()
        self._closed = False
        self._flushed = False
        self._context = self._create_context()

    @staticmethod
    def _import_av() -> Any:
        try:
            import av
        except ImportError as exc:
            raise VideoDecoderError(
                "PyAV is required for video decoding. Install it with `uv add av`."
            ) from exc
        return av

    def _create_context(self) -> Any:
        try:
            context = self._av.CodecContext.create("h264", "r")
            if self._options:
                context.options = dict(self._options)
            context.open()
        except Exception as exc:
            raise VideoDecoderError(
                f"Could not initialize H.264 decoder: {exc}"
            ) from exc
        return context

    @property
    def is_closed(self) -> bool:
        """Whether the decoder has been closed."""
        return self._closed

    @property
    def is_flushed(self) -> bool:
        """Whether end-of-stream has been sent to the decoder."""
        return self._flushed

    @staticmethod
    def _annexb_access_unit(nal_units: Iterable[bytes]) -> bytes:
        """Join depacketized NAL units into an Annex-B access unit."""
        if isinstance(nal_units, (bytes, bytearray, memoryview)):
            raise TypeError(
                "nal_units must be an iterable of NAL-unit bytes, not bytes"
            )

        try:
            units = tuple(nal_units)
        except TypeError as exc:
            raise TypeError("nal_units must be an iterable of bytes") from exc

        if not units:
            raise ValueError("access unit must contain at least one NAL unit")

        normalized: list[bytes] = []
        for index, nal in enumerate(units):
            if not isinstance(nal, bytes):
                raise TypeError(f"NAL unit at index {index} must be bytes")
            if not nal:
                raise ValueError(f"NAL unit at index {index} cannot be empty")
            # H264Depacketizer's contract excludes start codes. Reject them
            # rather than silently generating malformed/doubled prefixes.
            if nal.startswith(b"\x00\x00\x01") or nal.startswith(ANNEX_B_START_CODE):
                raise ValueError(
                    f"NAL unit at index {index} already contains an Annex-B start code"
                )
            if nal[0] & 0x80:
                raise ValueError(
                    f"NAL unit at index {index} has forbidden_zero_bit set"
                )
            normalized.append(ANNEX_B_START_CODE + nal)

        return b"".join(normalized)

    @staticmethod
    def _validate_timestamp(timestamp: int | None) -> int | None:
        if timestamp is None:
            return None
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise TypeError("timestamp must be an integer or None")
        if not 0 <= timestamp <= 0xFFFFFFFF:
            raise ValueError("timestamp must be an unsigned 32-bit integer")
        return timestamp

    def decode_access_unit(
        self,
        nal_units: Iterable[bytes],
        timestamp: int | None = None,
    ) -> tuple[Any, ...]:
        """Decode one complete access unit from ``H264Depacketizer``.

        Args:
            nal_units: Tuple/list of complete NAL units, without start codes.
            timestamp: Optional RTP timestamp in the 90 kHz video clock.

        Returns:
            Zero or more PyAV ``VideoFrame`` objects. Zero frames is valid
            because the decoder may buffer/reorder compressed frames.

        Raises:
            TypeError/ValueError: Invalid input.
            VideoDecoderError: Decoder is unavailable, closed, flushed, or
                FFmpeg reports a decoding error.
        """
        if self._closed:
            raise VideoDecoderError("decoder is closed")
        if self._flushed:
            raise VideoDecoderError(
                "decoder has been flushed; call reset() before decoding"
            )

        timestamp = self._validate_timestamp(timestamp)
        encoded = self._annexb_access_unit(nal_units)

        try:
            packet = self._av.Packet(encoded)
            if timestamp is not None:
                packet.pts = timestamp
                packet.dts = timestamp
                packet.time_base = Fraction(1, VIDEO_CLOCK_RATE)
            frames = self._context.decode(packet)
        except Exception as exc:
            raise VideoDecoderError(
                f"Failed to decode H.264 access unit: {exc}"
            ) from exc

        return tuple(frames)

    def decode(
        self,
        nal_units: Iterable[bytes],
        timestamp: int | None = None,
    ) -> tuple[Any, ...]:
        """Alias for :meth:`decode_access_unit`."""
        return self.decode_access_unit(nal_units, timestamp)

    def flush(self) -> tuple[Any, ...]:
        """Drain delayed frames at end-of-stream; safe to call repeatedly."""
        if self._closed or self._flushed:
            return ()

        try:
            frames = self._context.decode(None)
        except Exception as exc:
            raise VideoDecoderError(f"Failed to flush H.264 decoder: {exc}") from exc
        finally:
            self._flushed = True

        return tuple(frames)

    def reset(self) -> None:
        """Discard decoder state and create a fresh H.264 decoding context.

        Use this after a stream discontinuity or when starting a new H.264
        stream. It does not perform RTP codec negotiation.
        """
        if self._closed:
            raise VideoDecoderError("cannot reset a closed decoder")
        self._context = self._create_context()
        self._flushed = False

    def close(self) -> tuple[Any, ...]:
        """Flush delayed frames and close the decoder; safe to call repeatedly."""
        if self._closed:
            return ()
        frames = self.flush()
        self._closed = True
        return frames

    def __enter__(self) -> VideoDecoder:
        if self._closed:
            raise VideoDecoderError("decoder is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
