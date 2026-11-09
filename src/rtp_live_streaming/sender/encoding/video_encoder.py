"""Video encoding for the RTP streaming sender.

PyAV/FFmpeg performs video encoding.  This module deliberately does not
packetize the encoded data or send it over the network: callers should pass
H.264 access units to ``sender.packetization.h264`` (and VP8 packets to the
VP8 packetizer) before handing them to ``RTPSender``.

The encoder accepts :class:`av.VideoFrame` objects and returns encoded FFmpeg
packets together with an RTP-compatible 90 kHz timestamp.  PyAV is imported
lazily so importing the rest of the application does not require PyAV to be
installed; constructing a VideoEncoder does.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from rtp_live_streaming.common.config import VideoConfig
from rtp_live_streaming.common.payload_types import Codec

VIDEO_CLOCK_RATE = 90_000


class VideoEncoderError(RuntimeError):
    """Raised when the video encoder cannot be created or used."""


@dataclass(frozen=True, slots=True)
class EncodedVideoPacket:
    """One encoded FFmpeg packet.

    ``data`` is codec output, not an RTP packet. For H.264, the encoder is
    configured to produce Annex-B byte-stream output suitable for the
    project's H.264 packetizer.

    ``timestamp`` is expressed in the 90 kHz RTP video clock. Multiple
    packets emitted for the same input frame share the same timestamp.
    """

    data: bytes
    timestamp: int
    pts: int | None = None
    dts: int | None = None
    duration: int | None = None
    is_keyframe: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("data must be bytes")
        if not self.data:
            raise ValueError("encoded packet data cannot be empty")
        if not 0 <= self.timestamp <= 0xFFFFFFFF:
            raise ValueError("timestamp must be an unsigned 32-bit integer")


class VideoEncoder:
    """Encode raw PyAV video frames as H.264 or VP8.

    Args:
        config: Validated video settings. Defaults to ``VideoConfig()``.
        options: Optional FFmpeg codec options. These override the low-latency
            defaults for the selected codec.

    Notes:
        * Input frames are resized to the configured dimensions and converted
          to ``yuv420p`` when necessary.
        * Input frame PTS values use a time base of ``1 / fps``. If a frame has
          no PTS, a monotonically increasing PTS is assigned.
        * The RTP video clock is always 90 kHz for both H.264 and VP8.
        * ``flush()`` must be called at end-of-stream to retrieve delayed
          encoder output.
    """

    _ENCODERS: dict[Codec, str] = {
        Codec.H264: "libx264",
        Codec.VP8: "libvpx",
    }

    def __init__(
        self,
        config: VideoConfig | None = None,
        *,
        options: dict[str, str] | None = None,
    ) -> None:
        self.config = config or VideoConfig()
        if not isinstance(self.config, VideoConfig):
            raise TypeError("config must be a VideoConfig instance")

        self._av = self._import_av()
        self._options = dict(options or {})
        self._frame_index = 0
        self._closed = False
        self._flushed = False
        self._context = self._create_context()

    @staticmethod
    def _import_av() -> Any:
        try:
            import av
        except ImportError as exc:
            raise VideoEncoderError(
                "PyAV is required for video encoding. Install it with `uv add av`."
            ) from exc
        return av

    def _codec_options(self) -> dict[str, str]:
        """Return codec-specific low-latency options plus user overrides."""
        if self.config.codec == Codec.H264:
            defaults = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "profile": "baseline",
                "annexb": "1",
                "repeat-headers": "1",
            }
        else:
            defaults = {
                "deadline": "realtime",
                "cpu-used": "8",
                "lag-in-frames": "0",
                "auto-alt-ref": "0",
            }

        defaults.update(self._options)
        return defaults

    def _create_context(self) -> Any:
        codec_name = self._ENCODERS[self.config.codec]

        try:
            context = self._av.CodecContext.create(codec_name, "w")
            context.width = self.config.width
            context.height = self.config.height
            context.pix_fmt = "yuv420p"
            context.bit_rate = self.config.bitrate_kbps * 1000
            context.time_base = Fraction(1, self.config.fps)
            context.framerate = Fraction(self.config.fps, 1)
            context.options = self._codec_options()
            context.open()
        except Exception as exc:
            raise VideoEncoderError(
                f"Could not initialize {self.config.codec.value} encoder "
                f"({codec_name}): {exc}"
            ) from exc

        return context

    @property
    def codec(self) -> Codec:
        """Currently configured video codec."""
        return self.config.codec

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def fps(self) -> int:
        return self.config.fps

    @property
    def is_closed(self) -> bool:
        return self._closed

    @staticmethod
    def _to_rtp_timestamp(pts: int, time_base: Fraction) -> int:
        # Integer arithmetic avoids floating-point drift over long streams.
        ticks = Fraction(pts) * time_base * VIDEO_CLOCK_RATE
        return int(ticks) & 0xFFFFFFFF

    def _prepare_frame(self, frame: Any) -> Any:
        if self._closed or self._flushed:
            raise VideoEncoderError("encoder is already flushed or closed")

        video_frame_type = self._av.VideoFrame
        if not isinstance(frame, video_frame_type):
            raise TypeError("frame must be an av.VideoFrame")

        if frame.width != self.config.width or frame.height != self.config.height:
            frame = frame.reformat(
                width=self.config.width,
                height=self.config.height,
                format="yuv420p",
            )
        elif frame.format.name != "yuv420p":
            frame = frame.reformat(format="yuv420p")

        if frame.pts is None:
            frame.pts = self._frame_index
        frame.time_base = Fraction(1, self.config.fps)
        self._frame_index = max(self._frame_index + 1, frame.pts + 1)
        return frame

    def _convert_packets(self, packets: Any) -> tuple[EncodedVideoPacket, ...]:
        converted: list[EncodedVideoPacket] = []

        for packet in packets:
            data = bytes(packet)
            if not data:
                continue

            pts = packet.pts
            packet_time_base = packet.time_base or self._context.time_base
            if pts is None:
                # FFmpeg normally propagates PTS. This fallback is primarily
                # for unusual encoders that omit it; it remains monotonic.
                timestamp = (
                    (self._frame_index - 1) * VIDEO_CLOCK_RATE // self.config.fps
                )
            else:
                timestamp = self._to_rtp_timestamp(pts, Fraction(packet_time_base))

            duration = packet.duration if packet.duration not in (None, 0) else None
            converted.append(
                EncodedVideoPacket(
                    data=data,
                    timestamp=timestamp,
                    pts=pts,
                    dts=packet.dts,
                    duration=duration,
                    is_keyframe=bool(packet.is_keyframe),
                )
            )

        return tuple(converted)

    def encode(self, frame: Any) -> tuple[EncodedVideoPacket, ...]:
        """Encode one raw video frame.

        An encoder may buffer frames, so an empty tuple is a valid result.
        """
        prepared = self._prepare_frame(frame)
        try:
            packets = self._context.encode(prepared)
        except Exception as exc:
            raise VideoEncoderError(f"Failed to encode video frame: {exc}") from exc
        return self._convert_packets(packets)

    def flush(self) -> tuple[EncodedVideoPacket, ...]:
        """Flush delayed packets and prevent further frames from being encoded."""
        if self._closed or self._flushed:
            return ()

        try:
            packets = self._context.encode(None)
        except Exception as exc:
            raise VideoEncoderError(f"Failed to flush video encoder: {exc}") from exc
        finally:
            self._flushed = True

        return self._convert_packets(packets)

    def reconfigure(
        self,
        config: VideoConfig,
        *,
        options: dict[str, str] | None = None,
    ) -> tuple[EncodedVideoPacket, ...]:
        """Flush the current encoder and switch to a new configuration.

        Returns any delayed packets from the old encoder. The caller is
        responsible for coordinating the corresponding codec/payload-type
        change with the receiver before sending packets from the new encoder.
        This method does not reset RTP sequence numbers or SSRC; those belong
        to ``RTPSender``.
        """
        if self._closed:
            raise VideoEncoderError("cannot reconfigure a closed encoder")
        if not isinstance(config, VideoConfig):
            raise TypeError("config must be a VideoConfig instance")

        old_packets = self.flush()
        self.config = config
        if options is not None:
            self._options = dict(options)
        self._frame_index = 0
        self._flushed = False
        self._context = self._create_context()
        return old_packets

    def close(self) -> tuple[EncodedVideoPacket, ...]:
        """Flush and close the encoder. Safe to call more than once."""
        if self._closed:
            return ()

        packets = self.flush()
        self._closed = True
        return packets

    def __enter__(self) -> VideoEncoder:
        if self._closed:
            raise VideoEncoderError("encoder is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
