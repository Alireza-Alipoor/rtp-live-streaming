"""Shared configuration for the RTP streaming application.

Contains configuration models for network transport, video, audio,
and receiver buffering.

The supported values follow the project requirements.
"""

from dataclasses import dataclass
from ipaddress import ip_address
from typing import Self

from rtp_live_streaming.common.payload_types import Codec

# Project-supported configuration values.
SUPPORTED_VIDEO_BITRATES = (500, 1000, 2000, 4000)
SUPPORTED_VIDEO_RESOLUTIONS = (
    (640, 480),
    (1280, 720),
)
SUPPORTED_VIDEO_FPS = (15, 30)

SUPPORTED_AUDIO_BITRATES = (32, 64, 128)

SUPPORTED_JITTER_BUFFER_MS = (20, 50, 100, 200, 500)

DEFAULT_VIDEO_RTP_PORT = 5004
DEFAULT_VIDEO_RTCP_PORT = 5005
DEFAULT_AUDIO_RTP_PORT = 5006
DEFAULT_AUDIO_RTCP_PORT = 5007
DEFAULT_CONTROL_PORT = 6000

DEFAULT_VIDEO_BITRATE_KBPS = 1000
DEFAULT_AUDIO_BITRATE_KBPS = 64
DEFAULT_JITTER_BUFFER_MS = 50


class ConfigError(ValueError):
    """Raised when an application configuration is invalid."""


def _normalize_codec(codec: Codec | str) -> Codec:
    """Convert a codec name or enum to Codec."""
    try:
        return Codec(codec.lower())
    except (ValueError, AttributeError):
        raise ConfigError(f"Unsupported codec: {codec!r}") from None


def _validate_port(port: int, name: str) -> None:
    """Validate a TCP or UDP port number."""
    if not isinstance(port, int) or isinstance(port, bool):
        raise ConfigError(f"{name} must be an integer")

    if not 1 <= port <= 65535:
        raise ConfigError(f"{name} must be between 1 and 65535")


def _validate_bitrate(
    bitrate: int,
    supported_values: tuple[int, ...],
    name: str,
) -> None:
    """Validate a bitrate against supported project values."""
    if not isinstance(bitrate, int) or isinstance(bitrate, bool):
        raise ConfigError(f"{name} must be an integer")

    if bitrate not in supported_values:
        raise ConfigError(
            f"Unsupported {name}: {bitrate}. Supported values: {supported_values}"
        )


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Network configuration for RTP, RTCP, and control traffic."""

    bind_host: str = "0.0.0.0"
    destination_host: str = "127.0.0.1"

    video_rtp_port: int = DEFAULT_VIDEO_RTP_PORT
    video_rtcp_port: int = DEFAULT_VIDEO_RTCP_PORT

    audio_rtp_port: int = DEFAULT_AUDIO_RTP_PORT
    audio_rtcp_port: int = DEFAULT_AUDIO_RTCP_PORT

    control_port: int = DEFAULT_CONTROL_PORT

    socket_buffer_size: int = 1_048_576

    def __post_init__(self) -> None:
        """Validate network settings."""
        try:
            ip_address(self.bind_host)
        except ValueError:
            raise ConfigError(f"Invalid bind IP address: {self.bind_host!r}") from None

        try:
            ip_address(self.destination_host)
        except ValueError:
            raise ConfigError(
                f"Invalid destination IP address: {self.destination_host!r}"
            ) from None

        ports = {
            "video_rtp_port": self.video_rtp_port,
            "video_rtcp_port": self.video_rtcp_port,
            "audio_rtp_port": self.audio_rtp_port,
            "audio_rtcp_port": self.audio_rtcp_port,
            "control_port": self.control_port,
        }

        for name, port in ports.items():
            _validate_port(port, name)

        if len(set(ports.values())) != len(ports):
            raise ConfigError("Network ports must be unique")

        if (
            not isinstance(self.socket_buffer_size, int)
            or isinstance(self.socket_buffer_size, bool)
            or self.socket_buffer_size <= 0
        ):
            raise ConfigError("socket_buffer_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class VideoConfig:
    """Video encoding and streaming configuration."""

    codec: Codec | str = Codec.H264
    bitrate_kbps: int = DEFAULT_VIDEO_BITRATE_KBPS

    width: int = 1280
    height: int = 720

    fps: int = 30

    def __post_init__(self) -> None:
        """Normalize and validate video settings."""
        object.__setattr__(self, "codec", _normalize_codec(self.codec))

        if self.codec not in (Codec.H264, Codec.VP8):
            raise ConfigError(f"{self.codec.value} is not a supported video codec")

        _validate_bitrate(
            self.bitrate_kbps,
            SUPPORTED_VIDEO_BITRATES,
            "video bitrate",
        )

        if (self.width, self.height) not in SUPPORTED_VIDEO_RESOLUTIONS:
            raise ConfigError(
                f"Unsupported video resolution: "
                f"{self.width}x{self.height}. "
                f"Supported resolutions: {SUPPORTED_VIDEO_RESOLUTIONS}"
            )

        if (
            not isinstance(self.fps, int)
            or isinstance(self.fps, bool)
            or self.fps not in SUPPORTED_VIDEO_FPS
        ):
            raise ConfigError(
                f"Unsupported FPS: {self.fps}. Supported values: {SUPPORTED_VIDEO_FPS}"
            )


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio encoding and streaming configuration."""

    codec: Codec | str = Codec.OPUS
    bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS
    channels: int = 1

    def __post_init__(self) -> None:
        """Normalize and validate audio settings."""
        object.__setattr__(self, "codec", _normalize_codec(self.codec))

        if self.codec not in (Codec.OPUS, Codec.PCMU):
            raise ConfigError(f"{self.codec.value} is not a supported audio codec")

        if (
            not isinstance(self.channels, int)
            or isinstance(self.channels, bool)
            or self.channels not in (1, 2)
        ):
            raise ConfigError("Audio channels must be 1 or 2")

        if self.codec == Codec.OPUS:
            _validate_bitrate(
                self.bitrate_kbps,
                SUPPORTED_AUDIO_BITRATES,
                "audio bitrate",
            )

        elif self.codec == Codec.PCMU:
            # G.711 PCMU uses 8-bit samples at 8 kHz:
            # 8000 * 8 = 64000 bits per second.
            if self.bitrate_kbps != 64:
                raise ConfigError("PCMU has a fixed bitrate of 64 kbps")

            if self.channels != 1:
                raise ConfigError("PCMU static payload type 0 uses mono audio")

    @property
    def sample_rate_hz(self) -> int:
        """Return the required audio sample rate."""
        if self.codec == Codec.OPUS:
            return 48_000

        return 8_000


@dataclass(frozen=True, slots=True)
class ReceiverConfig:
    """Receiver-side buffering and playback configuration."""

    jitter_buffer_ms: int = DEFAULT_JITTER_BUFFER_MS

    def __post_init__(self) -> None:
        """Validate receiver settings."""
        if (
            not isinstance(self.jitter_buffer_ms, int)
            or isinstance(self.jitter_buffer_ms, bool)
            or self.jitter_buffer_ms not in SUPPORTED_JITTER_BUFFER_MS
        ):
            raise ConfigError(
                f"Unsupported jitter buffer size: "
                f"{self.jitter_buffer_ms}. "
                f"Supported values: {SUPPORTED_JITTER_BUFFER_MS}"
            )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""

    network: NetworkConfig = NetworkConfig()
    video: VideoConfig = VideoConfig()
    audio: AudioConfig = AudioConfig()
    receiver: ReceiverConfig = ReceiverConfig()

    @classmethod
    def default(cls) -> Self:
        """Create an application configuration with default values."""
        return cls()
