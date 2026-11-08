"""Clock utilities for RTP timestamps and media timing.

RTP timestamps represent media sampling time, not Unix time.

Clock rates:
    H.264: 90,000 Hz
    VP8:   90,000 Hz
    Opus:  48,000 Hz
    PCMU:   8,000 Hz
"""

import time
from enum import IntEnum

from rtp_live_streaming.common.payload_types import Codec


class ClockRate(IntEnum):
    """RTP clock rates for supported codecs."""

    VIDEO = 90_000
    OPUS = 48_000
    PCMU = 8_000


CODEC_CLOCK_RATES: dict[Codec, ClockRate] = {
    Codec.H264: ClockRate.VIDEO,
    Codec.VP8: ClockRate.VIDEO,
    Codec.OPUS: ClockRate.OPUS,
    Codec.PCMU: ClockRate.PCMU,
}

NANOSECONDS_PER_SECOND = 1_000_000_000
UINT32_MODULUS = 1 << 32
UINT32_MAX = UINT32_MODULUS - 1


class ClockError(ValueError):
    """Raised when clock configuration or timestamp input is invalid."""


def get_clock_rate(codec: Codec | str) -> int:
    """Return the RTP clock rate for a supported codec."""
    try:
        normalized_codec = Codec(codec.lower())
    except (ValueError, AttributeError):
        raise ClockError(f"Unsupported codec: {codec!r}") from None

    return int(CODEC_CLOCK_RATES[normalized_codec])


def monotonic_ns() -> int:
    """Return monotonic time in nanoseconds."""
    return time.monotonic_ns()


def monotonic_seconds() -> float:
    """Return monotonic time in seconds."""
    return time.monotonic()


def elapsed_seconds(start_ns: int, end_ns: int | None = None) -> float:
    """Calculate elapsed time between two monotonic timestamps."""
    if not isinstance(start_ns, int) or isinstance(start_ns, bool):
        raise ClockError("start_ns must be an integer")

    if end_ns is None:
        end_ns = monotonic_ns()

    if not isinstance(end_ns, int) or isinstance(end_ns, bool):
        raise ClockError("end_ns must be an integer")

    if end_ns < start_ns:
        raise ClockError("end_ns cannot be earlier than start_ns")

    return (end_ns - start_ns) / NANOSECONDS_PER_SECOND


def seconds_to_rtp_ticks(seconds: float, clock_rate: int) -> int:
    """Convert a duration in seconds to RTP clock ticks."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ClockError("seconds must be a number")

    if seconds < 0:
        raise ClockError("seconds cannot be negative")

    if not isinstance(clock_rate, int) or isinstance(clock_rate, bool):
        raise ClockError("clock_rate must be an integer")

    if clock_rate <= 0:
        raise ClockError("clock_rate must be positive")

    return round(seconds * clock_rate)


def rtp_ticks_to_seconds(ticks: int, clock_rate: int) -> float:
    """Convert RTP clock ticks to seconds."""
    if not isinstance(ticks, int) or isinstance(ticks, bool):
        raise ClockError("ticks must be an integer")

    if ticks < 0:
        raise ClockError("ticks cannot be negative")

    if not isinstance(clock_rate, int) or isinstance(clock_rate, bool):
        raise ClockError("clock_rate must be an integer")

    if clock_rate <= 0:
        raise ClockError("clock_rate must be positive")

    return ticks / clock_rate


def timestamp_from_elapsed(
    elapsed_ns: int,
    clock_rate: int,
    initial_timestamp: int = 0,
) -> int:
    """Generate a 32-bit RTP timestamp from elapsed monotonic time.

    RTP timestamps wrap around after 2**32 ticks.
    """
    if not isinstance(elapsed_ns, int) or isinstance(elapsed_ns, bool):
        raise ClockError("elapsed_ns must be an integer")

    if elapsed_ns < 0:
        raise ClockError("elapsed_ns cannot be negative")

    if not isinstance(initial_timestamp, int) or isinstance(initial_timestamp, bool):
        raise ClockError("initial_timestamp must be an integer")

    if not 0 <= initial_timestamp <= UINT32_MAX:
        raise ClockError("initial_timestamp must be between 0 and 2**32 - 1")

    ticks = round(elapsed_ns * clock_rate / NANOSECONDS_PER_SECOND)

    return (initial_timestamp + ticks) % UINT32_MODULUS


def timestamp_difference(newer: int, older: int) -> int:
    """Calculate the forward difference between two RTP timestamps.

    Handles 32-bit RTP timestamp wraparound.
    """
    _validate_rtp_timestamp(newer)
    _validate_rtp_timestamp(older)

    return (newer - older) % UINT32_MODULUS


def _validate_rtp_timestamp(timestamp: int) -> None:
    """Validate a 32-bit unsigned RTP timestamp."""
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        raise ClockError("RTP timestamp must be an integer")

    if not 0 <= timestamp <= UINT32_MAX:
        raise ClockError("RTP timestamp must be between 0 and 2**32 - 1")


class RTPClock:
    """Generate RTP timestamps relative to a monotonic clock.

    Each RTP stream should have its own clock instance and initial
    timestamp. The initial timestamp can be randomized by the caller.
    """

    def __init__(
        self,
        clock_rate: int,
        initial_timestamp: int = 0,
    ) -> None:
        if not isinstance(clock_rate, int) or isinstance(clock_rate, bool):
            raise ClockError("clock_rate must be an integer")

        if clock_rate <= 0:
            raise ClockError("clock_rate must be positive")

        _validate_rtp_timestamp(initial_timestamp)

        self.clock_rate = clock_rate
        self.initial_timestamp = initial_timestamp
        self.start_time_ns = monotonic_ns()

    @classmethod
    def for_codec(
        cls,
        codec: Codec | str,
        initial_timestamp: int = 0,
    ) -> "RTPClock":
        """Create an RTP clock using a supported codec's clock rate."""
        return cls(
            clock_rate=get_clock_rate(codec),
            initial_timestamp=initial_timestamp,
        )

    def timestamp(self) -> int:
        """Return the current 32-bit RTP timestamp."""
        elapsed_ns = monotonic_ns() - self.start_time_ns

        return timestamp_from_elapsed(
            elapsed_ns=elapsed_ns,
            clock_rate=self.clock_rate,
            initial_timestamp=self.initial_timestamp,
        )

    def ticks_from_seconds(self, seconds: float) -> int:
        """Convert seconds to ticks using this clock's rate."""
        return seconds_to_rtp_ticks(seconds, self.clock_rate)

    def seconds_from_ticks(self, ticks: int) -> float:
        """Convert ticks to seconds using this clock's rate."""
        return rtp_ticks_to_seconds(ticks, self.clock_rate)
