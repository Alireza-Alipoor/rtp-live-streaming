"""A small, bounded RTP packet reordering and playout buffer.

The buffer accepts parsed :class:`RTPPacket` instances and releases them in
sequence-number order after a configurable playout delay. It deliberately does
not depacketize media: the receiver should pass each packet returned by
:meth:`pop_ready` to the selected codec depacketizer.

All time values are based on ``time.monotonic()``. RTP sequence numbers are
compared modulo 2**16, so the normal wrap from 65535 to 0 is handled.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.rtp_packet import RTPPacket


_SEQUENCE_MODULUS = 1 << 16
_SEQUENCE_MASK = _SEQUENCE_MODULUS - 1
_SEQUENCE_HALF_RANGE = _SEQUENCE_MODULUS // 2


@dataclass(frozen=True, slots=True)
class JitterBufferStats:
    """Counters and current occupancy for a :class:`JitterBuffer`."""

    received_packets: int
    released_packets: int
    reordered_packets: int
    duplicate_packets: int
    late_packets: int
    lost_packets: int
    overflow_packets: int
    buffered_packets: int


class JitterBuffer:
    """Reorder RTP packets and release them after a fixed playout delay.

    Parameters
    ----------
    buffer_ms:
        Time to wait after a packet arrives before making it eligible for
        release. The project's suggested values are 20, 50, 100, 200 and
        500 milliseconds. The class accepts any non-negative value so tests
        and experiments can use finer-grained settings.
    max_packets:
        Maximum number of packets held at once. New packets are rejected
        when the buffer is full; already-buffered packets are never silently
        overwritten.

    Notes
    -----
    ``push`` only inserts a packet. Call ``pop_ready`` periodically from the
    receiver's consumer loop to obtain packets whose playout deadline has
    passed. If the next expected sequence number is missing, the buffer waits
    until the earliest queued packet beyond that gap reaches its deadline,
    then records the missing sequence numbers as lost and continues.

    One instance is intended for one RTP source/SSRC. Use ``reset`` when
    switching to a different source or starting a new RTP sequence space.
    """

    def __init__(self, buffer_ms: float = 50, max_packets: int = 512) -> None:
        if buffer_ms < 0:
            raise ValueError("buffer_ms must be non-negative")
        if max_packets < 1:
            raise ValueError("max_packets must be at least 1")

        self.buffer_ms = float(buffer_ms)
        self.max_packets = int(max_packets)
        self._delay_seconds = self.buffer_ms / 1000.0
        self._packets: dict[int, tuple[float, RTPPacket]] = {}
        self._expected_sequence: int | None = None
        self._highest_sequence: int | None = None
        self._playout_started = False
        self._lock = threading.Lock()

        self._received_packets = 0
        self._released_packets = 0
        self._reordered_packets = 0
        self._duplicate_packets = 0
        self._late_packets = 0
        self._lost_packets = 0
        self._overflow_packets = 0

    @property
    def occupancy(self) -> int:
        """Number of packets currently waiting in the buffer."""
        with self._lock:
            return len(self._packets)

    @property
    def expected_sequence(self) -> int | None:
        """Sequence number that the buffer currently expects next."""
        with self._lock:
            return self._expected_sequence

    @property
    def stats(self) -> JitterBufferStats:
        """Return a consistent snapshot of buffer counters."""
        with self._lock:
            return JitterBufferStats(
                received_packets=self._received_packets,
                released_packets=self._released_packets,
                reordered_packets=self._reordered_packets,
                duplicate_packets=self._duplicate_packets,
                late_packets=self._late_packets,
                lost_packets=self._lost_packets,
                overflow_packets=self._overflow_packets,
                buffered_packets=len(self._packets),
            )

    def push(self, packet: RTPPacket, arrival_time: float | None = None) -> bool:
        """Insert a packet, returning ``False`` if it is rejected.

        Before the first release, an earlier sequence number may move the
        initial playout point backwards. This lets the buffer tolerate packets
        arriving out of order during startup. Once playback has begun, packets
        behind ``expected_sequence`` are late and are discarded.
        """
        sequence = self._validate_sequence_number(packet.sequence_number)
        arrived_at = time.monotonic() if arrival_time is None else float(arrival_time)

        with self._lock:
            self._received_packets += 1

            if sequence in self._packets:
                self._duplicate_packets += 1
                return False
            if len(self._packets) >= self.max_packets:
                self._overflow_packets += 1
                return False

            if self._expected_sequence is None:
                self._expected_sequence = sequence
                self._highest_sequence = sequence
            else:
                assert self._highest_sequence is not None
                from_expected = self._forward_distance(
                    self._expected_sequence, sequence
                )
                from_highest = self._forward_distance(self._highest_sequence, sequence)

                # A distance of exactly half the sequence space has no
                # unambiguous forward/backward interpretation (RFC-style
                # serial number arithmetic). Treat it as late/invalid.
                if (
                    from_expected == _SEQUENCE_HALF_RANGE
                    or from_highest == _SEQUENCE_HALF_RANGE
                ):
                    self._late_packets += 1
                    return False

                if not self._playout_started and from_expected > _SEQUENCE_HALF_RANGE:
                    # The packet precedes our provisional startup sequence.
                    # Re-anchor startup so the earliest observed packet leads.
                    self._expected_sequence = sequence
                    self._reordered_packets += 1
                elif self._playout_started and from_expected > _SEQUENCE_HALF_RANGE:
                    self._late_packets += 1
                    return False
                elif from_highest > _SEQUENCE_HALF_RANGE:
                    # It arrived behind a sequence number already observed.
                    self._reordered_packets += 1
                elif 0 < from_highest < _SEQUENCE_HALF_RANGE:
                    self._highest_sequence = sequence

            self._packets[sequence] = (arrived_at, packet)
            return True

    def pop_ready(self, now: float | None = None) -> list[RTPPacket]:
        """Release packets in sequence order once their playout delay expires.

        Each packet becomes eligible at ``arrival_time + buffer_ms``. If the
        expected sequence number is missing, the nearest later packet's
        deadline determines when the gap is declared lost.
        """
        current_time = time.monotonic() if now is None else float(now)
        ready: list[RTPPacket] = []

        with self._lock:
            while self._expected_sequence is not None and self._packets:
                expected = self._expected_sequence
                expected_entry = self._packets.get(expected)

                if expected_entry is not None:
                    arrival, packet = expected_entry
                    if current_time < arrival + self._delay_seconds:
                        break
                    del self._packets[expected]
                    ready.append(packet)
                    self._released_packets += 1
                    self._playout_started = True
                    self._expected_sequence = (expected + 1) & _SEQUENCE_MASK
                    continue

                next_sequence, next_arrival = self._nearest_ahead_packet()
                if (
                    next_sequence is None
                    or current_time < next_arrival + self._delay_seconds
                ):
                    break

                missing = self._forward_distance(expected, next_sequence)
                self._lost_packets += missing
                self._expected_sequence = next_sequence
                self._playout_started = True

        return ready

    def flush(self) -> list[RTPPacket]:
        """Release every buffered packet in sequence order immediately.

        Any sequence-number gaps encountered during the flush are counted as
        packet loss. This is useful when shutting down or changing streams; it
        should not normally be used during uninterrupted playback.
        """
        ready: list[RTPPacket] = []
        with self._lock:
            while self._expected_sequence is not None and self._packets:
                expected_entry = self._packets.pop(self._expected_sequence, None)
                if expected_entry is not None:
                    ready.append(expected_entry[1])
                    self._released_packets += 1
                    self._playout_started = True
                    self._expected_sequence = (
                        self._expected_sequence + 1
                    ) & _SEQUENCE_MASK
                    continue

                next_sequence, _ = self._nearest_ahead_packet()
                if next_sequence is None:
                    break
                self._lost_packets += self._forward_distance(
                    self._expected_sequence, next_sequence
                )
                self._expected_sequence = next_sequence
                self._playout_started = True

        return ready

    def reset(self, *, reset_stats: bool = False) -> None:
        """Discard buffered packets and start a new sequence-number space.

        Set ``reset_stats=True`` to clear counters as well. The default keeps
        cumulative counters useful across stream discontinuities.
        """
        with self._lock:
            self._packets.clear()
            self._expected_sequence = None
            self._highest_sequence = None
            self._playout_started = False
            if reset_stats:
                self._received_packets = 0
                self._released_packets = 0
                self._reordered_packets = 0
                self._duplicate_packets = 0
                self._late_packets = 0
                self._lost_packets = 0
                self._overflow_packets = 0

    def _nearest_ahead_packet(self) -> tuple[int | None, float]:
        """Find the queued packet nearest to the current expected sequence."""
        assert self._expected_sequence is not None
        candidates = (
            (
                self._forward_distance(self._expected_sequence, sequence),
                sequence,
                arrival,
            )
            for sequence, (arrival, _) in self._packets.items()
        )
        ahead = (item for item in candidates if 0 < item[0] < _SEQUENCE_HALF_RANGE)
        nearest = min(ahead, default=None, key=lambda item: item[0])
        if nearest is None:
            return None, 0.0
        return nearest[1], nearest[2]

    @staticmethod
    def _forward_distance(start: int, end: int) -> int:
        return (end - start) & _SEQUENCE_MASK

    @staticmethod
    def _validate_sequence_number(sequence: int) -> int:
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise TypeError("RTP sequence_number must be an integer")
        if not 0 <= sequence <= _SEQUENCE_MASK:
            raise ValueError("RTP sequence_number must be between 0 and 65535")
        return sequence


__all__ = ["JitterBuffer", "JitterBufferStats"]
