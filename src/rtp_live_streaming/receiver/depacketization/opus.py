"""Opus RTP depacketization according to RFC 7587.

The Opus RTP payload format carries one complete Opus packet per RTP packet.
It adds no codec-specific payload header and does not define fragmentation
across RTP packets. Depacketization is consequently a validated pass-through:
the input RTP payload is returned unchanged as one encoded Opus packet.

The RTP marker bit marks the beginning of an audio talkspurt. It is metadata,
not part of the Opus bitstream, and is exposed as ``last_marker`` by the
stateful interface. Each instance should be associated with one RTP stream.

RTP sequence-number reordering and loss detection belong in the jitter buffer,
which must deliver packets in order before calling this module.

Reference: RFC 7587, "RTP Payload Format for the Opus Speech and Audio Codec".
"""

from __future__ import annotations

from dataclasses import dataclass

BytesLike = bytes | bytearray | memoryview


class OpusDepacketizationError(ValueError):
    """Raised when an RTP payload cannot be accepted as an Opus packet."""


def _validate_payload(payload: BytesLike) -> bytes:
    """Validate and normalize one RTP Opus payload."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise OpusDepacketizationError("Opus RTP payload must be bytes-like")

    packet = bytes(payload)
    if not packet:
        raise OpusDepacketizationError("Opus RTP payload cannot be empty")

    # The first byte is the Opus TOC (Table Of Contents) byte. All 256
    # bit patterns are defined by the Opus packet format, so deeper codec
    # framing validation is intentionally left to the Opus decoder.
    return packet


def depacketize_packet(payload: BytesLike) -> bytes:
    """Return the encoded Opus packet carried by one RTP payload.

    RFC 7587 does not add an RTP payload descriptor around Opus data. The
    complete RTP payload, including its TOC byte and frame framing, is the
    encoded Opus packet itself.

    Args:
        payload: RTP payload bytes (not the RTP header).

    Returns:
        The encoded Opus packet as immutable ``bytes``.

    Raises:
        OpusDepacketizationError: If the payload is not bytes-like or empty.
    """
    return _validate_payload(payload)


@dataclass(slots=True)
class OpusDepacketizer:
    """Depacketize one Opus RTP stream.

    Call :meth:`push` once for each RTP packet released by the jitter buffer.
    Unlike video depacketizers, this class returns a packet immediately:
    every RTP Opus payload is already a complete encoded Opus packet.

    The marker bit is retained in :attr:`last_marker` because it signals the
    start of a talkspurt. It is not included in the returned audio bytes.
    """

    _last_marker: bool = False

    @property
    def last_marker(self) -> bool:
        """Marker bit of the most recently accepted RTP packet."""
        return self._last_marker

    def push(self, payload: BytesLike, marker: bool = False) -> bytes:
        """Accept one RTP payload and return its complete Opus packet.

        Args:
            payload: RTP payload containing exactly one Opus packet.
            marker: RTP M bit; true indicates the start of a talkspurt.

        Returns:
            Complete encoded Opus packet bytes.

        Raises:
            OpusDepacketizationError: If the payload or marker is invalid.
        """
        if not isinstance(marker, bool):
            raise OpusDepacketizationError("marker must be a boolean")

        try:
            packet = _validate_payload(payload)
        except OpusDepacketizationError:
            self.reset()
            raise

        self._last_marker = marker
        return packet

    def reset(self) -> None:
        """Clear packet metadata after a stream reset or codec change."""
        self._last_marker = False


# Explicit RTP-named alias for callers that use codec-specific class names.
OpusRTPDepacketizer = OpusDepacketizer


__all__ = [
    "OpusDepacketizationError",
    "OpusDepacketizer",
    "OpusRTPDepacketizer",
    "depacketize_packet",
]
