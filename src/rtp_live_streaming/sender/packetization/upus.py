"""Opus RTP packetization according to RFC 7587.

An encoded Opus packet is already a self-delimiting Opus payload (including
its TOC byte and frame framing). Unlike H.264 and VP8, Opus does not need an
additional codec-specific RTP payload descriptor or fragmentation header.
This packetizer therefore places exactly one complete encoded Opus packet in
one RTP payload and never splits it across RTP packets.

The caller controls the RTP marker bit. It is normally set on the first
packet of an audio talkspurt, not on every audio packet. RTP timestamps use
the 48 kHz Opus clock and are assigned by the sender.
"""

from __future__ import annotations

from dataclasses import dataclass


class OpusPacketizationError(ValueError):
    """Raised when an encoded Opus packet cannot be packetized."""


@dataclass(frozen=True, slots=True)
class OpusRTPPayload:
    """One complete Opus RTP payload and its RTP marker bit."""

    payload: bytes
    marker: bool = False


def _validate_encoded_packet(
    encoded_packet: bytes | bytearray | memoryview,
) -> bytes:
    """Validate and normalize one encoded Opus packet.

    The first byte is the Opus TOC byte. All bit patterns in that byte are
    defined by the Opus RTP payload format; validating the remaining Opus
    frame structure is the encoder/decoder's responsibility.
    """
    if not isinstance(encoded_packet, (bytes, bytearray, memoryview)):
        raise OpusPacketizationError("encoded_packet must be bytes-like")

    packet = bytes(encoded_packet)
    if not packet:
        raise OpusPacketizationError("encoded Opus packet cannot be empty")

    return packet


def _validate_max_payload_size(max_payload_size: int) -> None:
    if not isinstance(max_payload_size, int) or isinstance(max_payload_size, bool):
        raise OpusPacketizationError("max_payload_size must be an integer")
    if max_payload_size <= 0:
        raise OpusPacketizationError("max_payload_size must be a positive integer")


def packetize_packet(
    encoded_packet: bytes | bytearray | memoryview,
    max_payload_size: int,
    *,
    marker: bool = False,
) -> tuple[OpusRTPPayload, ...]:
    """Wrap one complete encoded Opus packet as one RTP payload.

    Args:
        encoded_packet: One complete packet produced by an Opus encoder. It
            must include the Opus TOC byte and Opus packet framing.
        max_payload_size: Maximum permitted RTP payload size, excluding the
            12-byte RTP base header. The packet is rejected rather than
            fragmented if it exceeds this limit.
        marker: RTP marker bit. Set this for the first packet of a talkspurt
            when the application detects a talkspurt boundary.

    Returns:
        A one-element tuple, matching the output convention of the project's
        H.264 and VP8 packetizers.

    Raises:
        OpusPacketizationError: If the packet, size limit, or marker is invalid.

    Notes:
        Opus uses a 48,000 Hz RTP clock for every supported sampling rate.
        Timestamp calculation is intentionally left to the RTP sender.
    """
    packet = _validate_encoded_packet(encoded_packet)
    _validate_max_payload_size(max_payload_size)

    if not isinstance(marker, bool):
        raise OpusPacketizationError("marker must be a boolean")

    if len(packet) > max_payload_size:
        raise OpusPacketizationError(
            f"Opus packet is {len(packet)} bytes; RTP payload limit is "
            f"{max_payload_size}. Opus packets must not be fragmented."
        )

    return (OpusRTPPayload(payload=packet, marker=marker),)


def packetize(
    encoded_packet: bytes | bytearray | memoryview,
    max_payload_size: int,
    *,
    marker: bool = False,
) -> tuple[OpusRTPPayload, ...]:
    """Codec-neutral alias for :func:`packetize_packet`."""
    return packetize_packet(encoded_packet, max_payload_size, marker=marker)


__all__ = [
    "OpusPacketizationError",
    "OpusRTPPayload",
    "packetize_packet",
    "packetize",
]
