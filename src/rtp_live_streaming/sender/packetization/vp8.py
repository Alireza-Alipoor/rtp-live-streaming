"""VP8 RTP packetization according to RFC 7741.

This module packetizes one encoded VP8 frame into one or more RTP payloads.
It implements the mandatory one-byte VP8 payload descriptor and deliberately
omits optional descriptor extensions (PictureID, TL0PICIDX, T/K fields).

The first RTP payload of a frame has S=1 and PartID=0. Continuation payloads
have S=0. The RTP marker bit is set only on the final payload of the frame.
The returned payload bytes can be passed directly to ``RTPPacket.payload``.
"""

from dataclasses import dataclass

VP8_PAYLOAD_DESCRIPTOR_SIZE = 1

# RFC 7741, section 4.2: mandatory descriptor fields.
X_BIT = 0x80  # Extension bit
R_BIT = 0x40  # Reserved bit
N_BIT = 0x20  # Non-reference frame bit
S_BIT = 0x10  # Start of VP8 partition
PARTITION_ID_MASK = 0x0F


class VP8PacketizationError(ValueError):
    """Raised when an encoded VP8 frame cannot be packetized."""


@dataclass(frozen=True, slots=True)
class VP8RTPPayload:
    """One VP8 RTP payload and its associated RTP marker bit."""

    payload: bytes
    marker: bool = False


def _validate_frame(frame: bytes | bytearray | memoryview) -> bytes:
    """Validate and normalize one encoded VP8 frame."""
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise VP8PacketizationError("VP8 frame must be bytes-like")

    data = bytes(frame)
    if not data:
        raise VP8PacketizationError("VP8 frame cannot be empty")

    return data


def _validate_max_payload_size(max_payload_size: int) -> None:
    if not isinstance(max_payload_size, int) or isinstance(max_payload_size, bool):
        raise VP8PacketizationError("max_payload_size must be an integer")
    if max_payload_size <= VP8_PAYLOAD_DESCRIPTOR_SIZE:
        raise VP8PacketizationError(
            "max_payload_size must be greater than the one-byte VP8 payload descriptor"
        )


def packetize_frame(
    frame: bytes | bytearray | memoryview,
    max_payload_size: int,
) -> tuple[VP8RTPPayload, ...]:
    """Split one encoded VP8 frame into RTP payloads.

    Args:
        frame: A complete encoded VP8 frame from the encoder (not a raw
            ``av.VideoFrame`` and not an RTP packet).
        max_payload_size: Maximum size of the complete RTP payload, including
            the VP8 payload descriptor. This should normally be
            ``max_datagram_size - 12`` for an RTP packet without extensions.

    Returns:
        A tuple of :class:`VP8RTPPayload` objects. The first payload starts
        partition 0 (S=1); subsequent payloads are continuations (S=0). Only
        the final payload has ``marker=True``.

    Notes:
        VP8 uses a 90 kHz RTP clock. Timestamp assignment is handled by the
        RTP sender, not by this packetizer.
    """
    data = _validate_frame(frame)
    _validate_max_payload_size(max_payload_size)

    fragment_data_size = max_payload_size - VP8_PAYLOAD_DESCRIPTOR_SIZE
    packets: list[VP8RTPPayload] = []

    for offset in range(0, len(data), fragment_data_size):
        fragment = data[offset : offset + fragment_data_size]
        is_start = offset == 0

        # X=0, R=0, N=0, PartID=0. Set S only on the first packet.
        descriptor = S_BIT if is_start else 0x00
        packets.append(
            VP8RTPPayload(
                payload=bytes((descriptor,)) + fragment,
                marker=False,
            )
        )

    # RFC 7741: marker identifies the final RTP packet of the VP8 frame.
    last = packets[-1]
    packets[-1] = VP8RTPPayload(payload=last.payload, marker=True)
    return tuple(packets)


def packetize(
    frame: bytes | bytearray | memoryview,
    max_payload_size: int,
) -> tuple[VP8RTPPayload, ...]:
    """Alias for :func:`packetize_frame` for a codec-neutral call site."""
    return packetize_frame(frame, max_payload_size)


__all__ = [
    "VP8PacketizationError",
    "VP8RTPPayload",
    "packetize_frame",
    "packetize",
]
