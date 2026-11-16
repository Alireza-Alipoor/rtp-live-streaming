"""VP8 RTP depacketization according to RFC 7741.

The depacketizer removes the VP8 payload descriptor and reconstructs one
encoded VP8 frame from its RTP payload fragments. The RTP marker bit identifies
the final packet of a frame.

Packets must be supplied in sequence-number order (normally by the receiver's
jitter buffer). A sequence-number gap should cause the caller to call
:meth:`VP8Depacketizer.reset` before feeding subsequent packets; this module
does not have access to RTP sequence numbers and therefore cannot detect loss
by itself.

Returned data is a raw VP8 elementary-stream frame, suitable for a VP8
 decoder. It is not wrapped in an IVF container.

Reference: RFC 7741, "RTP Payload Format for VP8 Video".
"""

from __future__ import annotations

from dataclasses import dataclass, field

type BytesLike = bytes | bytearray | memoryview

# Mandatory payload descriptor (RFC 7741, section 4.2).
X_BIT = 0x80
R_BIT = 0x40
N_BIT = 0x20
S_BIT = 0x10
PARTITION_ID_MASK = 0x0F
MAX_PARTITION_ID = 8

# Optional extension flags.
I_BIT = 0x80
L_BIT = 0x40
T_BIT = 0x20
K_BIT = 0x10

DEFAULT_MAX_FRAME_SIZE = 16 * 1024 * 1024


class VP8DepacketizationError(ValueError):
    """Raised when a VP8 RTP payload is malformed or cannot be reassembled."""


@dataclass(frozen=True, slots=True)
class VP8PayloadDescriptor:
    """Parsed fields from a VP8 RTP payload descriptor.

    ``picture_id`` is normalized to an integer for both its 7-bit and 15-bit
    wire encodings. Fields not carried in the descriptor are ``None``.
    """

    non_reference: bool
    start_of_partition: bool
    partition_id: int
    picture_id: int | None = None
    tl0_pic_idx: int | None = None
    temporal_id: int | None = None
    layer_sync: bool | None = None
    key_idx: int | None = None
    descriptor_size: int = 1


def _as_bytes(payload: BytesLike) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise VP8DepacketizationError("VP8 RTP payload must be bytes-like")
    return bytes(payload)


def parse_payload_descriptor(
    payload: BytesLike,
) -> tuple[VP8PayloadDescriptor, bytes]:
    """Parse a VP8 payload descriptor and return ``(descriptor, frame_data)``.

    Supports the mandatory descriptor and all optional fields defined by
    RFC 7741: PictureID (7- or 15-bit), TL0PICIDX, TID/Y, and KEYIDX.
    """
    data = _as_bytes(payload)
    if not data:
        raise VP8DepacketizationError("VP8 RTP payload cannot be empty")

    first = data[0]
    if first & R_BIT:
        raise VP8DepacketizationError("VP8 descriptor reserved R bit must be zero")

    has_extension = bool(first & X_BIT)
    non_reference = bool(first & N_BIT)
    start = bool(first & S_BIT)
    partition_id = first & PARTITION_ID_MASK
    if partition_id > MAX_PARTITION_ID:
        raise VP8DepacketizationError(
            f"Invalid VP8 partition ID {partition_id}; expected 0-{MAX_PARTITION_ID}"
        )

    offset = 1
    picture_id: int | None = None
    tl0_pic_idx: int | None = None
    temporal_id: int | None = None
    layer_sync: bool | None = None
    key_idx: int | None = None

    if has_extension:
        if offset >= len(data):
            raise VP8DepacketizationError("VP8 extension flags byte is missing")
        extension = data[offset]
        offset += 1

        if extension & 0x0F:
            raise VP8DepacketizationError("VP8 extension reserved bits must be zero")

        if extension & I_BIT:
            if offset >= len(data):
                raise VP8DepacketizationError("VP8 PictureID is missing")
            picture_id_first = data[offset]
            offset += 1
            if picture_id_first & 0x80:
                if offset >= len(data):
                    raise VP8DepacketizationError("15-bit VP8 PictureID is truncated")
                picture_id = ((picture_id_first & 0x7F) << 8) | data[offset]
                offset += 1
            else:
                picture_id = picture_id_first & 0x7F

        if extension & L_BIT:
            if offset >= len(data):
                raise VP8DepacketizationError("VP8 TL0PICIDX is missing")
            tl0_pic_idx = data[offset]
            offset += 1

        if extension & (T_BIT | K_BIT):
            if offset >= len(data):
                raise VP8DepacketizationError("VP8 TID/KEYIDX byte is missing")
            temporal_key = data[offset]
            offset += 1
            if extension & T_BIT:
                temporal_id = (temporal_key >> 6) & 0x03
                layer_sync = bool(temporal_key & 0x20)
            if extension & K_BIT:
                key_idx = temporal_key & 0x1F

    if offset >= len(data):
        raise VP8DepacketizationError("VP8 RTP payload contains no encoded frame data")

    descriptor = VP8PayloadDescriptor(
        non_reference=non_reference,
        start_of_partition=start,
        partition_id=partition_id,
        picture_id=picture_id,
        tl0_pic_idx=tl0_pic_idx,
        temporal_id=temporal_id,
        layer_sync=layer_sync,
        key_idx=key_idx,
        descriptor_size=offset,
    )
    return descriptor, data[offset:]


@dataclass(slots=True)
class VP8Depacketizer:
    """Reassemble RTP payloads into complete VP8 frames.

    Create one instance per RTP source/SSRC. Call :meth:`push` for every
    packet released by the jitter buffer. It returns ``None`` until the RTP
    marker bit completes a frame, then returns the reconstructed frame bytes.

    ``max_frame_size`` bounds memory use for malformed or untrusted traffic.
    """

    max_frame_size: int = DEFAULT_MAX_FRAME_SIZE
    _current_frame: bytearray | None = field(default=None, init=False, repr=False)
    _frame_non_reference: bool | None = field(default=None, init=False, repr=False)
    _frame_picture_id: int | None = field(default=None, init=False, repr=False)
    _last_partition_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_frame_size, int)
            or isinstance(self.max_frame_size, bool)
            or self.max_frame_size <= 0
        ):
            raise ValueError("max_frame_size must be a positive integer")

    @property
    def assembling(self) -> bool:
        """Whether an incomplete frame is currently buffered."""
        return self._current_frame is not None

    @property
    def buffered_bytes(self) -> int:
        """Number of encoded frame bytes currently buffered."""
        return 0 if self._current_frame is None else len(self._current_frame)

    def push(self, payload: BytesLike, marker: bool = False) -> bytes | None:
        """Consume one RTP payload and optionally return a complete frame.

        Args:
            payload: RTP payload including the VP8 payload descriptor.
            marker: RTP M bit. It must be true on the final packet of a frame.

        Returns:
            Complete raw VP8 frame bytes, or ``None`` while assembling.

        Raises:
            VP8DepacketizationError: If the descriptor, partition sequence, or
                frame size is invalid. State is reset after an error.
        """
        if not isinstance(marker, bool):
            raise VP8DepacketizationError("marker must be a boolean")

        try:
            descriptor, fragment = parse_payload_descriptor(payload)

            # A new frame always begins at the start of partition zero. If one
            # arrives while another frame is incomplete, the earlier frame was
            # truncated (e.g. its final RTP packet was lost). Resynchronize on
            # the new frame instead of allowing the two frames to be combined.
            starts_new_frame = (
                descriptor.start_of_partition and descriptor.partition_id == 0
            )
            if self._current_frame is None or starts_new_frame:
                self._begin_frame(descriptor)
            else:
                self._validate_continuation(descriptor)

            assert self._current_frame is not None
            if len(self._current_frame) + len(fragment) > self.max_frame_size:
                raise VP8DepacketizationError(
                    f"VP8 frame exceeds configured limit of {self.max_frame_size} bytes"
                )
            self._current_frame.extend(fragment)

            if not marker:
                return None

            completed = bytes(self._current_frame)
            self.reset()
            return completed

        except VP8DepacketizationError:
            self.reset()
            raise

    def _begin_frame(self, descriptor: VP8PayloadDescriptor) -> None:
        if not descriptor.start_of_partition or descriptor.partition_id != 0:
            raise VP8DepacketizationError(
                "VP8 frame must begin with S=1 and partition ID 0"
            )
        self._current_frame = bytearray()
        self._frame_non_reference = descriptor.non_reference
        self._frame_picture_id = descriptor.picture_id
        self._last_partition_id = 0

    def _validate_continuation(self, descriptor: VP8PayloadDescriptor) -> None:
        if descriptor.non_reference != self._frame_non_reference:
            raise VP8DepacketizationError(
                "VP8 N bit changed while reassembling a frame"
            )

        if descriptor.start_of_partition:
            expected_partition = self._last_partition_id + 1
            if expected_partition > MAX_PARTITION_ID:
                raise VP8DepacketizationError(
                    "VP8 frame contains more than 9 partitions"
                )
            if descriptor.partition_id != expected_partition:
                raise VP8DepacketizationError(
                    "VP8 partition start is out of order: "
                    f"expected {expected_partition}, got {descriptor.partition_id}"
                )
            self._last_partition_id = descriptor.partition_id
        elif descriptor.partition_id != self._last_partition_id:
            raise VP8DepacketizationError(
                "VP8 continuation packet does not match the active partition"
            )

        # A PictureID may be omitted from individual packets. When both the
        # first packet and a later packet carry one, however, they must match.
        if (
            self._frame_picture_id is not None
            and descriptor.picture_id is not None
            and descriptor.picture_id != self._frame_picture_id
        ):
            raise VP8DepacketizationError(
                "VP8 PictureID changed while reassembling a frame"
            )

    def reset(self) -> None:
        """Discard a partial frame (e.g. after detected RTP packet loss)."""
        self._current_frame = None
        self._frame_non_reference = None
        self._frame_picture_id = None
        self._last_partition_id = 0


# More explicit alias for callers that use RTP terminology.
VP8RTPDepacketizer = VP8Depacketizer


__all__ = [
    "DEFAULT_MAX_FRAME_SIZE",
    "MAX_PARTITION_ID",
    "VP8DepacketizationError",
    "VP8PayloadDescriptor",
    "VP8Depacketizer",
    "VP8RTPDepacketizer",
    "parse_payload_descriptor",
]
