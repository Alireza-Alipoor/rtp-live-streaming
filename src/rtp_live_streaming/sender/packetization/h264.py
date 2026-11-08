"""H.264 RTP packetization according to RFC 6184.

Implements packetization mode 1:
    - Single NAL unit packets (types 1-23).
    - Fragmentation Units (FU-A, type 28).

STAP-A aggregation and interleaved packetization are not implemented.

The returned payloads can be passed directly to RTPPacket.payload.
The marker flag indicates the final RTP packet of an access unit.
"""

from collections.abc import Iterable
from dataclasses import dataclass

NAL_HEADER_SIZE = 1
FU_A_HEADER_SIZE = 2

NAL_TYPE_MIN = 1
NAL_TYPE_MAX = 23
NAL_TYPE_FU_A = 28

FORBIDDEN_ZERO_BIT = 0x80
NAL_REF_IDC_MASK = 0x60
NAL_TYPE_MASK = 0x1F

FU_START_BIT = 0x80
FU_END_BIT = 0x40
FU_RESERVED_BIT = 0x20

START_CODE_3 = b"\x00\x00\x01"
START_CODE_4 = b"\x00\x00\x00\x01"


class H264PacketizationError(ValueError):
    """Raised when an H.264 NAL unit cannot be packetized."""


@dataclass(frozen=True, slots=True)
class H264RTPPayload:
    """An H.264 RTP payload and its associated marker bit."""

    payload: bytes
    marker: bool = False


def _validate_nal_unit(nal_unit: bytes) -> int:
    """Validate a single NAL unit and return its NAL unit type."""
    if not isinstance(nal_unit, bytes):
        raise H264PacketizationError("NAL unit must be bytes")

    if not nal_unit:
        raise H264PacketizationError("NAL unit cannot be empty")

    header = nal_unit[0]

    if header & FORBIDDEN_ZERO_BIT:
        raise H264PacketizationError("NAL unit forbidden_zero_bit must be zero")

    nal_type = header & NAL_TYPE_MASK

    if not NAL_TYPE_MIN <= nal_type <= NAL_TYPE_MAX:
        raise H264PacketizationError(
            f"Unsupported NAL unit type: {nal_type}. "
            "Expected a single NAL unit type between 1 and 23."
        )

    return nal_type


def _find_start_codes(data: bytes) -> list[tuple[int, int]]:
    """Find Annex-B start codes.

    Returns:
        A list of (position, start_code_length) tuples.

    Four-byte start codes are checked before three-byte start codes
    to avoid interpreting their suffix as a separate start code.
    """
    start_codes: list[tuple[int, int]] = []
    index = 0

    while index < len(data):
        if data.startswith(START_CODE_4, index):
            start_codes.append((index, 4))
            index += 4

        elif data.startswith(START_CODE_3, index):
            start_codes.append((index, 3))
            index += 3

        else:
            index += 1

    return start_codes


def extract_nal_units(data: bytes) -> tuple[bytes, ...]:
    """Extract NAL units from Annex-B data or a single raw NAL unit.

    If no Annex-B start code is found, the input is treated as one
    raw NAL unit.

    Leading bytes before the first Annex-B start code are ignored.
    Empty NAL units are rejected.
    """
    if not isinstance(data, bytes):
        raise H264PacketizationError("Input data must be bytes")

    if not data:
        raise H264PacketizationError("Input data cannot be empty")

    start_codes = _find_start_codes(data)

    if not start_codes:
        _validate_nal_unit(data)
        return (data,)

    nal_units: list[bytes] = []

    for index, (start_position, start_code_length) in enumerate(start_codes):
        nal_start = start_position + start_code_length

        if index + 1 < len(start_codes):
            nal_end = start_codes[index + 1][0]
        else:
            nal_end = len(data)

        nal_unit = data[nal_start:nal_end]

        # Annex-B permits trailing_zero_8bits between NAL units
        # and after the final NAL unit.
        nal_unit = nal_unit.rstrip(b"\x00")

        if not nal_unit:
            raise H264PacketizationError("Annex-B stream contains an empty NAL unit")

        _validate_nal_unit(nal_unit)
        nal_units.append(nal_unit)

    if not nal_units:
        raise H264PacketizationError("No NAL units found")

    return tuple(nal_units)


def _packetize_nal_unit(
    nal_unit: bytes,
    max_payload_size: int,
) -> list[H264RTPPayload]:
    """Packetize one NAL unit without assigning the final marker."""
    nal_type = _validate_nal_unit(nal_unit)

    if (
        not isinstance(max_payload_size, int)
        or isinstance(max_payload_size, bool)
        or max_payload_size <= 0
    ):
        raise H264PacketizationError("max_payload_size must be a positive integer")

    # A NAL unit fits into one RTP packet.
    if len(nal_unit) <= max_payload_size:
        return [
            H264RTPPayload(
                payload=nal_unit,
                marker=False,
            )
        ]

    # FU-A requires two bytes of overhead:
    # one FU indicator and one FU header.
    if max_payload_size <= FU_A_HEADER_SIZE:
        raise H264PacketizationError(
            "max_payload_size must be greater than 2 for FU-A fragmentation"
        )

    original_header = nal_unit[0]

    # Preserve F and NRI bits from the original NAL header.
    fu_indicator = (
        original_header & (FORBIDDEN_ZERO_BIT | NAL_REF_IDC_MASK)
    ) | NAL_TYPE_FU_A

    # The original NAL header is replaced by the FU headers.
    nal_data = nal_unit[NAL_HEADER_SIZE:]

    fragment_size = max_payload_size - FU_A_HEADER_SIZE

    fragments: list[H264RTPPayload] = []

    offset = 0
    while offset < len(nal_data):
        fragment = nal_data[offset : offset + fragment_size]

        is_start = offset == 0
        is_end = offset + len(fragment) == len(nal_data)

        fu_header = nal_type

        if is_start:
            fu_header |= FU_START_BIT

        if is_end:
            fu_header |= FU_END_BIT

        payload = bytes((fu_indicator, fu_header)) + fragment

        fragments.append(
            H264RTPPayload(
                payload=payload,
                marker=False,
            )
        )

        offset += len(fragment)

    return fragments


def packetize_nal_unit(
    nal_unit: bytes,
    max_payload_size: int,
    marker: bool = True,
) -> tuple[H264RTPPayload, ...]:
    """Packetize one NAL unit into RTP payloads.

    Args:
        nal_unit: One complete H.264 NAL unit, including its header.
        max_payload_size: Maximum RTP payload size in bytes.
        marker: Whether the final packet should have marker=True.

    Returns:
        A tuple of H264RTPPayload objects.
    """
    if not isinstance(marker, bool):
        raise H264PacketizationError("marker must be a boolean")

    packets = _packetize_nal_unit(
        nal_unit=nal_unit,
        max_payload_size=max_payload_size,
    )

    packets[-1] = H264RTPPayload(
        payload=packets[-1].payload,
        marker=marker,
    )

    return tuple(packets)


def packetize_access_unit(
    access_unit: bytes | Iterable[bytes],
    max_payload_size: int,
) -> tuple[H264RTPPayload, ...]:
    """Packetize all NAL units belonging to one access unit.

    Args:
        access_unit:
            Either Annex-B encoded bytes or an iterable of raw NAL units.
        max_payload_size:
            Maximum RTP payload size in bytes.

    Returns:
        A tuple of RTP payloads. Only the final packet has marker=True.
    """
    if isinstance(access_unit, bytes):
        nal_units = extract_nal_units(access_unit)
    else:
        try:
            nal_units = tuple(access_unit)
        except TypeError:
            raise H264PacketizationError(
                "access_unit must be bytes or an iterable of NAL units"
            ) from None

    if not nal_units:
        raise H264PacketizationError("Access unit must contain at least one NAL unit")

    packets: list[H264RTPPayload] = []

    for nal_unit in nal_units:
        packets.extend(
            _packetize_nal_unit(
                nal_unit=nal_unit,
                max_payload_size=max_payload_size,
            )
        )

    # RFC 6184: marker bit is set on the final packet of the access unit.
    last_packet = packets[-1]

    packets[-1] = H264RTPPayload(
        payload=last_packet.payload,
        marker=True,
    )

    return tuple(packets)
