"""RTCP packet structures, serialization, and parsing.

Supported packet types:
    SR  (200): Sender Report
    RR  (201): Receiver Report

Unknown RTCP packet types are preserved as RawRTCPPacket instances.

RTCP packets use network byte order (big-endian).
"""

import struct
from dataclasses import dataclass

RTCP_VERSION = 2
RTCP_HEADER_SIZE = 4

RTCP_PT_SR = 200
RTCP_PT_RR = 201

UINT8_MAX = (1 << 8) - 1
UINT16_MAX = (1 << 16) - 1
UINT24_MAX = (1 << 24) - 1
UINT32_MAX = (1 << 32) - 1

INT24_MIN = -(1 << 23)
INT24_MAX = (1 << 23) - 1

MAX_REPORT_BLOCKS = 31
REPORT_BLOCK_SIZE = 24

NTP_UNIX_EPOCH_OFFSET = 2_208_988_800
NTP_FRACTION_SCALE = 1 << 32


class RTCPError(ValueError):
    """Raised when an RTCP packet is invalid or malformed."""


def _validate_uint(value: int, bits: int, name: str) -> None:
    """Validate an unsigned integer with the specified bit width."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise RTCPError(f"{name} must be an integer")

    if not 0 <= value < (1 << bits):
        raise RTCPError(f"{name} must be between 0 and {(1 << bits) - 1}")


def _validate_int24(value: int, name: str) -> None:
    """Validate a signed 24-bit integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise RTCPError(f"{name} must be an integer")

    if not INT24_MIN <= value <= INT24_MAX:
        raise RTCPError(f"{name} must be between {INT24_MIN} and {INT24_MAX}")


def _validate_padding_size(padding_size: int) -> None:
    """Validate RTCP packet padding.

    All supported RTCP packet bodies are word-aligned, so padding
    must also preserve 32-bit alignment.
    """
    if not isinstance(padding_size, int) or isinstance(padding_size, bool):
        raise RTCPError("padding_size must be an integer")

    if padding_size < 0 or padding_size > UINT8_MAX:
        raise RTCPError("padding_size must be between 0 and 255")

    if padding_size and padding_size % 4 != 0:
        raise RTCPError("padding_size must be a multiple of 4")


def _validate_report_blocks(
    report_blocks: tuple["ReportBlock", ...],
) -> None:
    """Validate an RTCP report block collection."""
    if len(report_blocks) > MAX_REPORT_BLOCKS:
        raise RTCPError(
            f"An RTCP packet cannot contain more than {MAX_REPORT_BLOCKS} report blocks"
        )

    if not all(isinstance(block, ReportBlock) for block in report_blocks):
        raise RTCPError("All report blocks must be ReportBlock instances")


@dataclass(frozen=True, slots=True)
class ReportBlock:
    """Reception statistics for one RTP synchronization source.

    Fields follow RFC 3550 section 6.4.1.
    """

    ssrc: int
    fraction_lost: int
    cumulative_lost: int
    extended_highest_sequence_number: int
    interarrival_jitter: int
    last_sender_report: int
    delay_since_last_sender_report: int

    def __post_init__(self) -> None:
        _validate_uint(self.ssrc, 32, "ssrc")
        _validate_uint(self.fraction_lost, 8, "fraction_lost")
        _validate_int24(self.cumulative_lost, "cumulative_lost")
        _validate_uint(
            self.extended_highest_sequence_number,
            32,
            "extended_highest_sequence_number",
        )
        _validate_uint(self.interarrival_jitter, 32, "interarrival_jitter")
        _validate_uint(self.last_sender_report, 32, "last_sender_report")
        _validate_uint(
            self.delay_since_last_sender_report,
            32,
            "delay_since_last_sender_report",
        )

    def to_bytes(self) -> bytes:
        """Serialize the report block into exactly 24 bytes."""
        cumulative_lost = self.cumulative_lost & UINT24_MAX

        return (
            struct.pack("!IB", self.ssrc, self.fraction_lost)
            + cumulative_lost.to_bytes(3, byteorder="big")
            + struct.pack(
                "!IIII",
                self.extended_highest_sequence_number,
                self.interarrival_jitter,
                self.last_sender_report,
                self.delay_since_last_sender_report,
            )
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ReportBlock":
        """Parse a 24-byte RTCP report block."""
        if len(data) != REPORT_BLOCK_SIZE:
            raise RTCPError(f"Report block must be {REPORT_BLOCK_SIZE} bytes")

        ssrc, fraction_lost = struct.unpack_from("!IB", data, 0)

        cumulative_lost = int.from_bytes(data[5:8], byteorder="big")

        # Convert unsigned 24-bit representation to signed integer.
        if cumulative_lost & (1 << 23):
            cumulative_lost -= 1 << 24

        (
            extended_highest_sequence_number,
            interarrival_jitter,
            last_sender_report,
            delay_since_last_sender_report,
        ) = struct.unpack_from("!IIII", data, 8)

        return cls(
            ssrc=ssrc,
            fraction_lost=fraction_lost,
            cumulative_lost=cumulative_lost,
            extended_highest_sequence_number=extended_highest_sequence_number,
            interarrival_jitter=interarrival_jitter,
            last_sender_report=last_sender_report,
            delay_since_last_sender_report=delay_since_last_sender_report,
        )


@dataclass(frozen=True, slots=True)
class SenderReport:
    """RTCP Sender Report (PT=200).

    The sender report contains sender information followed by zero
    or more reception report blocks.
    """

    ssrc: int
    ntp_seconds: int
    ntp_fraction: int
    rtp_timestamp: int
    sender_packet_count: int
    sender_octet_count: int
    report_blocks: tuple[ReportBlock, ...] = ()
    padding_size: int = 0

    packet_type: int = RTCP_PT_SR

    def __post_init__(self) -> None:
        _validate_uint(self.ssrc, 32, "ssrc")
        _validate_uint(self.ntp_seconds, 32, "ntp_seconds")
        _validate_uint(self.ntp_fraction, 32, "ntp_fraction")
        _validate_uint(self.rtp_timestamp, 32, "rtp_timestamp")
        _validate_uint(self.sender_packet_count, 32, "sender_packet_count")
        _validate_uint(self.sender_octet_count, 32, "sender_octet_count")

        _validate_report_blocks(self.report_blocks)
        _validate_padding_size(self.padding_size)

    def to_bytes(self) -> bytes:
        """Serialize the Sender Report."""
        body = struct.pack(
            "!IIIIII",
            self.ssrc,
            self.ntp_seconds,
            self.ntp_fraction,
            self.rtp_timestamp,
            self.sender_packet_count,
            self.sender_octet_count,
        )

        body += b"".join(block.to_bytes() for block in self.report_blocks)

        return _serialize_packet(
            packet_type=RTCP_PT_SR,
            count=len(self.report_blocks),
            body=body,
            padding_size=self.padding_size,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SenderReport":
        """Parse one complete Sender Report packet."""
        parsed = _parse_single_packet(data)

        if parsed[0] != RTCP_PT_SR:
            raise RTCPError("Packet is not a Sender Report")

        count, body, padding_size = parsed[1], parsed[2], parsed[3]

        expected_size = 24 + count * REPORT_BLOCK_SIZE
        if len(body) != expected_size:
            raise RTCPError("Invalid Sender Report body length")

        (
            ssrc,
            ntp_seconds,
            ntp_fraction,
            rtp_timestamp,
            sender_packet_count,
            sender_octet_count,
        ) = struct.unpack_from("!IIIIII", body, 0)

        report_blocks = _parse_report_blocks(body[24:], count)

        return cls(
            ssrc=ssrc,
            ntp_seconds=ntp_seconds,
            ntp_fraction=ntp_fraction,
            rtp_timestamp=rtp_timestamp,
            sender_packet_count=sender_packet_count,
            sender_octet_count=sender_octet_count,
            report_blocks=report_blocks,
            padding_size=padding_size,
        )


@dataclass(frozen=True, slots=True)
class ReceiverReport:
    """RTCP Receiver Report (PT=201)."""

    ssrc: int
    report_blocks: tuple[ReportBlock, ...] = ()
    padding_size: int = 0

    packet_type: int = RTCP_PT_RR

    def __post_init__(self) -> None:
        _validate_uint(self.ssrc, 32, "ssrc")
        _validate_report_blocks(self.report_blocks)
        _validate_padding_size(self.padding_size)

    def to_bytes(self) -> bytes:
        """Serialize the Receiver Report."""
        body = struct.pack("!I", self.ssrc)
        body += b"".join(block.to_bytes() for block in self.report_blocks)

        return _serialize_packet(
            packet_type=RTCP_PT_RR,
            count=len(self.report_blocks),
            body=body,
            padding_size=self.padding_size,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ReceiverReport":
        """Parse one complete Receiver Report packet."""
        parsed = _parse_single_packet(data)

        if parsed[0] != RTCP_PT_RR:
            raise RTCPError("Packet is not a Receiver Report")

        count, body, padding_size = parsed[1], parsed[2], parsed[3]

        expected_size = 4 + count * REPORT_BLOCK_SIZE
        if len(body) != expected_size:
            raise RTCPError("Invalid Receiver Report body length")

        (ssrc,) = struct.unpack_from("!I", body, 0)
        report_blocks = _parse_report_blocks(body[4:], count)

        return cls(
            ssrc=ssrc,
            report_blocks=report_blocks,
            padding_size=padding_size,
        )


@dataclass(frozen=True, slots=True)
class RawRTCPPacket:
    """An RTCP packet type not interpreted by this module.

    Preserves the packet type, count/subtype field, body, and padding.
    Useful for compound RTCP packets containing SDES, BYE, or APP.
    """

    packet_type: int
    count: int
    body: bytes
    padding_size: int = 0

    def __post_init__(self) -> None:
        _validate_uint(self.packet_type, 8, "packet_type")
        _validate_uint(self.count, 5, "count")

        if self.packet_type < 192 or self.packet_type > 223:
            raise RTCPError("Invalid RTCP packet type")

        if not isinstance(self.body, bytes):
            raise RTCPError("body must be bytes")

        if len(self.body) % 4 != 0:
            raise RTCPError("RTCP packet body must be 32-bit aligned")

        _validate_padding_size(self.padding_size)

    def to_bytes(self) -> bytes:
        """Serialize the raw RTCP packet."""
        return _serialize_packet(
            packet_type=self.packet_type,
            count=self.count,
            body=self.body,
            padding_size=self.padding_size,
        )


RTCPPacket = SenderReport | ReceiverReport | RawRTCPPacket


def _serialize_packet(
    packet_type: int,
    count: int,
    body: bytes,
    padding_size: int = 0,
) -> bytes:
    """Serialize a generic RTCP packet header and body."""
    _validate_uint(packet_type, 8, "packet_type")
    _validate_uint(count, 5, "count")
    _validate_padding_size(padding_size)

    if len(body) % 4 != 0:
        raise RTCPError("RTCP packet body must be 32-bit aligned")

    padding = b""
    if padding_size:
        padding = b"\x00" * (padding_size - 1) + bytes([padding_size])

    packet_size = RTCP_HEADER_SIZE + len(body) + len(padding)
    length_words = packet_size // 4 - 1

    _validate_uint(length_words, 16, "RTCP length")

    first_byte = (RTCP_VERSION << 6) | count

    if padding_size:
        first_byte |= 0x20

    header = struct.pack(
        "!BBH",
        first_byte,
        packet_type,
        length_words,
    )

    return header + body + padding


def _parse_single_packet(
    data: bytes,
) -> tuple[int, int, bytes, int]:
    """Parse one complete RTCP packet.

    Returns:
        (packet_type, count, body_without_padding, padding_size)
    """
    if not isinstance(data, bytes):
        raise RTCPError("RTCP data must be bytes")

    if len(data) < RTCP_HEADER_SIZE:
        raise RTCPError("RTCP packet is shorter than its header")

    if len(data) % 4 != 0:
        raise RTCPError("RTCP packet size must be 32-bit aligned")

    first_byte, packet_type, length_words = struct.unpack_from("!BBH", data, 0)

    version = first_byte >> 6
    padding_flag = bool(first_byte & 0x20)
    count = first_byte & 0x1F

    if version != RTCP_VERSION:
        raise RTCPError(f"Unsupported RTCP version: {version}")

    if not 192 <= packet_type <= 223:
        raise RTCPError(f"Invalid RTCP packet type: {packet_type}")

    packet_size = (length_words + 1) * 4

    if packet_size != len(data):
        raise RTCPError(
            f"RTCP length field specifies {packet_size} bytes, "
            f"but received {len(data)} bytes"
        )

    body = data[RTCP_HEADER_SIZE:]
    padding_size = 0

    if padding_flag:
        if not body:
            raise RTCPError("RTCP padding flag set without padding")

        padding_size = body[-1]

        if padding_size == 0:
            raise RTCPError("Invalid RTCP padding size: 0")

        if padding_size > len(body):
            raise RTCPError("RTCP padding exceeds packet body")

        body = body[:-padding_size]

        if len(body) % 4 != 0:
            raise RTCPError("RTCP body is not 32-bit aligned")

    return packet_type, count, body, padding_size


def _parse_report_blocks(
    data: bytes,
    count: int,
) -> tuple[ReportBlock, ...]:
    """Parse the report blocks following an SR or RR body."""
    expected_size = count * REPORT_BLOCK_SIZE

    if len(data) != expected_size:
        raise RTCPError("Invalid RTCP report block data length")

    return tuple(
        ReportBlock.from_bytes(data[index : index + REPORT_BLOCK_SIZE])
        for index in range(0, len(data), REPORT_BLOCK_SIZE)
    )


def parse_rtcp_packet(data: bytes) -> RTCPPacket:
    """Parse one RTCP packet.

    Use parse_compound_rtcp() when the datagram contains multiple
    RTCP packets.
    """
    packet_type, _, _, _ = _parse_single_packet(data)

    if packet_type == RTCP_PT_SR:
        return SenderReport.from_bytes(data)

    if packet_type == RTCP_PT_RR:
        return ReceiverReport.from_bytes(data)

    parsed_type, count, body, padding_size = _parse_single_packet(data)

    return RawRTCPPacket(
        packet_type=parsed_type,
        count=count,
        body=body,
        padding_size=padding_size,
    )


def parse_compound_rtcp(data: bytes) -> tuple[RTCPPacket, ...]:
    """Parse a compound RTCP datagram into individual packets.

    The returned packets preserve their original order.
    """
    if not isinstance(data, bytes):
        raise RTCPError("RTCP data must be bytes")

    if not data:
        raise RTCPError("RTCP datagram cannot be empty")

    if len(data) % 4 != 0:
        raise RTCPError("RTCP datagram size must be 32-bit aligned")

    packets: list[RTCPPacket] = []
    offset = 0

    while offset < len(data):
        remaining = data[offset:]

        if len(remaining) < RTCP_HEADER_SIZE:
            raise RTCPError("Incomplete RTCP packet header")

        _, _, length_words = struct.unpack_from("!BBH", remaining, 0)
        packet_size = (length_words + 1) * 4

        if packet_size < RTCP_HEADER_SIZE:
            raise RTCPError("Invalid RTCP packet length")

        if offset + packet_size > len(data):
            raise RTCPError("Truncated RTCP compound packet")

        packet_data = data[offset : offset + packet_size]
        packets.append(parse_rtcp_packet(packet_data))

        offset += packet_size

    return tuple(packets)


def serialize_compound_rtcp(
    packets: tuple[RTCPPacket, ...] | list[RTCPPacket],
) -> bytes:
    """Serialize multiple RTCP packets into one compound datagram."""
    if not packets:
        raise RTCPError("An RTCP compound packet cannot be empty")

    serialized: list[bytes] = []

    for packet in packets:
        if not isinstance(
            packet,
            (SenderReport, ReceiverReport, RawRTCPPacket),
        ):
            raise RTCPError(f"Unsupported RTCP packet object: {type(packet).__name__}")

        serialized.append(packet.to_bytes())

    return b"".join(serialized)


def unix_to_ntp(unix_seconds: float) -> tuple[int, int]:
    """Convert Unix time to an NTP timestamp pair.

    Returns:
        (NTP seconds, NTP fractional seconds)

    The two fields are unsigned 32-bit integers.
    """
    if isinstance(unix_seconds, bool) or not isinstance(unix_seconds, (int, float)):
        raise RTCPError("unix_seconds must be a number")

    if unix_seconds < 0:
        raise RTCPError("unix_seconds cannot be negative")

    ntp_time = unix_seconds + NTP_UNIX_EPOCH_OFFSET

    ntp_seconds = int(ntp_time)
    ntp_fraction = round((ntp_time - ntp_seconds) * NTP_FRACTION_SCALE)

    if ntp_fraction >= NTP_FRACTION_SCALE:
        ntp_seconds += 1
        ntp_fraction = 0

    ntp_seconds %= 1 << 32

    return ntp_seconds, ntp_fraction


def ntp_to_unix(ntp_seconds: int, ntp_fraction: int) -> float:
    """Convert an NTP timestamp pair to Unix time."""
    _validate_uint(ntp_seconds, 32, "ntp_seconds")
    _validate_uint(ntp_fraction, 32, "ntp_fraction")

    return ntp_seconds - NTP_UNIX_EPOCH_OFFSET + ntp_fraction / NTP_FRACTION_SCALE
