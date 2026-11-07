
"""
RTP packet representation and serialization.

Implements the RTP version 2 packet format defined in RFC 3550.

This module handles RTP headers and payloads. Codec-specific RTP
packetization belongs in the sender/packetization package.
"""

from dataclasses import dataclass
import struct


RTP_VERSION = 2
RTP_HEADER_SIZE = 12

UINT16_MAX = 0xFFFF
UINT32_MAX = 0xFFFFFFFF


class RTPPacketError(ValueError):
    """Raised when an RTP packet is invalid."""


@dataclass(slots=True)
class RTPPacket:
    """Represents an RTP packet."""

    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes

    marker: bool = False
    csrcs: tuple[int, ...] = ()

    extension_profile: int | None = None
    extension_data: bytes = b""

    padding_size: int = 0

    def __post_init__(self) -> None:
        """Validate packet fields."""

        self.payload = bytes(self.payload)
        self.extension_data = bytes(self.extension_data)
        self.csrcs = tuple(self.csrcs)

        if not 0 <= self.payload_type <= 127:
            raise RTPPacketError("Payload type must be between 0 and 127.")

        if not 0 <= self.sequence_number <= UINT16_MAX:
            raise RTPPacketError("Sequence number must be 16-bit.")

        if not 0 <= self.timestamp <= UINT32_MAX:
            raise RTPPacketError("Timestamp must be 32-bit.")

        if not 0 <= self.ssrc <= UINT32_MAX:
            raise RTPPacketError("SSRC must be 32-bit.")

        if len(self.csrcs) > 15:
            raise RTPPacketError("Maximum number of CSRC entries is 15.")

        for csrc in self.csrcs:
            if not 0 <= csrc <= UINT32_MAX:
                raise RTPPacketError("Each CSRC must be 32-bit.")

        if self.extension_profile is None:
            if self.extension_data:
                raise RTPPacketError(
                    "Extension data requires an extension profile."
                )
        else:
            if not 0 <= self.extension_profile <= UINT16_MAX:
                raise RTPPacketError("Extension profile must be 16-bit.")

            if len(self.extension_data) % 4 != 0:
                raise RTPPacketError(
                    "Extension data length must be a multiple of 4 bytes."
                )

        if not 0 <= self.padding_size <= 255:
            raise RTPPacketError("Padding size must be between 0 and 255.")

    def to_bytes(self) -> bytes:
        """Serialize the RTP packet into bytes."""

        has_extension = self.extension_profile is not None
        has_padding = self.padding_size > 0

        # First byte: V(2) | P(1) | X(1) | CC(4)
        first_byte = (
            (RTP_VERSION << 6)
            | (int(has_padding) << 5)
            | (int(has_extension) << 4)
            | len(self.csrcs)
        )

        # Second byte: M(1) | PT(7)
        second_byte = (
            (int(self.marker) << 7)
            | self.payload_type
        )

        header = struct.pack(
            "!BBHII",
            first_byte,
            second_byte,
            self.sequence_number,
            self.timestamp,
            self.ssrc,
        )

        # Append CSRC identifiers.
        if self.csrcs:
            header += struct.pack(
                f"!{len(self.csrcs)}I",
                *self.csrcs,
            )

        # Append optional header extension.
        if has_extension:
            extension_length = len(self.extension_data) // 4

            header += struct.pack(
                "!HH",
                self.extension_profile,
                extension_length,
            )

            header += self.extension_data

        packet = header + self.payload

        # Append RTP padding when requested.
        if has_padding:
            padding = (
                b"\x00" * (self.padding_size - 1)
                + bytes([self.padding_size])
            )
            packet += padding

        return packet

    @classmethod
    def from_bytes(cls, data: bytes) -> "RTPPacket":
        """Parse an RTP packet from bytes."""

        if len(data) < RTP_HEADER_SIZE:
            raise RTPPacketError("RTP packet is smaller than 12 bytes.")

        first_byte, second_byte, sequence, timestamp, ssrc = (
            struct.unpack_from("!BBHII", data)
        )

        version = (first_byte >> 6) & 0b11
        padding_flag = (first_byte >> 5) & 0b1
        extension_flag = (first_byte >> 4) & 0b1
        csrc_count = first_byte & 0b1111

        if version != RTP_VERSION:
            raise RTPPacketError(
                f"Unsupported RTP version: {version}"
            )

        marker = bool((second_byte >> 7) & 0b1)
        payload_type = second_byte & 0b01111111

        offset = RTP_HEADER_SIZE

        # Parse CSRC identifiers.
        csrc_size = csrc_count * 4

        if len(data) < offset + csrc_size:
            raise RTPPacketError("Incomplete CSRC list.")

        csrcs = ()

        if csrc_count:
            csrcs = struct.unpack_from(
                f"!{csrc_count}I",
                data,
                offset,
            )

            offset += csrc_size

        # Parse optional header extension.
        extension_profile = None
        extension_data = b""

        if extension_flag:
            if len(data) < offset + 4:
                raise RTPPacketError("Incomplete RTP extension header.")

            extension_profile, extension_length = struct.unpack_from(
                "!HH",
                data,
                offset,
            )

            offset += 4

            extension_size = extension_length * 4

            if len(data) < offset + extension_size:
                raise RTPPacketError("Incomplete RTP extension data.")

            extension_data = data[offset:offset + extension_size]
            offset += extension_size

        # Parse optional padding.
        padding_size = 0
        payload_end = len(data)

        if padding_flag:
            padding_size = data[-1]

            if padding_size == 0:
                raise RTPPacketError("Invalid RTP padding size.")

            if padding_size > len(data) - offset:
                raise RTPPacketError("Invalid RTP padding length.")

            payload_end -= padding_size

        if payload_end < offset:
            raise RTPPacketError("Invalid RTP payload boundaries.")

        payload = data[offset:payload_end]

        return cls(
            payload_type=payload_type,
            sequence_number=sequence,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=payload,
            marker=marker,
            csrcs=csrcs,
            extension_profile=extension_profile,
            extension_data=extension_data,
            padding_size=padding_size,
        )
