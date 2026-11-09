"""UDP transport for RTP packets.

The sender deliberately accepts *encoded and packetized* payloads. Codec
encoding and codec-specific RTP payload formatting belong to the encoder and
packetization modules. This keeps the RTP header/sequence-number logic
independent from PyAV and from individual codecs.
"""

from __future__ import annotations

import secrets
import socket
import threading
from collections.abc import Iterable

from rtp_live_streaming.common.payload_types import (
    Codec,
    get_payload_type,
    validate_payload_type,
)
from rtp_live_streaming.common.rtp_packet import RTPPacket


class RTPSenderError(RuntimeError):
    """Raised when an RTP sender operation cannot be completed."""


class RTPSender:
    """Serialize RTP packets and transmit them over a UDP socket.

    Parameters
    ----------
    destination_host, destination_port:
        Remote UDP endpoint.
    codec:
        Codec associated with the outgoing RTP payload type.
    ssrc, initial_sequence_number:
        Optional RTP identifiers. Random values are used by default.
    max_datagram_size:
        Maximum UDP payload size. Keep this below the path MTU to avoid IP
        fragmentation; 1200 bytes is a conservative default.
    sock:
        Optional pre-created UDP socket, primarily useful for tests.

    ``timestamp`` passed to :meth:`send_payload` is a media-clock timestamp,
    not wall-clock time. The caller should use the appropriate RTP clock rate
    (90 kHz for video, 48 kHz for Opus, 8 kHz for PCMU).
    """

    def __init__(
        self,
        destination_host: str,
        destination_port: int,
        codec: Codec | str,
        *,
        ssrc: int | None = None,
        initial_sequence_number: int | None = None,
        max_datagram_size: int = 1200,
        sock: socket.socket | None = None,
        payload_type: int | None = None,
    ) -> None:
        if not destination_host:
            raise ValueError("destination_host cannot be empty")
        if not isinstance(destination_port, int) or isinstance(destination_port, bool):
            raise TypeError("destination_port must be an integer")
        if not 1 <= destination_port <= 65535:
            raise ValueError("destination_port must be between 1 and 65535")
        if not isinstance(max_datagram_size, int) or isinstance(
            max_datagram_size, bool
        ):
            raise TypeError("max_datagram_size must be an integer")
        if not 12 < max_datagram_size <= 65507:
            raise ValueError("max_datagram_size must be between 13 and 65507")

        self.destination = (destination_host, destination_port)
        self._codec = Codec(codec.lower())
        self._payload_type = (
            get_payload_type(self._codec)
            if payload_type is None
            else validate_payload_type(payload_type)
        )
        self.ssrc = (
            secrets.randbits(32)
            if ssrc is None
            else self._validate_uint(ssrc, 32, "ssrc")
        )
        self._sequence_number = (
            secrets.randbits(16)
            if initial_sequence_number is None
            else self._validate_uint(
                initial_sequence_number, 16, "initial_sequence_number"
            )
        )
        self.max_datagram_size = max_datagram_size
        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._owns_socket = sock is None
        self._closed = False
        self._lock = threading.Lock()

    @staticmethod
    def _validate_uint(value: int, bits: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if not 0 <= value < (1 << bits):
            raise ValueError(f"{name} must fit in {bits} bits")
        return value

    @property
    def codec(self) -> Codec:
        with self._lock:
            return self._codec

    @property
    def payload_type(self) -> int:
        with self._lock:
            return self._payload_type

    @property
    def next_sequence_number(self) -> int:
        with self._lock:
            return self._sequence_number

    @property
    def closed(self) -> bool:
        return self._closed

    def set_codec(self, codec: Codec | str, *, payload_type: int | None = None) -> None:
        """Change the payload type for subsequent packets.

        The control/negotiation channel must notify the receiver before this
        method is used. Existing sequence-number and SSRC state is preserved.
        """
        normalized = Codec(codec.lower())
        with self._lock:
            if self._closed:
                raise RTPSenderError("Cannot change codec on a closed RTP sender")
            self._codec = normalized
            self._payload_type = (
                get_payload_type(normalized)
                if payload_type is None
                else validate_payload_type(payload_type)
            )

    def send_payload(
        self,
        payload: bytes | bytearray | memoryview,
        timestamp: int,
        *,
        marker: bool = False,
    ) -> RTPPacket:
        """Wrap one codec-specific RTP payload and send its UDP datagram."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if not isinstance(marker, bool):
            raise TypeError("marker must be a boolean")
        data = bytes(payload)
        timestamp = self._validate_uint(timestamp, 32, "timestamp")

        with self._lock:
            if self._closed:
                raise RTPSenderError("Cannot send on a closed RTP sender")
            packet = RTPPacket(
                payload_type=self._payload_type,
                sequence_number=self._sequence_number,
                timestamp=timestamp,
                ssrc=self.ssrc,
                payload=data,
                marker=marker,
            )
            datagram = packet.to_bytes()
            if len(datagram) > self.max_datagram_size:
                raise RTPSenderError(
                    f"RTP datagram is {len(datagram)} bytes; limit is "
                    f"{self.max_datagram_size}. Packetize the media payload first."
                )
            try:
                sent = self._socket.sendto(datagram, self.destination)
            except OSError as exc:
                raise RTPSenderError(f"UDP send failed: {exc}") from exc
            if sent != len(datagram):
                raise RTPSenderError(f"UDP socket sent {sent} of {len(datagram)} bytes")
            self._sequence_number = (self._sequence_number + 1) & 0xFFFF
        return packet

    def send_packets(
        self, packets: Iterable[tuple[bytes, bool]], timestamp: int
    ) -> list[RTPPacket]:
        """Send packetizer output sharing one RTP timestamp.

        Each iterable item is ``(payload, marker)``. This is suitable for an
        H.264 access unit fragmented into multiple FU-A RTP payloads.
        """
        sent: list[RTPPacket] = []
        for payload, marker in packets:
            sent.append(self.send_payload(payload, timestamp, marker=marker))
        return sent

    def send_h264_access_unit(
        self,
        access_unit: bytes | Iterable[bytes],
        timestamp: int,
        *,
        max_payload_size: int | None = None,
    ) -> list[RTPPacket]:
        """Packetize and send one H.264 access unit (RFC 6184 mode 1)."""
        if self.codec is not Codec.H264:
            raise RTPSenderError("send_h264_access_unit requires the H.264 codec")
        from rtp_live_streaming.sender.packetization.h264 import packetize_access_unit

        payload_limit = (
            self.max_datagram_size - 12
            if max_payload_size is None
            else max_payload_size
        )
        payloads = packetize_access_unit(access_unit, payload_limit)
        return self.send_packets(
            ((item.payload, item.marker) for item in payloads), timestamp
        )

    def close(self) -> None:
        """Close the underlying socket if this instance created it."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_socket:
                self._socket.close()

    def __enter__(self) -> RTPSender:
        if self._closed:
            raise RTPSenderError("Cannot re-enter a closed RTP sender")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["RTPSender", "RTPSenderError"]
