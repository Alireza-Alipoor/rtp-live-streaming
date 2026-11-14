"""RTCP Sender Reports over UDP.

The media sender should call :meth:`record_packet` after each RTP datagram has
been sent successfully. :class:`RTCPSender` uses those counters to generate
RFC 3550 Sender Reports and sends them as compound RTCP packets with an SDES
CNAME item.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from rtp_live_streaming.common.rtcp import (
    RawRTCPPacket,
    ReportBlock,
    RTCPError,
    RTCPPacket,
    SenderReport,
    serialize_compound_rtcp,
    unix_to_ntp,
)

if TYPE_CHECKING:
    from rtp_live_streaming.common.rtp_packet import RTPPacket


class RTCPSenderError(RuntimeError):
    """Raised when an RTCP sender operation cannot be completed."""


ErrorCallback = Callable[[Exception], None]


def _validate_uint(value: int, bits: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must fit in {bits} bits")
    return value


def _make_sdes_cname(ssrc: int, cname: str) -> RawRTCPPacket:
    """Build an SDES packet containing one source's CNAME item."""
    encoded = cname.encode("utf-8")
    if not encoded:
        raise ValueError("cname cannot be empty")
    if len(encoded) > 255:
        raise ValueError("cname must not exceed 255 UTF-8 bytes")

    # SDES: SSRC, CNAME item (type=1, length, value), END, word padding.
    chunk = ssrc.to_bytes(4, "big") + bytes((1, len(encoded))) + encoded + b"\x00"
    chunk += b"\x00" * (-len(chunk) % 4)
    return RawRTCPPacket(packet_type=202, count=1, body=chunk)


class RTCPSender:
    """Generate and transmit periodic RTCP Sender Reports.

    Parameters
    ----------
    destination_host, destination_port:
        Remote RTCP UDP endpoint. In the conventional RTP/RTCP port pair this
        is usually the RTP port plus one.
    ssrc:
        Synchronization source identifier used by the corresponding RTPSender.
    clock_rate:
        RTP media clock rate (90_000 for video, 48_000 for Opus, 8_000 for
        PCMU). It is used only to extrapolate the RTP timestamp between media
        packets when a report is emitted.
    interval:
        Sender Report interval in seconds.
    cname:
        RTCP SDES canonical name. A stable per-source value is generated when
        omitted.
    sock:
        Optional UDP socket, useful for tests. The socket is not closed unless
        this instance created it.
    on_error:
        Optional callback for errors in the background reporting thread.

    Call ``record_packet(packet)`` only for RTP packets successfully sent.
    The octet count includes RTP payload octets, not the RTP header.
    """

    def __init__(
        self,
        destination_host: str,
        destination_port: int,
        ssrc: int,
        *,
        clock_rate: int = 90_000,
        interval: float = 5.0,
        cname: str | None = None,
        sock: socket.socket | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        if not destination_host:
            raise ValueError("destination_host cannot be empty")
        if not isinstance(destination_port, int) or isinstance(destination_port, bool):
            raise TypeError("destination_port must be an integer")
        if not 1 <= destination_port <= 65535:
            raise ValueError("destination_port must be between 1 and 65535")
        self.ssrc = _validate_uint(ssrc, 32, "ssrc")
        self.clock_rate = _validate_uint(clock_rate, 32, "clock_rate")
        if self.clock_rate == 0:
            raise ValueError("clock_rate must be positive")
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise TypeError("interval must be a number")
        if interval <= 0:
            raise ValueError("interval must be positive")

        self.destination = (destination_host, destination_port)
        self.interval = float(interval)
        self.cname = cname or f"rtp-stream-{self.ssrc:08x}"
        # Validate CNAME eagerly, before opening a socket.
        _make_sdes_cname(self.ssrc, self.cname)

        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._owns_socket = sock is None
        self._on_error = on_error
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._packet_count = 0
        self._octet_count = 0
        self._last_rtp_timestamp = 0
        self._last_rtp_monotonic = time.monotonic()
        self._reports_sent = 0

    @property
    def sender_packet_count(self) -> int:
        """Number of successfully sent RTP packets (modulo 2**32)."""
        with self._lock:
            return self._packet_count

    @property
    def sender_octet_count(self) -> int:
        """Number of RTP payload octets sent (modulo 2**32)."""
        with self._lock:
            return self._octet_count

    @property
    def reports_sent(self) -> int:
        with self._lock:
            return self._reports_sent

    @property
    def closed(self) -> bool:
        return self._closed

    def record_packet(self, packet: RTPPacket) -> None:
        """Account for one successfully transmitted RTP packet."""
        from rtp_live_streaming.common.rtp_packet import RTPPacket

        if not isinstance(packet, RTPPacket):
            raise TypeError("packet must be an RTPPacket")
        if packet.ssrc != self.ssrc:
            raise ValueError("RTP packet SSRC does not match RTCP sender SSRC")
        with self._lock:
            if self._closed:
                raise RTCPSenderError("Cannot record packets on a closed RTCP sender")
            self._packet_count = (self._packet_count + 1) & 0xFFFFFFFF
            self._octet_count = (self._octet_count + len(packet.payload)) & 0xFFFFFFFF
            self._last_rtp_timestamp = packet.timestamp
            self._last_rtp_monotonic = time.monotonic()

    def _current_rtp_timestamp(self, now_monotonic: float | None = None) -> int:
        with self._lock:
            timestamp = self._last_rtp_timestamp
            elapsed = max(
                0.0, (now_monotonic or time.monotonic()) - self._last_rtp_monotonic
            )
        return (timestamp + int(elapsed * self.clock_rate)) & 0xFFFFFFFF

    def build_sender_report(
        self,
        *,
        rtp_timestamp: int | None = None,
        unix_time: float | None = None,
        report_blocks: Iterable[ReportBlock] = (),
    ) -> SenderReport:
        """Create an SR snapshot without sending it."""
        wall_time = time.time() if unix_time is None else unix_time
        ntp_seconds, ntp_fraction = unix_to_ntp(wall_time)
        with self._lock:
            packet_count = self._packet_count
            octet_count = self._octet_count
            last_timestamp = self._last_rtp_timestamp
            last_monotonic = self._last_rtp_monotonic

        if rtp_timestamp is None:
            elapsed = max(0.0, time.monotonic() - last_monotonic)
            rtp_timestamp = (
                last_timestamp + int(elapsed * self.clock_rate)
            ) & 0xFFFFFFFF
        else:
            _validate_uint(rtp_timestamp, 32, "rtp_timestamp")

        return SenderReport(
            ssrc=self.ssrc,
            ntp_seconds=ntp_seconds,
            ntp_fraction=ntp_fraction,
            rtp_timestamp=rtp_timestamp,
            sender_packet_count=packet_count,
            sender_octet_count=octet_count,
            report_blocks=tuple(report_blocks),
        )

    def send_report(
        self,
        *,
        rtp_timestamp: int | None = None,
        unix_time: float | None = None,
        report_blocks: Iterable[ReportBlock] = (),
    ) -> SenderReport:
        """Send an SR + SDES CNAME compound RTCP datagram and return the SR."""
        with self._lock:
            if self._closed:
                raise RTCPSenderError("Cannot send on a closed RTCP sender")

        report = self.build_sender_report(
            rtp_timestamp=rtp_timestamp,
            unix_time=unix_time,
            report_blocks=report_blocks,
        )
        datagram = serialize_compound_rtcp(
            (report, _make_sdes_cname(self.ssrc, self.cname))
        )
        try:
            sent = self._socket.sendto(datagram, self.destination)
        except OSError as exc:
            raise RTCPSenderError(f"UDP RTCP send failed: {exc}") from exc
        if sent != len(datagram):
            raise RTCPSenderError(
                f"UDP socket sent {sent} of {len(datagram)} RTCP bytes"
            )

        with self._lock:
            self._reports_sent += 1
        return report

    def send_compound(self, packets: Iterable[RTCPPacket]) -> int:
        """Send caller-provided RTCP packets as one datagram.

        This low-level method does not add an SDES packet; callers are
        responsible for supplying a valid compound packet when required.
        Returns the number of bytes transmitted.
        """
        with self._lock:
            if self._closed:
                raise RTCPSenderError("Cannot send on a closed RTCP sender")
        try:
            datagram = serialize_compound_rtcp(tuple(packets))
            sent = self._socket.sendto(datagram, self.destination)
        except (OSError, RTCPError) as exc:
            raise RTCPSenderError(f"RTCP send failed: {exc}") from exc
        if sent != len(datagram):
            raise RTCPSenderError(
                f"UDP socket sent {sent} of {len(datagram)} RTCP bytes"
            )
        return sent

    def start(self, *, daemon: bool = True, send_immediately: bool = True) -> None:
        """Start periodic Sender Reports in a background thread."""
        with self._lock:
            if self._closed:
                raise RTCPSenderError("Cannot start a closed RTCP sender")
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"rtcp-sender-{self.ssrc:08x}",
                daemon=daemon,
            )
            thread = self._thread
        if send_immediately:
            self.send_report()
        thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                self.send_report()
            except Exception as exc:  # Keep the media sender alive on RTCP errors.
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass

    def stop(self) -> None:
        """Stop the periodic reporting thread without closing the socket."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.interval + 0.5))
        self._thread = None

    def close(self) -> None:
        """Stop reporting and close the socket if this instance owns it."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.stop()
        if self._owns_socket:
            self._socket.close()

    def __enter__(self) -> RTCPSender:
        if self._closed:
            raise RTCPSenderError("Cannot re-enter a closed RTCP sender")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["RTCPSender", "RTCPSenderError"]
