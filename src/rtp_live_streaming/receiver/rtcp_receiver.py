"""Receive and parse RTCP/UDP datagrams.

The receiver accepts both single RTCP packets and compound RTCP datagrams.
Parsing is delegated to :mod:`rtp_live_streaming.common.rtcp`, which supports
Sender Reports, Receiver Reports, and preserves unknown packet types as raw
RTCP packets (for example SDES packets).
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass

from rtp_live_streaming.common.rtcp import (
    ReceiverReport,
    RTCPError,
    RTCPPacket,
    SenderReport,
    parse_compound_rtcp,
)


class RTCPReceiverError(RuntimeError):
    """Raised when an RTCP receiver operation cannot be completed."""


@dataclass(frozen=True, slots=True)
class RTCPReceiverStats:
    """Snapshot of RTCP receive counters."""

    datagrams_received: int
    malformed_datagrams: int
    packets_received: int
    sender_reports: int
    receiver_reports: int
    other_packets: int
    active_sources: int


PacketCallback = Callable[[RTCPPacket], None]
ErrorCallback = Callable[[Exception], None]


class RTCPReceiver:
    """Receive RTCP datagrams over UDP.

    Parameters mirror :class:`~rtp_live_streaming.receiver.rtp_receiver.RTPReceiver`.
    Bind one instance to the RTCP port associated with an RTP media stream
    (commonly RTP port + 1).

    ``on_packet`` is called once for every parsed packet in a compound
    datagram. Callback failures are passed to ``on_error`` and do not stop
    the receive loop.

    An optional pre-created socket can be supplied for tests or for a socket
    managed by the application. Such a socket is not bound or closed by this
    instance; it is still configured with the requested receive timeout and
    receive-buffer size.
    """

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        bind_port: int = 5005,
        *,
        max_datagram_size: int = 65507,
        socket_buffer_size: int = 1_048_576,
        receive_timeout: float = 0.2,
        on_packet: PacketCallback | None = None,
        on_error: ErrorCallback | None = None,
        sock: socket.socket | None = None,
    ) -> None:
        if not isinstance(bind_port, int) or isinstance(bind_port, bool):
            raise TypeError("bind_port must be an integer")
        if not 0 <= bind_port <= 65535:
            raise ValueError("bind_port must be between 0 and 65535")
        if not isinstance(max_datagram_size, int) or isinstance(
            max_datagram_size, bool
        ):
            raise TypeError("max_datagram_size must be an integer")
        if not 4 <= max_datagram_size <= 65507:
            raise ValueError("max_datagram_size must be between 4 and 65507")
        if not isinstance(socket_buffer_size, int) or isinstance(
            socket_buffer_size, bool
        ):
            raise TypeError("socket_buffer_size must be an integer")
        if socket_buffer_size <= 0:
            raise ValueError("socket_buffer_size must be positive")
        if isinstance(receive_timeout, bool) or not isinstance(
            receive_timeout, (int, float)
        ):
            raise TypeError("receive_timeout must be a number")
        if receive_timeout <= 0:
            raise ValueError("receive_timeout must be positive")

        self.bind_address = (bind_host, bind_port)
        self.max_datagram_size = max_datagram_size
        self.receive_timeout = float(receive_timeout)
        self.on_packet = on_packet
        self.on_error = on_error

        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._owns_socket = sock is None
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, socket_buffer_size)
        self._socket.settimeout(self.receive_timeout)
        if self._owns_socket:
            try:
                self._socket.bind(self.bind_address)
            except OSError:
                self._socket.close()
                raise

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

        self._datagrams_received = 0
        self._malformed_datagrams = 0
        self._packets_received = 0
        self._sender_reports = 0
        self._receiver_reports = 0
        self._other_packets = 0
        self._sources: set[int] = set()
        self._last_peer: tuple[str, int] | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Actual local socket address (useful when binding to port 0)."""
        host, port = self._socket.getsockname()[:2]
        return str(host), int(port)

    @property
    def last_peer(self) -> tuple[str, int] | None:
        """Address of the most recently received RTCP datagram."""
        with self._lock:
            return self._last_peer

    @property
    def stats(self) -> RTCPReceiverStats:
        """Return a thread-safe snapshot of receiver counters."""
        with self._lock:
            return RTCPReceiverStats(
                datagrams_received=self._datagrams_received,
                malformed_datagrams=self._malformed_datagrams,
                packets_received=self._packets_received,
                sender_reports=self._sender_reports,
                receiver_reports=self._receiver_reports,
                other_packets=self._other_packets,
                active_sources=len(self._sources),
            )

    def receive_once(self, timeout: float | None = None) -> tuple[RTCPPacket, ...]:
        """Receive and parse at most one UDP datagram.

        Returns all RTCP packets found in the datagram. A socket timeout
        returns an empty tuple. Malformed datagrams are counted, reported via
        ``on_error``, and also return an empty tuple.
        """
        if self._closed:
            raise RTCPReceiverError("Cannot receive on a closed RTCP receiver")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a number")
            if timeout <= 0:
                raise ValueError("timeout must be positive")
            self._socket.settimeout(float(timeout))

        try:
            datagram, peer = self._socket.recvfrom(self.max_datagram_size)
        except TimeoutError:
            return ()
        except OSError as exc:
            if self._closed or self._stop_event.is_set():
                return ()
            raise RTCPReceiverError(f"UDP RTCP receive failed: {exc}") from exc

        with self._lock:
            self._datagrams_received += 1
            self._last_peer = (str(peer[0]), int(peer[1]))

        try:
            packets = parse_compound_rtcp(datagram)
        except (RTCPError, ValueError) as exc:
            with self._lock:
                self._malformed_datagrams += 1
            self._report_error(exc)
            return ()

        with self._lock:
            for packet in packets:
                self._packets_received += 1
                if isinstance(packet, SenderReport):
                    self._sender_reports += 1
                    self._sources.add(packet.ssrc)
                elif isinstance(packet, ReceiverReport):
                    self._receiver_reports += 1
                    self._sources.add(packet.ssrc)
                else:
                    self._other_packets += 1

        for packet in packets:
            if self.on_packet is not None:
                try:
                    self.on_packet(packet)
                except Exception as exc:  # User callbacks must not kill reception.
                    self._report_error(exc)

        return packets

    def start(self, *, daemon: bool = True) -> None:
        """Start the background receive loop."""
        if self._closed:
            raise RTCPReceiverError("Cannot start a closed RTCP receiver")
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rtcp-receiver-{self.address[1]}",
            daemon=daemon,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.receive_once()
            except RTCPReceiverError as exc:
                if not self._stop_event.is_set():
                    self._report_error(exc)
                    break

    def _report_error(self, error: Exception) -> None:
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the background receive loop without closing the socket."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.receive_timeout * 4))
        self._thread = None

    def close(self) -> None:
        """Stop reception and close the socket if this instance owns it."""
        if self._closed:
            return
        self.stop()
        self._closed = True
        if self._owns_socket:
            self._socket.close()

    def __enter__(self) -> RTCPReceiver:
        if self._closed:
            raise RTCPReceiverError("Cannot re-enter a closed RTCP receiver")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["RTCPReceiver", "RTCPReceiverError", "RTCPReceiverStats"]
