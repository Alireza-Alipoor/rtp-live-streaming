"""UDP RTP receiver with parsing, per-SSRC jitter buffering and callbacks."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from rtp_live_streaming.common.payload_types import (
    PAYLOAD_TYPE_TO_CODEC,
    Codec,
    validate_payload_type,
)
from rtp_live_streaming.common.rtp_packet import RTPPacket, RTPPacketError
from rtp_live_streaming.receiver.jitter_buffer import JitterBuffer


class RTPReceiverError(RuntimeError):
    """Raised when an RTP receiver operation cannot be completed."""


@dataclass(frozen=True, slots=True)
class RTPReceiverStats:
    """Receiver counters; malformed and unsupported datagrams are excluded
    from the jitter-buffer counters.
    """

    datagrams_received: int
    malformed_datagrams: int
    unsupported_payload_types: int
    packets_delivered: int
    active_sources: int


PacketCallback = Callable[[RTPPacket], None]
ErrorCallback = Callable[[Exception], None]


class RTPReceiver:
    """Receive RTP/UDP datagrams and release packets in playout order.

    A separate :class:`JitterBuffer` is maintained for each SSRC. The receiver
    invokes ``on_packet`` only after the packet's configured playout delay and
    sequence-order checks have been applied. Depacketization and decoding are
    intentionally left to the caller.

    Use one receiver instance per media RTP port (for example, one for video
    port 5004 and another for audio port 5006).
    """

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        bind_port: int = 5004,
        *,
        jitter_buffer_ms: float = 50,
        max_datagram_size: int = 65507,
        socket_buffer_size: int = 1_048_576,
        receive_timeout: float = 0.02,
        on_packet: PacketCallback | None = None,
        on_error: ErrorCallback | None = None,
        sock: socket.socket | None = None,
        payload_type_map: dict[int, Codec | str] | None = None,
    ) -> None:
        if not isinstance(bind_port, int) or isinstance(bind_port, bool):
            raise TypeError("bind_port must be an integer")
        if not 0 <= bind_port <= 65535:
            raise ValueError("bind_port must be between 0 and 65535")
        if max_datagram_size < 12 or max_datagram_size > 65507:
            raise ValueError("max_datagram_size must be between 12 and 65507")
        if receive_timeout <= 0:
            raise ValueError("receive_timeout must be positive")
        if socket_buffer_size <= 0:
            raise ValueError("socket_buffer_size must be positive")

        self.bind_address = (bind_host, bind_port)
        self.jitter_buffer_ms = float(jitter_buffer_ms)
        self.max_datagram_size = max_datagram_size
        self.receive_timeout = float(receive_timeout)
        self.on_packet = on_packet
        self.on_error = on_error
        self._payload_type_map = dict(PAYLOAD_TYPE_TO_CODEC)
        if payload_type_map is not None:
            for payload_type, codec in payload_type_map.items():
                self._payload_type_map[validate_payload_type(payload_type)] = Codec(
                    codec.lower()
                )

        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._owns_socket = sock is None
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, socket_buffer_size)
        self._socket.settimeout(self.receive_timeout)
        if self._owns_socket:
            self._socket.bind(self.bind_address)

        self._buffers: dict[int, JitterBuffer] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._datagrams_received = 0
        self._malformed_datagrams = 0
        self._unsupported_payload_types = 0
        self._packets_delivered = 0

    @property
    def address(self) -> tuple[str, int]:
        """Actual local address (useful when binding to port 0 in tests)."""
        host, port = self._socket.getsockname()[:2]
        return str(host), int(port)

    @property
    def stats(self) -> RTPReceiverStats:
        with self._lock:
            return RTPReceiverStats(
                datagrams_received=self._datagrams_received,
                malformed_datagrams=self._malformed_datagrams,
                unsupported_payload_types=self._unsupported_payload_types,
                packets_delivered=self._packets_delivered,
                active_sources=len(self._buffers),
            )

    @property
    def buffer_stats(self) -> dict[int, object]:
        """Return a snapshot of jitter-buffer statistics keyed by SSRC."""
        with self._lock:
            return {ssrc: buffer.stats for ssrc, buffer in self._buffers.items()}

    def codec_for_payload_type(self, payload_type: int) -> Codec:
        """Resolve a payload type using defaults plus negotiated overrides."""
        validate_payload_type(payload_type)
        try:
            return self._payload_type_map[payload_type]
        except KeyError:
            raise ValueError(
                f"No codec registered for RTP payload type {payload_type}."
            ) from None

    def receive_once(self, timeout: float | None = None) -> list[RTPPacket]:
        """Receive at most one datagram, then release all packets now ready.

        This method is useful for synchronous applications and tests. It may
        return packets received on an earlier call because their jitter-buffer
        deadlines have since expired.
        """
        if self._closed:
            raise RTPReceiverError("Cannot receive on a closed RTP receiver")
        if timeout is not None:
            if timeout <= 0:
                raise ValueError("timeout must be positive")
            self._socket.settimeout(float(timeout))

        try:
            datagram, _peer = self._socket.recvfrom(self.max_datagram_size)
        except TimeoutError:
            datagram = None
        except OSError as exc:
            if self._closed:
                return []
            raise RTPReceiverError(f"UDP receive failed: {exc}") from exc

        if datagram is not None:
            with self._lock:
                self._datagrams_received += 1
            try:
                packet = RTPPacket.from_bytes(datagram)
            except (RTPPacketError, ValueError) as exc:
                with self._lock:
                    self._malformed_datagrams += 1
                self._report_error(exc)
            else:
                try:
                    self.codec_for_payload_type(packet.payload_type)
                except ValueError as exc:
                    with self._lock:
                        self._unsupported_payload_types += 1
                    self._report_error(exc)
                else:
                    with self._lock:
                        jitter_buffer = self._buffers.setdefault(
                            packet.ssrc, JitterBuffer(self.jitter_buffer_ms)
                        )
                    jitter_buffer.push(packet)

        ready = self._drain_ready()
        for packet in ready:
            if self.on_packet is not None:
                try:
                    self.on_packet(packet)
                except Exception as exc:  # callback errors must not kill receive loop
                    self._report_error(exc)
        with self._lock:
            self._packets_delivered += len(ready)
        return ready

    def _drain_ready(self) -> list[RTPPacket]:
        now = time.monotonic()
        with self._lock:
            buffers = tuple(self._buffers.values())
        ready = [packet for buffer in buffers for packet in buffer.pop_ready(now)]
        # Packets from independent SSRCs have no common sequence-number space.
        # Preserve each source's ordering; cross-source ordering is arrival-loop order.
        return ready

    def start(self, *, daemon: bool = True) -> None:
        """Start the background receive loop."""
        if self._closed:
            raise RTPReceiverError("Cannot start a closed RTP receiver")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rtp-receiver-{self.address[1]}",
            daemon=daemon,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.receive_once()
            except RTPReceiverError as exc:
                if not self._stop_event.is_set():
                    self._report_error(exc)
                    break

    def _report_error(self, error: Exception) -> None:
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception:
                pass

    def stop(self, *, flush: bool = False) -> list[RTPPacket]:
        """Stop the background loop and optionally release queued packets."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.receive_timeout * 4))
        self._thread = None
        ready: list[RTPPacket] = []
        if flush:
            with self._lock:
                buffers = tuple(self._buffers.values())
            ready = [packet for buffer in buffers for packet in buffer.flush()]
            for packet in ready:
                if self.on_packet is not None:
                    self.on_packet(packet)
            with self._lock:
                self._packets_delivered += len(ready)
        return ready

    def close(self) -> None:
        """Stop receiving and close the socket if this instance owns it."""
        if self._closed:
            return
        self.stop()
        self._closed = True
        if self._owns_socket:
            self._socket.close()

    def __enter__(self) -> RTPReceiver:
        if self._closed:
            raise RTPReceiverError("Cannot re-enter a closed RTP receiver")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["RTPReceiver", "RTPReceiverError", "RTPReceiverStats"]
