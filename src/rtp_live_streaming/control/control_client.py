"""Async TCP client for the application-level RTP control channel.

The client uses newline-delimited JSON, matching ``ControlServer``.  It
serializes requests on each connection so that responses remain correlated
with their requests.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from rtp_live_streaming.common.config import DEFAULT_CONTROL_PORT

from .messages import (
    MAX_MESSAGE_SIZE,
    ControlMessage,
    MessageError,
    MessageType,
    StreamType,
    codec_switch_message,
    configure_message,
)

logger = logging.getLogger(__name__)


class ControlClientError(RuntimeError):
    """Base exception for control-client failures."""


class ControlConnectionError(ControlClientError):
    """Raised when the TCP control connection cannot be established or used."""


class ControlProtocolError(ControlClientError):
    """Raised when the peer sends an invalid or unexpected response."""


class ControlRequestTimeout(ControlClientError, TimeoutError):
    """Raised when a control request does not receive a response in time."""


class ControlClient:
    """Maintain a TCP connection to an RTP receiver's control server.

    Args:
        host: Receiver hostname or IP address.
        port: Receiver's TCP control port.
        connect_timeout: Maximum time allowed to establish a connection.
        request_timeout: Default maximum time allowed for each response.

    ``connect()`` may be called explicitly, or the first request will connect
    automatically.  Calls are serialized per connection; use one client per
    receiver.  A NACK is returned as a normal ``ControlMessage`` so callers
    can inspect its ``type`` and ``payload["reason"]``.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_CONTROL_PORT,
        *,
        connect_timeout: float = 5.0,
        request_timeout: float = 10.0,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string")
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 0 <= port <= 65535
        ):
            raise ValueError("port must be an integer between 0 and 65535")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than zero")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")

        self.host = host
        self.port = port
        self.connect_timeout = float(connect_timeout)
        self.request_timeout = float(request_timeout)

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Whether the TCP stream is currently open."""
        return (
            self._writer is not None
            and not self._writer.is_closing()
            and self._reader is not None
        )

    @property
    def peer_address(self) -> tuple[str, int] | None:
        """Return the connected peer address, if connected."""
        if not self.is_connected:
            return None
        peer = self._writer.get_extra_info("peername")  # type: ignore[union-attr]
        if peer is None:
            return None
        return str(peer[0]), int(peer[1])

    async def connect(self) -> None:
        """Open the TCP connection to the receiver."""
        if self.is_connected:
            return

        await self._discard_connection()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.host,
                    self.port,
                    limit=MAX_MESSAGE_SIZE + 1,
                ),
                timeout=self.connect_timeout,
            )
        except (TimeoutError, OSError) as exc:
            raise ControlConnectionError(
                f"Could not connect to control server at {self.host}:{self.port}: {exc}"
            ) from exc

        self._reader = reader
        self._writer = writer
        logger.info("Connected to control server at %s:%s", self.host, self.port)

    async def close(self) -> None:
        """Close the TCP connection cleanly."""
        writer = self._writer
        self._reader = None
        self._writer = None

        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

        logger.info("Control connection to %s:%s closed", self.host, self.port)

    async def request(
        self,
        message: ControlMessage,
        *,
        timeout: float | None = None,
    ) -> ControlMessage:
        """Send one request and wait for its matching response.

        The request/response exchange is protected by a lock.  If the
        connection fails or times out, it is discarded because the client
        cannot safely determine whether the peer processed the request.
        """
        if not isinstance(message, ControlMessage):
            raise TypeError("message must be a ControlMessage")
        if timeout is None:
            timeout = self.request_timeout
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        async with self._request_lock:
            await self.connect()
            assert self._reader is not None
            assert self._writer is not None

            try:
                self._writer.write(message.to_json().encode("utf-8") + b"\n")
                await asyncio.wait_for(self._writer.drain(), timeout=timeout)
                line = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                await self._discard_connection()
                raise ControlRequestTimeout(
                    f"Timed out waiting for response to {message.type.value}"
                ) from exc
            except (ConnectionError, OSError, ValueError) as exc:
                await self._discard_connection()
                raise ControlConnectionError(
                    f"Control connection failed while sending "
                    f"{message.type.value}: {exc}"
                ) from exc

            if not line:
                await self._discard_connection()
                raise ControlConnectionError(
                    "Control server closed the connection before responding"
                )
            if len(line) > MAX_MESSAGE_SIZE + 1:
                await self._discard_connection()
                raise ControlProtocolError(
                    "Control server response exceeds the maximum message size"
                )

            try:
                response = ControlMessage.from_json(line.rstrip(b"\r\n"))
            except MessageError as exc:
                await self._discard_connection()
                raise ControlProtocolError(
                    f"Control server sent an invalid response: {exc}"
                ) from exc

            if response.request_id != message.request_id:
                await self._discard_connection()
                raise ControlProtocolError(
                    "Response request_id does not match the outstanding request "
                    f"({response.request_id!r} != {message.request_id!r})"
                )

            return response

    async def configure(
        self,
        stream: StreamType | str,
        config: Any,
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> ControlMessage:
        """Ask the receiver to validate and apply an audio/video config."""
        message = configure_message(stream, config, request_id=request_id)
        return await self.request(message, timeout=timeout)

    async def switch_codec(
        self,
        stream: StreamType | str,
        codec: str,
        *,
        payload_type: int | None = None,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> ControlMessage:
        """Request a coordinated codec switch on the receiver."""
        message = codec_switch_message(
            stream,
            codec,
            payload_type=payload_type,
            request_id=request_id,
        )
        return await self.request(message, timeout=timeout)

    async def ping(
        self,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ControlMessage:
        """Check connectivity and return the server's PONG response."""
        message = ControlMessage.create(
            MessageType.PING,
            {} if payload is None else payload,
        )
        response = await self.request(message, timeout=timeout)
        if response.type is not MessageType.PONG:
            raise ControlProtocolError(f"Expected PONG, received {response.type.value}")
        return response

    async def _discard_connection(self) -> None:
        """Drop local stream references after an unusable connection."""
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def __aenter__(self) -> ControlClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.close()
