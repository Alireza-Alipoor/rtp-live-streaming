"""Async TCP server for the application-level RTP control channel.

Messages use newline-delimited JSON. The control channel is independent of
RTP/RTCP media sockets and carries configuration and codec-negotiation data.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

from rtp_live_streaming.common.config import (
    DEFAULT_CONTROL_PORT,
    AudioConfig,
    VideoConfig,
)

from .messages import (
    MAX_MESSAGE_SIZE,
    ControlMessage,
    MessageError,
    MessageType,
    StreamType,
    response_message,
)
from .negotiation import CodecTransition, NegotiationError, NegotiationManager

logger = logging.getLogger(__name__)

ConfigCallback = Callable[[StreamType, VideoConfig | AudioConfig], Any]
CodecSwitchCallback = Callable[[CodecTransition], Any]


class ControlServerError(RuntimeError):
    """Raised when the control server cannot be started or used."""


class ControlServer:
    """Async TCP control server backed by a :class:`NegotiationManager`.

    ``on_configure`` is called with a validated configuration before it is
    recorded as active. ``on_codec_switch`` is called with a PREPARING
    transition so the receiver can prepare its decoder. Either callback may
    be synchronous or asynchronous; raising rejects the request.

    A codec-switch ACK means preparation succeeded and the manager committed
    the control-plane state. Frame-accurate switching remains an application
    responsibility: coordinate the media boundary in the callback or a higher
    layer. This server does not inspect or modify RTP packets.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_CONTROL_PORT,
        *,
        negotiation_manager: NegotiationManager | None = None,
        on_configure: ConfigCallback | None = None,
        on_codec_switch: CodecSwitchCallback | None = None,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 0 <= port <= 65535
        ):
            raise ValueError("port must be an integer between 0 and 65535")

        self.host = host
        self.port = port
        self.negotiation_manager = negotiation_manager or NegotiationManager()
        self.on_configure = on_configure
        self.on_codec_switch = on_codec_switch
        self._server: asyncio.Server | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._negotiation_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the TCP listener is active."""
        return self._server is not None and self._server.is_serving()

    @property
    def address(self) -> tuple[str, int] | None:
        """Bound address, including the selected port when ``port=0``."""
        if self._server is None or not self._server.sockets:
            return None
        bound = self._server.sockets[0].getsockname()
        return str(bound[0]), int(bound[1])

    async def start(self) -> None:
        """Start accepting TCP control connections."""
        if self._server is not None:
            raise ControlServerError("Control server has already been started")
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                host=self.host,
                port=self.port,
                limit=MAX_MESSAGE_SIZE + 1,
            )
        except OSError as exc:
            raise ControlServerError(
                f"Could not bind control server to {self.host}:{self.port}: {exc}"
            ) from exc
        logger.info("Control server listening on %s:%s", *self.address)

    async def serve_forever(self) -> None:
        """Start the listener if necessary and serve until cancelled/closed."""
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        """Stop listening and close all active client connections."""
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

        writers = tuple(self._clients)
        for writer in writers:
            writer.close()
        if writers:
            await asyncio.gather(
                *(writer.wait_closed() for writer in writers),
                return_exceptions=True,
            )
        self._clients.clear()
        logger.info("Control server closed")

    async def __aenter__(self) -> ControlServer:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        self._clients.add(writer)
        logger.info("Control client connected: %s", peer)
        try:
            while not reader.at_eof():
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    await self._send_error(
                        writer, "Control message exceeds the maximum size"
                    )
                    break
                if not line:
                    break
                if len(line) > MAX_MESSAGE_SIZE + 1:
                    await self._send_error(
                        writer, "Control message exceeds the maximum size"
                    )
                    break

                raw_message = line.rstrip(b"\r\n")
                if not raw_message:
                    continue
                try:
                    message = ControlMessage.from_json(raw_message)
                except MessageError as exc:
                    await self._send_error(writer, str(exc))
                    continue

                try:
                    response = await self._process_message(message)
                except Exception:
                    logger.exception(
                        "Failed to process control request %s from %s",
                        message.request_id,
                        peer,
                    )
                    response = self._error_response(
                        message.request_id, "Internal control-server error"
                    )
                await self._write_message(writer, response)
        except (ConnectionError, asyncio.CancelledError):
            raise
        finally:
            self._clients.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            logger.info("Control client disconnected: %s", peer)

    async def _process_message(self, message: ControlMessage) -> ControlMessage:
        if message.type is MessageType.PING:
            return ControlMessage.create(
                MessageType.PONG,
                {"echo": dict(message.payload)},
                request_id=message.request_id,
            )
        if message.type is MessageType.CONFIGURE:
            async with self._negotiation_lock:
                return await self._handle_configure(message)
        if message.type is MessageType.CODEC_SWITCH:
            async with self._negotiation_lock:
                return await self._handle_codec_switch(message)
        return self._error_response(
            message.request_id,
            f"Message type {message.type.value!r} is not a valid client request",
        )

    async def _handle_configure(self, request: ControlMessage) -> ControlMessage:
        try:
            stream, config_data = self._parse_configure_payload(request.payload)
            validated = self.negotiation_manager.validate_config(stream, config_data)
            if self.on_configure is not None:
                await self._invoke_callback(self.on_configure, stream, validated)
            active = self.negotiation_manager.negotiate_config(stream, validated)
        except (NegotiationError, MessageError, TypeError, ValueError) as exc:
            return response_message(request, accepted=False, reason=str(exc))
        except Exception as exc:
            logger.exception("Receiver rejected configuration request")
            return response_message(
                request,
                accepted=False,
                reason=f"Receiver could not apply configuration: {exc}",
            )
        return response_message(
            request,
            accepted=True,
            payload={"stream": stream.value, "config": asdict(active)},
        )

    async def _handle_codec_switch(self, request: ControlMessage) -> ControlMessage:
        try:
            stream, codec, payload_type = self._parse_codec_switch_payload(
                request.payload
            )
            transition = self.negotiation_manager.prepare_transition(
                stream,
                codec,
                payload_type=payload_type,
                request_id=request.request_id,
            )
            if self.on_codec_switch is not None:
                await self._invoke_callback(self.on_codec_switch, transition)
            self.negotiation_manager.mark_ready(request.request_id)
            committed = self.negotiation_manager.commit_transition(request.request_id)
        except (NegotiationError, MessageError, TypeError, ValueError) as exc:
            self._abort_if_pending(request.request_id, str(exc))
            return response_message(request, accepted=False, reason=str(exc))
        except Exception as exc:
            logger.exception("Receiver rejected codec-switch request")
            self._abort_if_pending(request.request_id, str(exc))
            return response_message(
                request,
                accepted=False,
                reason=f"Receiver could not prepare codec: {exc}",
            )
        return response_message(
            request,
            accepted=True,
            payload={
                "stream": committed.stream.value,
                "old_codec": committed.old_codec.value,
                "codec": committed.new_codec.value,
                "payload_type": committed.payload_type,
                "state": committed.state.value,
            },
        )

    @staticmethod
    def _parse_configure_payload(
        payload: Mapping[str, Any],
    ) -> tuple[StreamType, Mapping[str, Any]]:
        if set(payload) != {"stream", "config"}:
            raise MessageError(
                "configure payload must contain exactly 'stream' and 'config'"
            )
        try:
            stream = StreamType(payload["stream"])
        except (ValueError, TypeError):
            raise MessageError(
                f"Unsupported stream: {payload.get('stream')!r}"
            ) from None
        config = payload["config"]
        if not isinstance(config, Mapping):
            raise MessageError("configure payload 'config' must be an object")
        return stream, config

    @staticmethod
    def _parse_codec_switch_payload(
        payload: Mapping[str, Any],
    ) -> tuple[StreamType, str, int | None]:
        allowed = {"stream", "codec", "payload_type"}
        if not {"stream", "codec"} <= set(payload) or set(payload) - allowed:
            raise MessageError(
                "codec_switch payload requires 'stream' and 'codec'; "
                "only optional 'payload_type' is allowed"
            )
        try:
            stream = StreamType(payload["stream"])
        except (ValueError, TypeError):
            raise MessageError(
                f"Unsupported stream: {payload.get('stream')!r}"
            ) from None
        codec = payload["codec"]
        if not isinstance(codec, str) or not codec.strip():
            raise MessageError("codec_switch payload 'codec' must be a string")
        payload_type = payload.get("payload_type")
        if payload_type is not None and (
            not isinstance(payload_type, int) or isinstance(payload_type, bool)
        ):
            raise MessageError("payload_type must be an integer")
        return stream, codec, payload_type

    def _abort_if_pending(self, request_id: str, reason: str) -> None:
        if self.negotiation_manager.get_pending_transition(request_id) is not None:
            self.negotiation_manager.abort_transition(request_id, reason=reason)

    @staticmethod
    async def _invoke_callback(callback: Callable[..., Any], *args: Any) -> Any:
        result = callback(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _send_error(self, writer: asyncio.StreamWriter, reason: str) -> None:
        await self._write_message(writer, self._error_response(None, reason))

    @staticmethod
    def _error_response(request_id: str | None, reason: str) -> ControlMessage:
        return ControlMessage.create(
            MessageType.ERROR,
            {"reason": reason},
            request_id=request_id or "server-error",
        )

    @staticmethod
    async def _write_message(
        writer: asyncio.StreamWriter, message: ControlMessage
    ) -> None:
        writer.write(message.to_json().encode("utf-8") + b"\n")
        await writer.drain()
