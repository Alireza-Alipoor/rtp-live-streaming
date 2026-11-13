"""Message models and JSON serialization for the application control channel.

The control channel is separate from RTP/RTCP media transport.  Messages are
intended to be carried over a reliable byte stream (for example, TCP).  This
module defines message structure and serialization only; it does not perform
network I/O or negotiation.

Wire format (one JSON object per message)::

    {
        "version": 1,
        "type": "configure",
        "request_id": "a UUID",
        "payload": {
            "stream": "video",
            "config": {"bitrate_kbps": 1000}
        }
    }

The TCP layer is responsible for framing messages, e.g. newline-delimited JSON.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

PROTOCOL_VERSION = 1
MAX_MESSAGE_SIZE = 64 * 1024


class MessageError(ValueError):
    """Raised when a control message is invalid or cannot be decoded."""


class MessageType(StrEnum):
    """Supported application-level control message types."""

    CONFIGURE = "configure"
    CONFIGURE_ACK = "configure_ack"
    CONFIGURE_NACK = "configure_nack"

    CODEC_SWITCH = "codec_switch"
    CODEC_SWITCH_ACK = "codec_switch_ack"
    CODEC_SWITCH_NACK = "codec_switch_nack"

    PING = "ping"
    PONG = "pong"
    ERROR = "error"


class StreamType(StrEnum):
    """Media stream targeted by a control request."""

    VIDEO = "video"
    AUDIO = "audio"


def _json_compatible(value: Any, *, field_name: str = "payload") -> Any:
    """Convert supported Python values into JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value), field_name=field_name)

    if isinstance(value, StrEnum):
        return value.value

    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MessageError(f"{field_name} object keys must be strings")
            converted[key] = _json_compatible(item, field_name=f"{field_name}.{key}")
        return converted

    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise MessageError(
        f"{field_name} contains a value that is not JSON serializable: "
        f"{type(value).__name__}"
    )


def _validate_request_id(request_id: str) -> str:
    if not isinstance(request_id, str) or not request_id.strip():
        raise MessageError("request_id must be a non-empty string")
    if len(request_id) > 128:
        raise MessageError("request_id must not exceed 128 characters")
    return request_id


@dataclass(frozen=True, slots=True)
class ControlMessage:
    """A single application-level control message.

    ``payload`` is deliberately generic so the control protocol can evolve
    without requiring a new Python class for every message type.  The
    negotiation layer validates message-specific payload fields.
    """

    type: MessageType
    request_id: str
    payload: Mapping[str, Any]
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        try:
            normalized_type = (
                self.type
                if isinstance(self.type, MessageType)
                else MessageType(self.type)
            )
        except (ValueError, TypeError):
            raise MessageError(f"Unsupported message type: {self.type!r}") from None

        object.__setattr__(self, "type", normalized_type)
        _validate_request_id(self.request_id)

        if not isinstance(self.payload, Mapping):
            raise MessageError("payload must be a mapping/object")

        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version != PROTOCOL_VERSION
        ):
            raise MessageError(
                f"Unsupported protocol version: {self.version!r}; "
                f"supported version is {PROTOCOL_VERSION}"
            )

        # Validate JSON compatibility at construction time, rather than
        # allowing a later send operation to fail unexpectedly.
        _json_compatible(self.payload)

    @classmethod
    def create(
        cls,
        message_type: MessageType | str,
        payload: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> ControlMessage:
        """Create a message with a generated UUID unless an ID is supplied."""
        try:
            normalized_type = (
                message_type
                if isinstance(message_type, MessageType)
                else MessageType(message_type)
            )
        except (ValueError, TypeError):
            raise MessageError(f"Unsupported message type: {message_type!r}") from None

        return cls(
            type=normalized_type,
            request_id=str(uuid4()) if request_id is None else request_id,
            payload={} if payload is None else payload,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the message as a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "type": self.type.value,
            "request_id": self.request_id,
            "payload": _json_compatible(self.payload),
        }

    def to_json(self) -> str:
        """Serialize the message to compact JSON (without a line delimiter)."""
        try:
            encoded = json.dumps(
                self.to_dict(),
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise MessageError(f"Could not serialize control message: {exc}") from exc

        if len(encoded.encode("utf-8")) > MAX_MESSAGE_SIZE:
            raise MessageError(
                f"Control message exceeds maximum size of {MAX_MESSAGE_SIZE} bytes"
            )
        return encoded

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControlMessage:
        """Parse and validate a message dictionary."""
        if not isinstance(data, Mapping):
            raise MessageError("A control message must be a JSON object")

        required = {"version", "type", "request_id", "payload"}
        missing = required - data.keys()
        if missing:
            raise MessageError(
                f"Missing required message fields: {', '.join(sorted(missing))}"
            )

        extra = data.keys() - required
        if extra:
            raise MessageError(f"Unexpected message fields: {', '.join(sorted(extra))}")

        if not isinstance(data["type"], str):
            raise MessageError("message type must be a string")

        return cls(
            version=data["version"],
            type=data["type"],
            request_id=data["request_id"],
            payload=data["payload"],
        )

    @classmethod
    def from_json(cls, raw: str | bytes | bytearray) -> ControlMessage:
        """Deserialize a JSON message, enforcing the wire-size limit."""
        if isinstance(raw, str):
            try:
                raw_bytes = raw.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise MessageError("Message is not valid UTF-8") from exc
        elif isinstance(raw, (bytes, bytearray)):
            raw_bytes = bytes(raw)
        else:
            raise MessageError("raw message must be str or bytes")

        if len(raw_bytes) > MAX_MESSAGE_SIZE:
            raise MessageError(
                f"Control message exceeds maximum size of {MAX_MESSAGE_SIZE} bytes"
            )

        try:
            decoded = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MessageError("Message is not valid UTF-8") from exc

        try:
            data = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise MessageError(f"Invalid control-message JSON: {exc.msg}") from exc

        if not isinstance(data, dict):
            raise MessageError("A control message must be a JSON object")

        return cls.from_dict(data)


def configure_message(
    stream: StreamType | str,
    config: Any,
    *,
    request_id: str | None = None,
) -> ControlMessage:
    """Build a configuration request for an audio or video stream.

    ``config`` may be a dataclass (e.g. common.config.VideoConfig) or a
    JSON-compatible mapping.  Detailed config validation belongs to the
    configuration/negotiation layer.
    """
    try:
        normalized_stream = (
            stream if isinstance(stream, StreamType) else StreamType(stream)
        )
    except (ValueError, TypeError):
        raise MessageError(f"Unsupported stream: {stream!r}") from None

    return ControlMessage.create(
        MessageType.CONFIGURE,
        {
            "stream": normalized_stream.value,
            "config": _json_compatible(config, field_name="config"),
        },
        request_id=request_id,
    )


def codec_switch_message(
    stream: StreamType | str,
    codec: str,
    *,
    payload_type: int | None = None,
    request_id: str | None = None,
) -> ControlMessage:
    """Build a codec-switch request.

    ``payload_type`` is optional because the receiver may already have a
    negotiated mapping.  When supplied, it must be a 7-bit RTP payload type.
    """
    try:
        normalized_stream = (
            stream if isinstance(stream, StreamType) else StreamType(stream)
        )
    except (ValueError, TypeError):
        raise MessageError(f"Unsupported stream: {stream!r}") from None

    if not isinstance(codec, str) or not codec.strip():
        raise MessageError("codec must be a non-empty string")

    payload: dict[str, Any] = {
        "stream": normalized_stream.value,
        "codec": codec.lower(),
    }

    if payload_type is not None:
        if (
            not isinstance(payload_type, int)
            or isinstance(payload_type, bool)
            or not 0 <= payload_type <= 127
        ):
            raise MessageError("payload_type must be an integer between 0 and 127")
        payload["payload_type"] = payload_type

    return ControlMessage.create(
        MessageType.CODEC_SWITCH,
        payload,
        request_id=request_id,
    )


def response_message(
    request: ControlMessage,
    *,
    accepted: bool,
    reason: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> ControlMessage:
    """Create an ACK/NACK response correlated with ``request``."""
    if not isinstance(request, ControlMessage):
        raise MessageError("request must be a ControlMessage")

    if not isinstance(accepted, bool):
        raise MessageError("accepted must be a bool")

    if reason is not None and not isinstance(reason, str):
        raise MessageError("reason must be a string or None")

    response_types = {
        MessageType.CONFIGURE: (
            MessageType.CONFIGURE_ACK,
            MessageType.CONFIGURE_NACK,
        ),
        MessageType.CODEC_SWITCH: (
            MessageType.CODEC_SWITCH_ACK,
            MessageType.CODEC_SWITCH_NACK,
        ),
    }

    if request.type not in response_types:
        raise MessageError(
            f"Message type {request.type.value!r} does not support ACK/NACK"
        )

    ack_type, nack_type = response_types[request.type]
    response_payload = dict(payload or {})
    if reason is not None:
        response_payload["reason"] = reason

    return ControlMessage.create(
        ack_type if accepted else nack_type,
        response_payload,
        request_id=request.request_id,
    )
