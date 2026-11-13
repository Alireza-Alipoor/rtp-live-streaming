"""Configuration validation and codec-transition coordination.

This module implements application-level negotiation only.  It does not
perform socket I/O, reconfigure encoders/decoders, or manipulate RTP packets.
The control server/client are responsible for exchanging messages; sender and
receiver components are responsible for applying committed configurations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any
from uuid import uuid4

from rtp_live_streaming.common.config import (
    AudioConfig,
    ConfigError,
    VideoConfig,
)
from rtp_live_streaming.common.payload_types import (
    Codec,
    PayloadTypeError,
    get_payload_type,
    validate_payload_type,
)

from .messages import StreamType


class NegotiationError(ValueError):
    """Raised when a configuration or codec transition cannot be negotiated."""


class TransitionState(StrEnum):
    """Lifecycle states for a coordinated codec change."""

    PREPARING = "preparing"
    READY = "ready"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class CodecTransition:
    """Snapshot of a codec-switch transaction."""

    request_id: str
    stream: StreamType
    old_codec: Codec
    new_codec: Codec
    payload_type: int
    state: TransitionState = TransitionState.PREPARING


class NegotiationManager:
    """Validate stream configurations and coordinate codec changes.

    The manager maintains the currently negotiated audio/video configuration.
    A codec transition is deliberately split into three steps:

    1. ``prepare_transition`` validates the request and records PREPARING.
    2. The control server asks the receiver to prepare its decoder, then calls
       ``mark_ready`` after the receiver confirms it is ready.
    3. The sender/receiver coordination layer calls ``commit_transition`` at
       the agreed switch point, or ``abort_transition`` if preparation fails.

    Only one codec transition may be pending per media stream. Audio and video
    transitions can be pending independently.
    """

    def __init__(
        self,
        *,
        video_config: VideoConfig | None = None,
        audio_config: AudioConfig | None = None,
    ) -> None:
        self._configs: dict[StreamType, VideoConfig | AudioConfig] = {
            StreamType.VIDEO: video_config or VideoConfig(),
            StreamType.AUDIO: audio_config or AudioConfig(),
        }
        self._pending: dict[str, CodecTransition] = {}

    def get_current_config(self, stream: StreamType | str) -> VideoConfig | AudioConfig:
        """Return the currently negotiated configuration for a stream."""
        normalized_stream = self._normalize_stream(stream)
        return self._configs[normalized_stream]

    def validate_config(
        self,
        stream: StreamType | str,
        config: VideoConfig | AudioConfig | Mapping[str, Any],
    ) -> VideoConfig | AudioConfig:
        """Validate and normalize a proposed audio/video configuration.

        A mapping is converted to the corresponding shared config dataclass,
        so all project constraints (supported codecs, bitrates, dimensions,
        frame rates and audio channel rules) remain centralized in common.config.
        This method does not modify the active configuration.
        """
        normalized_stream = self._normalize_stream(stream)
        config_class = (
            VideoConfig if normalized_stream is StreamType.VIDEO else AudioConfig
        )

        if isinstance(config, config_class):
            return config

        if not isinstance(config, Mapping):
            raise NegotiationError(
                f"{normalized_stream.value} config must be a "
                f"{config_class.__name__} or a mapping"
            )

        try:
            return config_class(**dict(config))
        except (ConfigError, TypeError) as exc:
            raise NegotiationError(
                f"Invalid {normalized_stream.value} configuration: {exc}"
            ) from exc

    def negotiate_config(
        self,
        stream: StreamType | str,
        config: VideoConfig | AudioConfig | Mapping[str, Any],
    ) -> VideoConfig | AudioConfig:
        """Validate and record a new stream configuration.

        The caller should send CONFIGURE_ACK only after this succeeds. Actual
        encoder/decoder reconfiguration remains the responsibility of the
        media pipeline. If applying the media configuration fails, the caller
        should restore the previous config or coordinate a compensating update.
        """
        normalized_stream = self._normalize_stream(stream)
        validated = self.validate_config(normalized_stream, config)
        self._configs[normalized_stream] = validated
        return validated

    def prepare_transition(
        self,
        stream: StreamType | str,
        codec: Codec | str,
        *,
        payload_type: int | None = None,
        request_id: str | None = None,
    ) -> CodecTransition:
        """Validate and register a codec-switch request in PREPARING state.

        The project has a fixed codec-to-payload-type mapping in
        common.payload_types. If ``payload_type`` is supplied, it must match
        that mapping. Dynamic RTP payload types still need to be communicated
        to the peer as part of the control protocol.
        """
        normalized_stream = self._normalize_stream(stream)
        normalized_codec = self._normalize_codec(codec)
        self._validate_codec_for_stream(normalized_stream, normalized_codec)

        expected_payload_type = get_payload_type(normalized_codec)
        if payload_type is None:
            normalized_payload_type = expected_payload_type
        else:
            try:
                validate_payload_type(payload_type)
            except PayloadTypeError as exc:
                raise NegotiationError(str(exc)) from exc
            if payload_type != expected_payload_type:
                raise NegotiationError(
                    f"Payload type {payload_type} does not match the project's "
                    f"mapping for {normalized_codec.value} "
                    f"({expected_payload_type})"
                )
            normalized_payload_type = payload_type

        if request_id is None:
            normalized_request_id = str(uuid4())
        elif isinstance(request_id, str) and request_id.strip():
            normalized_request_id = request_id
        else:
            raise NegotiationError("request_id must be a non-empty string")

        if normalized_request_id in self._pending:
            raise NegotiationError(
                f"A transition with request_id {normalized_request_id!r} "
                "is already pending"
            )

        for pending in self._pending.values():
            if pending.stream is normalized_stream and pending.state in (
                TransitionState.PREPARING,
                TransitionState.READY,
            ):
                raise NegotiationError(
                    f"A codec transition for {normalized_stream.value} "
                    "is already pending"
                )

        current_config = self._configs[normalized_stream]
        old_codec = current_config.codec
        if normalized_codec == old_codec:
            raise NegotiationError(
                f"{normalized_stream.value} is already using {normalized_codec.value}"
            )

        transition = CodecTransition(
            request_id=normalized_request_id,
            stream=normalized_stream,
            old_codec=old_codec,
            new_codec=normalized_codec,
            payload_type=normalized_payload_type,
        )
        self._pending[normalized_request_id] = transition
        return transition

    def mark_ready(self, request_id: str) -> CodecTransition:
        """Mark a prepared transition READY after the receiver is prepared."""
        transition = self._get_pending_transition(request_id)
        if transition.state is not TransitionState.PREPARING:
            raise NegotiationError(
                f"Cannot mark transition {request_id!r} ready from "
                f"state {transition.state.value!r}"
            )

        updated = replace(transition, state=TransitionState.READY)
        self._pending[request_id] = updated
        return updated

    def commit_transition(self, request_id: str) -> CodecTransition:
        """Commit a READY codec transition and update the active config.

        The returned transition is COMMITTED. The caller must still apply the
        corresponding encoder/decoder changes at the negotiated media
        boundary; this class only updates the control-plane state.
        """
        transition = self._get_pending_transition(request_id)
        if transition.state is not TransitionState.READY:
            raise NegotiationError(
                f"Cannot commit transition {request_id!r}: "
                "receiver has not marked it ready"
            )

        current = self._configs[transition.stream]
        try:
            if transition.stream is StreamType.VIDEO:
                updated_config = replace(current, codec=transition.new_codec)
            elif transition.new_codec is Codec.PCMU:
                # PCMU is fixed at 64 kbps and mono in this project.
                updated_config = replace(
                    current,
                    codec=Codec.PCMU,
                    bitrate_kbps=64,
                    channels=1,
                )
            else:
                # When moving from PCMU to Opus, retain a valid project Opus
                # bitrate and mono channel count.
                updated_config = replace(
                    current,
                    codec=Codec.OPUS,
                    bitrate_kbps=(
                        current.bitrate_kbps
                        if current.bitrate_kbps in (32, 64, 128)
                        else 64
                    ),
                )
        except ConfigError as exc:
            raise NegotiationError(f"Could not apply negotiated codec: {exc}") from exc

        self._configs[transition.stream] = updated_config
        committed = replace(transition, state=TransitionState.COMMITTED)
        self._pending.pop(request_id)
        return committed

    def abort_transition(
        self,
        request_id: str,
        *,
        reason: str | None = None,
    ) -> CodecTransition:
        """Abort a pending transition without changing the active config.

        ``reason`` is accepted for caller-side logging/API symmetry; it is not
        stored in the transition object.
        """
        if reason is not None and not isinstance(reason, str):
            raise NegotiationError("reason must be a string or None")

        transition = self._get_pending_transition(request_id)
        aborted = replace(transition, state=TransitionState.ABORTED)
        self._pending.pop(request_id)
        return aborted

    def get_pending_transition(self, request_id: str) -> CodecTransition | None:
        """Return a pending transition, or None if the ID is not pending."""
        if not isinstance(request_id, str):
            return None
        return self._pending.get(request_id)

    def _get_pending_transition(self, request_id: str) -> CodecTransition:
        if not isinstance(request_id, str) or not request_id:
            raise NegotiationError("request_id must be a non-empty string")

        try:
            return self._pending[request_id]
        except KeyError:
            raise NegotiationError(
                f"No pending codec transition for request_id {request_id!r}"
            ) from None

    @staticmethod
    def _normalize_stream(stream: StreamType | str) -> StreamType:
        try:
            return stream if isinstance(stream, StreamType) else StreamType(stream)
        except (ValueError, TypeError):
            raise NegotiationError(f"Unsupported stream: {stream!r}") from None

    @staticmethod
    def _normalize_codec(codec: Codec | str) -> Codec:
        try:
            return codec if isinstance(codec, Codec) else Codec(codec.lower())
        except (ValueError, AttributeError):
            raise NegotiationError(f"Unsupported codec: {codec!r}") from None

    @staticmethod
    def _validate_codec_for_stream(stream: StreamType, codec: Codec) -> None:
        allowed = (
            (Codec.H264, Codec.VP8)
            if stream is StreamType.VIDEO
            else (Codec.OPUS, Codec.PCMU)
        )
        if codec not in allowed:
            allowed_names = ", ".join(item.value for item in allowed)
            raise NegotiationError(
                f"Codec {codec.value!r} is not valid for {stream.value}; "
                f"supported codecs: {allowed_names}"
            )
