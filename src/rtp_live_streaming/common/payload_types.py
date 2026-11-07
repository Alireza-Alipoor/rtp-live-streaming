"""RTP payload type assignments and lookup helpers.

Static payload types follow the RTP/AVP profile where applicable.
Dynamic payload types are project defaults and must be negotiated
between the sender and receiver.

Project defaults:
    PCMU:  0   (static)
    H264:  96  (dynamic)
    VP8:   97  (dynamic)
    OPUS:  111 (dynamic)
"""

from enum import IntEnum, StrEnum


class PayloadType(IntEnum):
    """RTP payload type numbers used by this project."""

    PCMU = 0
    H264 = 96
    VP8 = 97
    OPUS = 111


class Codec(StrEnum):
    """Supported media codecs."""

    H264 = "h264"
    VP8 = "vp8"
    OPUS = "opus"
    PCMU = "pcmu"


# Codec -> RTP payload type
CODEC_TO_PAYLOAD_TYPE: dict[Codec, PayloadType] = {
    Codec.H264: PayloadType.H264,
    Codec.VP8: PayloadType.VP8,
    Codec.OPUS: PayloadType.OPUS,
    Codec.PCMU: PayloadType.PCMU,
}

# RTP payload type -> codec
PAYLOAD_TYPE_TO_CODEC: dict[int, Codec] = {
    int(payload_type): codec for codec, payload_type in CODEC_TO_PAYLOAD_TYPE.items()
}


class PayloadTypeError(ValueError):
    """Raised when an RTP payload type is invalid or unsupported."""


def is_valid_payload_type(payload_type: object) -> bool:
    """Return whether a value is a valid 7-bit RTP payload type.

    This checks the RTP numeric range, not whether the value is
    assigned to a codec in this project.
    """
    return (
        isinstance(payload_type, int)
        and not isinstance(payload_type, bool)
        and 0 <= payload_type <= 127
    )


def validate_payload_type(payload_type: int) -> int:
    """Validate and return an RTP payload type number.

    Raises:
        PayloadTypeError: If the value is not an integer in 0..127.
    """
    if not is_valid_payload_type(payload_type):
        raise PayloadTypeError(
            f"Invalid RTP payload type: {payload_type!r}. "
            "Expected an integer between 0 and 127."
        )

    return payload_type


def get_payload_type(codec: Codec | str) -> int:
    """Return the configured RTP payload type for a codec.

    Codec names are case-insensitive.

    Raises:
        PayloadTypeError: If the codec is not supported.
    """
    try:
        normalized_codec = Codec(codec.lower())
    except (ValueError, AttributeError):
        raise PayloadTypeError(f"Unsupported codec: {codec!r}") from None

    return int(CODEC_TO_PAYLOAD_TYPE[normalized_codec])


def get_codec(payload_type: int) -> Codec:
    """Return the codec associated with a registered payload type.

    Raises:
        PayloadTypeError: If the payload type is invalid or unassigned.
    """
    validate_payload_type(payload_type)

    try:
        return PAYLOAD_TYPE_TO_CODEC[payload_type]
    except KeyError:
        raise PayloadTypeError(
            f"No codec registered for RTP payload type {payload_type}."
        ) from None


def get_codec_name(payload_type: int) -> str:
    """Return the codec name associated with a payload type."""
    return get_codec(payload_type).value


def is_registered_payload_type(payload_type: object) -> bool:
    """Return whether a valid payload type is registered in this project."""
    return is_valid_payload_type(payload_type) and payload_type in PAYLOAD_TYPE_TO_CODEC
