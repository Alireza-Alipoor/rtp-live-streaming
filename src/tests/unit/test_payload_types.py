import pytest

from rtp_live_streaming.common.payload_types import (
    CODEC_TO_PAYLOAD_TYPE,
    Codec,
    PayloadTypeError,
    get_codec,
    get_codec_name,
    get_payload_type,
    is_registered_payload_type,
    is_valid_payload_type,
    validate_payload_type,
)


@pytest.mark.parametrize(
    ("codec", "expected"),
    [
        ("h264", 96),
        ("vp8", 97),
        ("opus", 111),
        ("pcmu", 0),
    ],
)
def test_get_payload_type(codec, expected):
    assert get_payload_type(codec) == expected


@pytest.mark.parametrize(
    ("payload_type", "expected"),
    [
        (96, Codec.H264),
        (97, Codec.VP8),
        (111, Codec.OPUS),
        (0, Codec.PCMU),
    ],
)
def test_get_codec(payload_type, expected):
    assert get_codec(payload_type) == expected


def test_codec_lookup_is_case_insensitive():
    assert get_payload_type("H264") == 96
    assert get_payload_type("OpUs") == 111


def test_get_codec_name():
    assert get_codec_name(96) == "h264"
    assert get_codec_name(0) == "pcmu"


@pytest.mark.parametrize("payload_type", [0, 1, 96, 111, 127])
def test_valid_payload_type(payload_type):
    assert is_valid_payload_type(payload_type)


@pytest.mark.parametrize("payload_type", [-1, 128, 256, 1.5, "96", True])
def test_invalid_payload_type(payload_type):
    assert not is_valid_payload_type(payload_type)


@pytest.mark.parametrize("payload_type", [96, 97, 111, 0])
def test_registered_payload_type(payload_type):
    assert is_registered_payload_type(payload_type)


@pytest.mark.parametrize("payload_type", [1, 95, 98, 110, 112, 127])
def test_unregistered_payload_type(payload_type):
    assert not is_registered_payload_type(payload_type)


def test_validate_payload_type_returns_value():
    assert validate_payload_type(96) == 96


@pytest.mark.parametrize("payload_type", [-1, 128, "96", True])
def test_validate_payload_type_raises(payload_type):
    with pytest.raises(PayloadTypeError):
        validate_payload_type(payload_type)


def test_unknown_codec_raises():
    with pytest.raises(PayloadTypeError):
        get_payload_type("av1")


def test_unregistered_payload_type_raises():
    with pytest.raises(PayloadTypeError):
        get_codec(100)


def test_payload_type_mapping_is_consistent():
    for codec, payload_type in CODEC_TO_PAYLOAD_TYPE.items():
        assert get_payload_type(codec) == int(payload_type)
        assert get_codec(int(payload_type)) == codec
