import pytest

from rtp_live_streaming.common.rtp_packet import (
    RTPPacket,
    RTPPacketError,
)


def make_packet(**overrides):
    values = {
        "payload_type": 96,
        "sequence_number": 100,
        "timestamp": 90000,
        "ssrc": 12345,
        "payload": b"Hello RTP",
    }
    values.update(overrides)
    return RTPPacket(**values)


def test_serialization_round_trip():
    packet = make_packet()

    decoded = RTPPacket.from_bytes(packet.to_bytes())

    assert decoded == packet


def test_header_size():
    packet = make_packet()

    assert len(packet.to_bytes()) == 12 + len(packet.payload)


@pytest.mark.parametrize("sequence", [0, 1, 65535])
def test_sequence_number_boundaries(sequence):
    packet = make_packet(sequence_number=sequence)

    decoded = RTPPacket.from_bytes(packet.to_bytes())

    assert decoded.sequence_number == sequence


@pytest.mark.parametrize("timestamp", [0, 1, 4294967295])
def test_timestamp_boundaries(timestamp):
    packet = make_packet(timestamp=timestamp)

    decoded = RTPPacket.from_bytes(packet.to_bytes())

    assert decoded.timestamp == timestamp


def test_marker_bit():
    packet = make_packet(marker=True)

    decoded = RTPPacket.from_bytes(packet.to_bytes())

    assert decoded.marker is True


def test_payload_integrity():
    payload = bytes(range(256))
    packet = make_packet(payload=payload)

    decoded = RTPPacket.from_bytes(packet.to_bytes())

    assert decoded.payload == payload


def test_csrc_serialization():
    packet = make_packet(csrcs=(100, 200, 300))

    decoded = RTPPacket.from_bytes(packet.to_bytes())

    assert decoded.csrcs == (100, 200, 300)


def test_extension_serialization():
    packet = make_packet(
        extension_profile=0xBEDE,
        extension_data=b"\x10\x20\x30\x40",
    )

    decoded = RTPPacket.from_bytes(packet.to_bytes())

    assert decoded.extension_profile == 0xBEDE
    assert decoded.extension_data == b"\x10\x20\x30\x40"


def test_padding_serialization():
    packet = make_packet(padding_size=4)

    decoded = RTPPacket.from_bytes(packet.to_bytes())

    assert decoded.padding_size == 4
    assert decoded.payload == packet.payload


@pytest.mark.parametrize(
    "kwargs",
    [
        {"payload_type": 128},
        {"sequence_number": 65536},
        {"timestamp": 4294967296},
        {"ssrc": -1},
        {"padding_size": 256},
        {"csrcs": tuple(range(16))},
    ],
)
def test_invalid_fields(kwargs):
    with pytest.raises(RTPPacketError):
        make_packet(**kwargs)


def test_packet_smaller_than_header():
    with pytest.raises(RTPPacketError):
        RTPPacket.from_bytes(b"\x80" * 11)


def test_invalid_rtp_version():
    data = bytearray(make_packet().to_bytes())
    data[0] = 0x40  # RTP version 1

    with pytest.raises(RTPPacketError):
        RTPPacket.from_bytes(bytes(data))


def test_invalid_padding():
    data = bytearray(make_packet().to_bytes())
    data[0] |= 0x20  # Set padding bit
    data[-1] = 0

    with pytest.raises(RTPPacketError):
        RTPPacket.from_bytes(bytes(data))
