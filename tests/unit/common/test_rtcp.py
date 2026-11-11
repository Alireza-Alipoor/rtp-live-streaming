import struct

import pytest

from rtp_live_streaming.common.rtcp import (
    INT24_MAX,
    INT24_MIN,
    MAX_REPORT_BLOCKS,
    NTP_FRACTION_SCALE,
    NTP_UNIX_EPOCH_OFFSET,
    REPORT_BLOCK_SIZE,
    RTCP_PT_RR,
    RawRTCPPacket,
    ReceiverReport,
    ReportBlock,
    RTCPError,
    SenderReport,
    ntp_to_unix,
    parse_compound_rtcp,
    parse_rtcp_packet,
    serialize_compound_rtcp,
    unix_to_ntp,
)


def make_report_block(**overrides):
    values = {
        "ssrc": 1111,
        "fraction_lost": 10,
        "cumulative_lost": 20,
        "extended_highest_sequence_number": 300,
        "interarrival_jitter": 400,
        "last_sender_report": 500,
        "delay_since_last_sender_report": 600,
    }
    values.update(overrides)
    return ReportBlock(**values)


def make_sender_report(**overrides):
    values = {
        "ssrc": 12345,
        "ntp_seconds": 3_900_000_000,
        "ntp_fraction": 1_000_000,
        "rtp_timestamp": 90000,
        "sender_packet_count": 100,
        "sender_octet_count": 2000,
    }
    values.update(overrides)
    return SenderReport(**values)


def make_receiver_report(**overrides):
    values = {"ssrc": 54321}
    values.update(overrides)
    return ReceiverReport(**values)


def test_report_block_round_trip():
    block = make_report_block()
    assert ReportBlock.from_bytes(block.to_bytes()) == block


def test_report_block_serialized_size():
    assert len(make_report_block().to_bytes()) == REPORT_BLOCK_SIZE


def test_report_block_negative_cumulative_lost_round_trip():
    block = make_report_block(cumulative_lost=-42)
    assert ReportBlock.from_bytes(block.to_bytes()).cumulative_lost == -42


@pytest.mark.parametrize("value", [INT24_MIN, INT24_MAX, 0, -1, 1])
def test_report_block_cumulative_lost_boundaries(value):
    block = make_report_block(cumulative_lost=value)
    assert ReportBlock.from_bytes(block.to_bytes()).cumulative_lost == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("fraction_lost", -1),
        ("fraction_lost", 256),
        ("cumulative_lost", INT24_MAX + 1),
        ("cumulative_lost", INT24_MIN - 1),
        ("ssrc", -1),
        ("ssrc", 1 << 32),
        ("fraction_lost", True),
        ("ssrc", "1"),
    ],
)
def test_report_block_invalid_fields_raise(field, value):
    with pytest.raises(RTCPError):
        make_report_block(**{field: value})


def test_report_block_from_bytes_wrong_size_raises():
    with pytest.raises(RTCPError):
        ReportBlock.from_bytes(b"\x00" * (REPORT_BLOCK_SIZE - 1))


def test_sender_report_round_trip():
    packet = make_sender_report()
    assert SenderReport.from_bytes(packet.to_bytes()) == packet


def test_sender_report_with_report_blocks_round_trip():
    packet = make_sender_report(
        report_blocks=(make_report_block(), make_report_block(ssrc=2222))
    )
    decoded = SenderReport.from_bytes(packet.to_bytes())
    assert decoded == packet
    assert len(decoded.report_blocks) == 2


def test_sender_report_size():
    packet = make_sender_report()
    assert len(packet.to_bytes()) == 4 + 24


def test_sender_report_with_padding_round_trip():
    packet = make_sender_report(padding_size=4)
    decoded = SenderReport.from_bytes(packet.to_bytes())
    assert decoded == packet
    assert decoded.padding_size == 4


def test_sender_report_too_many_report_blocks_raises():
    blocks = tuple(make_report_block() for _ in range(MAX_REPORT_BLOCKS + 1))
    with pytest.raises(RTCPError):
        make_sender_report(report_blocks=blocks)


def test_sender_report_invalid_report_block_type_raises():
    with pytest.raises(RTCPError):
        make_sender_report(report_blocks=("not-a-block",))


@pytest.mark.parametrize("padding", [1, 2, 3, 5, 256, -4, True])
def test_sender_report_invalid_padding_raises(padding):
    with pytest.raises(RTCPError):
        make_sender_report(padding_size=padding)


def test_sender_report_from_bytes_wrong_type_raises():
    rr = make_receiver_report().to_bytes()
    with pytest.raises(RTCPError):
        SenderReport.from_bytes(rr)


def test_sender_report_from_bytes_bad_body_length_raises():
    packet = make_sender_report().to_bytes()
    bad = packet[:4] + packet[4:-4]
    with pytest.raises(RTCPError):
        SenderReport.from_bytes(bad)


def test_receiver_report_round_trip():
    packet = make_receiver_report()
    assert ReceiverReport.from_bytes(packet.to_bytes()) == packet


def test_receiver_report_with_report_blocks_round_trip():
    packet = make_receiver_report(
        report_blocks=(make_report_block(), make_report_block(ssrc=3333))
    )
    decoded = ReceiverReport.from_bytes(packet.to_bytes())
    assert decoded == packet
    assert len(decoded.report_blocks) == 2


def test_receiver_report_size():
    packet = make_receiver_report()
    assert len(packet.to_bytes()) == 4 + 4


def test_receiver_report_with_padding_round_trip():
    packet = make_receiver_report(padding_size=8)
    decoded = ReceiverReport.from_bytes(packet.to_bytes())
    assert decoded == packet
    assert decoded.padding_size == 8


def test_receiver_report_from_bytes_wrong_type_raises():
    sr = make_sender_report().to_bytes()
    with pytest.raises(RTCPError):
        ReceiverReport.from_bytes(sr)


def test_receiver_report_from_bytes_bad_body_length_raises():
    packet = make_receiver_report().to_bytes()
    bad = packet[:4] + packet[4:-4]
    with pytest.raises(RTCPError):
        ReceiverReport.from_bytes(bad)


def test_receiver_report_too_many_report_blocks_raises():
    blocks = tuple(make_report_block() for _ in range(MAX_REPORT_BLOCKS + 1))
    with pytest.raises(RTCPError):
        make_receiver_report(report_blocks=blocks)


def test_raw_rtcp_packet_round_trip():
    packet = RawRTCPPacket(
        packet_type=202,
        count=1,
        body=b"\x01\x02\x03\x04",
    )
    decoded = parse_rtcp_packet(packet.to_bytes())
    assert isinstance(decoded, RawRTCPPacket)
    assert decoded == packet


def test_raw_rtcp_packet_with_padding_round_trip():
    packet = RawRTCPPacket(
        packet_type=203,
        count=0,
        body=b"\x00\x00\x00\x00",
        padding_size=4,
    )
    decoded = parse_rtcp_packet(packet.to_bytes())
    assert isinstance(decoded, RawRTCPPacket)
    assert decoded.padding_size == 4
    assert decoded.body == b"\x00\x00\x00\x00"


@pytest.mark.parametrize("pt", [191, 224, 0, 255])
def test_raw_rtcp_packet_invalid_type_raises(pt):
    with pytest.raises(RTCPError):
        RawRTCPPacket(packet_type=pt, count=0, body=b"")


def test_raw_rtcp_packet_unaligned_body_raises():
    with pytest.raises(RTCPError):
        RawRTCPPacket(packet_type=200, count=0, body=b"\x00\x00\x00")


def test_raw_rtcp_packet_non_bytes_body_raises():
    with pytest.raises(RTCPError):
        RawRTCPPacket(packet_type=200, count=0, body="abc")


def test_parse_rtcp_packet_sender_report():
    packet = make_sender_report()
    decoded = parse_rtcp_packet(packet.to_bytes())
    assert isinstance(decoded, SenderReport)
    assert decoded == packet


def test_parse_rtcp_packet_receiver_report():
    packet = make_receiver_report()
    decoded = parse_rtcp_packet(packet.to_bytes())
    assert isinstance(decoded, ReceiverReport)
    assert decoded == packet


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00\x00",
        b"\x80\xc8\x00\x00\x00",  # not 4-byte aligned
    ],
)
def test_parse_rtcp_packet_invalid_data_raises(data):
    with pytest.raises(RTCPError):
        parse_rtcp_packet(data)


def test_parse_rtcp_packet_bad_version_raises():
    packet = bytearray(make_sender_report().to_bytes())
    packet[0] = (1 << 6) | (packet[0] & 0x3F)
    with pytest.raises(RTCPError):
        parse_rtcp_packet(bytes(packet))


def test_parse_rtcp_packet_length_mismatch_raises():
    packet = make_sender_report().to_bytes()
    with pytest.raises(RTCPError):
        parse_rtcp_packet(packet + b"\x00\x00\x00\x00")


def test_parse_compound_rtcp():
    sr = make_sender_report()
    rr = make_receiver_report()
    data = serialize_compound_rtcp([sr, rr])
    decoded = parse_compound_rtcp(data)
    assert decoded == (sr, rr)


def test_parse_compound_rtcp_preserves_order():
    packets = (
        make_sender_report(),
        RawRTCPPacket(packet_type=202, count=0, body=b"\x00\x00\x00\x00"),
        make_receiver_report(),
    )
    decoded = parse_compound_rtcp(serialize_compound_rtcp(packets))
    assert decoded == packets


@pytest.mark.parametrize("data", [b"", b"\x80\xc8\x00"])
def test_parse_compound_rtcp_invalid_data_raises(data):
    with pytest.raises(RTCPError):
        parse_compound_rtcp(data)


def test_parse_compound_rtcp_truncated_raises():
    data = serialize_compound_rtcp([make_sender_report()])
    with pytest.raises(RTCPError):
        parse_compound_rtcp(data[:-4])


def test_serialize_compound_rtcp_empty_raises():
    with pytest.raises(RTCPError):
        serialize_compound_rtcp([])


def test_serialize_compound_rtcp_unsupported_object_raises():
    with pytest.raises(RTCPError):
        serialize_compound_rtcp([object()])


def test_unix_to_ntp_epoch_offset():
    seconds, fraction = unix_to_ntp(0)
    assert seconds == NTP_UNIX_EPOCH_OFFSET
    assert fraction == 0


def test_unix_to_ntp_fraction():
    seconds, fraction = unix_to_ntp(0.5)
    assert seconds == NTP_UNIX_EPOCH_OFFSET
    assert fraction == NTP_FRACTION_SCALE // 2


def test_unix_to_ntp_fraction_overflow_carry():
    seconds, fraction = unix_to_ntp(0.99999999999)
    assert fraction == 0
    assert seconds == NTP_UNIX_EPOCH_OFFSET + 1


@pytest.mark.parametrize("bad", [-0.1, True, "1", None])
def test_unix_to_ntp_invalid_raises(bad):
    with pytest.raises(RTCPError):
        unix_to_ntp(bad)


def test_ntp_to_unix_round_trip():
    for value in [0.0, 1.5, 1234567.25, 1700000000.5]:
        seconds, fraction = unix_to_ntp(value)
        assert ntp_to_unix(seconds, fraction) == pytest.approx(value, abs=1e-6)


@pytest.mark.parametrize(
    "seconds,fraction",
    [(-1, 0), (1 << 32, 0), (0, 1 << 32), (0, -1)],
)
def test_ntp_to_unix_invalid_raises(seconds, fraction):
    with pytest.raises(RTCPError):
        ntp_to_unix(seconds, fraction)


def test_rtcp_header_bits():
    packet = make_receiver_report(
        report_blocks=(make_report_block(), make_report_block()),
    ).to_bytes()
    first_byte = packet[0]
    assert first_byte >> 6 == 2
    assert first_byte & 0x1F == 2
    assert first_byte & 0x20 == 0
    assert packet[1] == RTCP_PT_RR


def test_rtcp_padding_flag_set():
    packet = make_receiver_report(padding_size=4).to_bytes()
    assert packet[0] & 0x20
    assert packet[-1] == 4


def test_padding_flag_without_padding_raises():
    packet = bytearray(make_receiver_report().to_bytes())
    packet[0] |= 0x20
    with pytest.raises(RTCPError):
        parse_rtcp_packet(bytes(packet))


def test_padding_zero_size_raises():
    body = struct.pack("!I", 54321) + b"\x00"
    length_words = 4 // 4
    packet = struct.pack("!BBH", (2 << 6) | 0x20, RTCP_PT_RR, length_words) + body
    with pytest.raises(RTCPError):
        parse_rtcp_packet(packet)
