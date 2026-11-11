import pytest

from rtp_live_streaming.sender.packetization.h264 import (
    FU_A_HEADER_SIZE,
    FU_END_BIT,
    FU_START_BIT,
    NAL_HEADER_SIZE,
    NAL_TYPE_FU_A,
    START_CODE_3,
    START_CODE_4,
    H264PacketizationError,
    H264RTPPayload,
    extract_nal_units,
    packetize_access_unit,
    packetize_nal_unit,
)


def make_nal_unit(nal_type=5, nri=3, payload=b"\xaa", forbidden=False):
    header = (0x80 if forbidden else 0x00) | ((nri & 0x03) << 5) | (nal_type & 0x1F)
    return bytes((header,)) + payload


def test_h264_rtp_payload_default_marker_false():
    p = H264RTPPayload(payload=b"\x00")
    assert p.marker is False


def test_h264_rtp_payload_equality():
    assert H264RTPPayload(b"\x00") == H264RTPPayload(b"\x00", False)


def test_extract_nal_units_single_raw_nal():
    nal = make_nal_unit()
    assert extract_nal_units(nal) == (nal,)


def test_extract_nal_units_annex_b_single_nal():
    nal = make_nal_unit()
    assert extract_nal_units(START_CODE_4 + nal) == (nal,)


def test_extract_nal_units_three_byte_start_code():
    nal = make_nal_unit()
    assert extract_nal_units(START_CODE_3 + nal) == (nal,)


def test_extract_nal_units_multiple_nals():
    n1 = make_nal_unit(nal_type=7, payload=b"\x01")
    n2 = make_nal_unit(nal_type=8, payload=b"\x02")
    data = START_CODE_4 + n1 + START_CODE_3 + n2
    assert extract_nal_units(data) == (n1, n2)


def test_extract_nal_units_prefers_four_byte_start_code():
    nal = b"\x65\xaa"
    assert extract_nal_units(START_CODE_4 + nal) == (nal,)


def test_extract_nal_units_leading_bytes_ignored():
    nal = b"\x65\xaa"
    data = b"\xde\xad\xbe\xef" + START_CODE_4 + nal
    assert extract_nal_units(data) == (nal,)


def test_extract_nal_units_trailing_zeros_stripped():
    nal = b"\x65\xaa\xbb"
    data = START_CODE_4 + nal + b"\x00\x00\x00"
    assert extract_nal_units(data) == (nal,)


def test_extract_nal_units_empty_nal_raises():
    with pytest.raises(H264PacketizationError):
        extract_nal_units(START_CODE_4 + START_CODE_4)


@pytest.mark.parametrize("bad", ["abc", 123, None, b""])
def test_extract_nal_units_invalid_data_raises(bad):
    with pytest.raises(H264PacketizationError):
        extract_nal_units(bad)


@pytest.mark.parametrize(
    "bad_nal",
    [
        b"\x80\x00",
        b"\x00\x00",
        b"\x18\x00",
        b"\x1c\x00",
    ],
)
def test_extract_nal_units_invalid_raw_nal_raises(bad_nal):
    with pytest.raises(H264PacketizationError):
        extract_nal_units(bad_nal)


@pytest.mark.parametrize(
    "bad_nal",
    [
        b"\x80\x00",
        b"\x1c\x00",
    ],
)
def test_extract_nal_units_invalid_annex_b_nal_raises(bad_nal):
    with pytest.raises(H264PacketizationError):
        extract_nal_units(START_CODE_4 + bad_nal)


def test_packetize_nal_unit_fits_single_packet():
    nal = make_nal_unit(payload=b"\xaa" * 10)
    packets = packetize_nal_unit(nal, max_payload_size=100)
    assert len(packets) == 1
    assert packets[0].payload == nal


def test_packetize_nal_unit_marker_default_true():
    packets = packetize_nal_unit(make_nal_unit(payload=b"\xaa"), max_payload_size=100)
    assert packets[0].marker is True


def test_packetize_nal_unit_marker_false():
    packets = packetize_nal_unit(
        make_nal_unit(payload=b"\xaa"), max_payload_size=100, marker=False
    )
    assert packets[0].marker is False


@pytest.mark.parametrize("bad_marker", [0, 1, "yes", None])
def test_packetize_nal_unit_invalid_marker_raises(bad_marker):
    with pytest.raises(H264PacketizationError):
        packetize_nal_unit(make_nal_unit(), max_payload_size=100, marker=bad_marker)


def test_packetize_nal_unit_fragments_structure():
    nal = b"\x65" + b"\xaa" * 100
    packets = packetize_nal_unit(nal, max_payload_size=30)

    assert len(packets) == 4
    assert packets[0].payload[0] == 0x7C
    assert packets[0].payload[1] == (FU_START_BIT | 5)

    for p in packets[1:-1]:
        assert p.payload[0] == 0x7C
        assert p.payload[1] == 5

    assert packets[-1].payload[0] == 0x7C
    assert packets[-1].payload[1] == (FU_END_BIT | 5)


def test_packetize_nal_unit_fragment_sizes():
    nal = b"\x65" + b"\xaa" * 100
    packets = packetize_nal_unit(nal, max_payload_size=30)
    assert [len(p.payload) for p in packets] == [30, 30, 30, 18]


def test_packetize_nal_unit_fu_indicator_preserves_nri():
    nal = b"\x41" + b"\xaa" * 100
    packets = packetize_nal_unit(nal, max_payload_size=10)
    for p in packets:
        assert p.payload[0] == 0x5C


def test_packetize_nal_unit_reassembles_to_original():
    payload = bytes(range(256)) * 2
    nal = b"\x65" + payload
    packets = packetize_nal_unit(nal, max_payload_size=100)

    fu_indicator = packets[0].payload[0]
    fu_header = packets[0].payload[1]
    reconstructed_header = (fu_indicator & 0xE0) | (fu_header & 0x1F)
    reassembled = b"".join(p.payload[2:] for p in packets)

    assert bytes((reconstructed_header,)) + reassembled == nal


def test_packetize_nal_unit_exact_fit_single_packet():
    nal = b"\x65" + b"\xaa" * 10
    packets = packetize_nal_unit(nal, max_payload_size=len(nal))
    assert len(packets) == 1
    assert packets[0].payload == nal


def test_packetize_nal_unit_one_byte_over_uses_fu_a():
    nal = b"\x65" + b"\xaa" * 10
    packets = packetize_nal_unit(nal, max_payload_size=len(nal) - 1)
    assert len(packets) == 2
    assert all(p.payload[0] == 0x7C for p in packets)


@pytest.mark.parametrize("bad_size", [0, -1, 1, 2, 1.5, True, "10"])
def test_packetize_nal_unit_invalid_max_payload_raises(bad_size):
    with pytest.raises(H264PacketizationError):
        packetize_nal_unit(b"\x65" + b"\xaa" * 100, max_payload_size=bad_size)


@pytest.mark.parametrize(
    "bad_nal",
    [b"", b"\x80\x00", b"\x1c\x00", b"\x00\x00", "abc"],
)
def test_packetize_nal_unit_invalid_nal_raises(bad_nal):
    with pytest.raises(H264PacketizationError):
        packetize_nal_unit(bad_nal, max_payload_size=100)


def test_packetize_nal_unit_last_fragment_only_has_marker():
    nal = b"\x65" + b"\xaa" * 200
    packets = packetize_nal_unit(nal, max_payload_size=50)
    assert len(packets) > 1
    assert all(not p.marker for p in packets[:-1])
    assert packets[-1].marker is True


def test_packetize_access_unit_from_annex_b():
    n1 = make_nal_unit(nal_type=7, payload=b"\xaa" * 10)
    n2 = make_nal_unit(nal_type=8, payload=b"\xbb" * 10)
    data = START_CODE_4 + n1 + START_CODE_3 + n2

    packets = packetize_access_unit(data, max_payload_size=1000)

    assert len(packets) == 2
    assert packets[0].payload == n1
    assert packets[1].payload == n2


def test_packetize_access_unit_from_iterable():
    n1 = make_nal_unit(payload=b"\xaa" * 10)
    n2 = make_nal_unit(payload=b"\xbb" * 10)

    packets = packetize_access_unit([n1, n2], max_payload_size=1000)

    assert len(packets) == 2
    assert packets[0].payload == n1
    assert packets[1].payload == n2


def test_packetize_access_unit_from_tuple():
    n1 = make_nal_unit()
    packets = packetize_access_unit((n1,), max_payload_size=1000)
    assert len(packets) == 1
    assert packets[0].marker is True


def test_packetize_access_unit_marker_only_on_last():
    n1 = b"\x65" + b"\xaa" * 100
    n2 = b"\x65" + b"\xbb" * 100

    packets = packetize_access_unit([n1, n2], max_payload_size=30)

    assert len(packets) == 8
    assert [p.marker for p in packets] == [False] * 7 + [True]


def test_packetize_access_unit_single_nal_marker_only_on_last_fragment():
    nal = b"\x65" + b"\xaa" * 200
    packets = packetize_access_unit([nal], max_payload_size=50)
    assert len(packets) > 1
    assert all(not p.marker for p in packets[:-1])
    assert packets[-1].marker is True


def test_packetize_access_unit_empty_bytes_raises():
    with pytest.raises(H264PacketizationError):
        packetize_access_unit(b"", max_payload_size=1000)


def test_packetize_access_unit_empty_iterable_raises():
    with pytest.raises(H264PacketizationError):
        packetize_access_unit([], max_payload_size=1000)


@pytest.mark.parametrize("bad", [42, 1.5, None])
def test_packetize_access_unit_non_iterable_raises(bad):
    with pytest.raises(H264PacketizationError):
        packetize_access_unit(bad, max_payload_size=1000)


def test_packetize_access_unit_invalid_nal_raises():
    with pytest.raises(H264PacketizationError):
        packetize_access_unit([b"\x1c\x00"], max_payload_size=1000)


def test_packetize_access_unit_fu_a_header_constant():
    assert FU_A_HEADER_SIZE == 2
    assert NAL_HEADER_SIZE == 1
    assert NAL_TYPE_FU_A == 28
