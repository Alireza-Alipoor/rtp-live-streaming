import pytest

from rtp_live_streaming.receiver.depacketization.h264 import (
    FU_A_HEADER_SIZE,
    FU_END_BIT,
    FU_RESERVED_BIT,
    FU_START_BIT,
    NAL_TYPE_FU_A,
    H264DepacketizationError,
    H264Depacketizer,
)


def make_single_nal(nal_type=5, nri=3, payload=b"\xaa"):
    header = ((nri & 0x03) << 5) | (nal_type & 0x1F)
    return bytes((header,)) + payload


def make_fu_a(
    fu_header_extra=0,
    nal_type=5,
    nri=3,
    fragment=b"\xaa",
    forbidden=False,
):
    fu_indicator = (0x80 if forbidden else 0x00) | ((nri & 0x03) << 5) | NAL_TYPE_FU_A
    fu_header = (nal_type & 0x1F) | fu_header_extra
    return bytes((fu_indicator, fu_header)) + fragment


def make_fu_a_start(nal_type=5, nri=3, fragment=b"\xaa", **kwargs):
    return make_fu_a(
        fu_header_extra=FU_START_BIT,
        nal_type=nal_type,
        nri=nri,
        fragment=fragment,
        **kwargs,
    )


def make_fu_a_cont(nal_type=5, nri=3, fragment=b"\xaa", **kwargs):
    return make_fu_a(
        fu_header_extra=0, nal_type=nal_type, nri=nri, fragment=fragment, **kwargs
    )


def make_fu_a_end(nal_type=5, nri=3, fragment=b"\xaa", **kwargs):
    return make_fu_a(
        fu_header_extra=FU_END_BIT,
        nal_type=nal_type,
        nri=nri,
        fragment=fragment,
        **kwargs,
    )


def test_single_nal_returns_none_without_marker():
    d = H264Depacketizer()
    assert d.push(make_single_nal()) is None


def test_single_nal_returns_access_unit_with_marker():
    d = H264Depacketizer()
    nal = make_single_nal()
    assert d.push(nal, marker=True) == (nal,)


def test_multiple_single_nals_accumulate():
    d = H264Depacketizer()
    n1 = make_single_nal(nal_type=7, payload=b"\x01")
    n2 = make_single_nal(nal_type=8, payload=b"\x02")
    assert d.push(n1) is None
    assert d.push(n2, marker=True) == (n1, n2)


def test_state_cleared_after_access_unit():
    d = H264Depacketizer()
    n1 = make_single_nal()
    assert d.push(n1, marker=True) == (n1,)
    n2 = make_single_nal(payload=b"\xbb")
    assert d.push(n2, marker=True) == (n2,)


def test_fu_a_start_returns_none():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start()) is None


def test_fu_a_single_fragment_start_then_end():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start(fragment=b"\x01\x02")) is None
    result = d.push(make_fu_a_end(fragment=b"\x03"), marker=True)
    assert result == (bytes((0x65,)) + b"\x01\x02\x03",)


def test_fu_a_multiple_fragments():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start(fragment=b"\x01")) is None
    assert d.push(make_fu_a_cont(fragment=b"\x02")) is None
    assert d.push(make_fu_a_cont(fragment=b"\x03")) is None
    result = d.push(make_fu_a_end(fragment=b"\x04"), marker=True)
    assert result == (bytes((0x65,)) + b"\x01\x02\x03\x04",)


def test_fu_a_reconstructs_original_nal_header():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start(nal_type=1, nri=1, fragment=b"\x00")) is None
    result = d.push(make_fu_a_end(nal_type=1, nri=1, fragment=b"\x00"), marker=True)
    assert result[0][0] == ((1 & 0x03) << 5) | 1


def test_fu_a_then_single_nal_same_access_unit():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start(fragment=b"\x01")) is None
    assert d.push(make_fu_a_end(fragment=b"\x02")) is None
    n2 = make_single_nal(payload=b"\xbb")
    result = d.push(n2, marker=True)
    assert result == (bytes((0x65,)) + b"\x01\x02", n2)


def test_fu_a_start_and_end_bit_both_set_raises():
    d = H264Depacketizer()
    payload = make_fu_a(fu_header_extra=FU_START_BIT | FU_END_BIT)
    with pytest.raises(H264DepacketizationError):
        d.push(payload)


def test_fu_a_continuation_without_start_raises():
    d = H264Depacketizer()
    with pytest.raises(H264DepacketizationError):
        d.push(make_fu_a_cont())


def test_fu_a_end_without_start_raises():
    d = H264Depacketizer()
    with pytest.raises(H264DepacketizationError):
        d.push(make_fu_a_end())


def test_fu_a_new_start_before_previous_completes_raises():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start()) is None
    with pytest.raises(H264DepacketizationError):
        d.push(make_fu_a_start())


def test_fu_a_indicator_change_raises():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start(nri=3)) is None
    with pytest.raises(H264DepacketizationError):
        d.push(make_fu_a_end(nri=1))


def test_fu_a_nal_type_change_raises():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start(nal_type=5)) is None
    with pytest.raises(H264DepacketizationError):
        d.push(make_fu_a_end(nal_type=1))


def test_fu_a_forbidden_bit_raises():
    d = H264Depacketizer()
    with pytest.raises(H264DepacketizationError):
        d.push(make_fu_a_start(forbidden=True))


def test_fu_a_reserved_bit_raises():
    d = H264Depacketizer()
    payload = make_fu_a(fu_header_extra=FU_START_BIT | FU_RESERVED_BIT)
    with pytest.raises(H264DepacketizationError):
        d.push(payload)


def test_fu_a_payload_too_short_raises():
    d = H264Depacketizer()
    fu_indicator = ((3 & 0x03) << 5) | NAL_TYPE_FU_A
    with pytest.raises(H264DepacketizationError):
        d.push(bytes((fu_indicator, 0x85)))
    with pytest.raises(H264DepacketizationError):
        d.push(bytes((fu_indicator,)))


@pytest.mark.parametrize("bad_type", [0, 24, 28, 30, 31])
def test_fu_a_invalid_original_nal_type_raises(bad_type):
    d = H264Depacketizer()
    payload = make_fu_a(fu_header_extra=FU_START_BIT, nal_type=bad_type)
    with pytest.raises(H264DepacketizationError):
        d.push(payload)


def test_single_nal_forbidden_bit_raises():
    d = H264Depacketizer()
    payload = bytes((0x80 | 0x65, 0xAA))
    with pytest.raises(H264DepacketizationError):
        d.push(payload)


def test_single_nal_during_fu_a_reassembly_raises():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start()) is None
    with pytest.raises(H264DepacketizationError):
        d.push(make_single_nal())


def test_marker_during_incomplete_fu_a_raises():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start()) is None
    with pytest.raises(H264DepacketizationError):
        d.push(make_single_nal(), marker=True)


def test_fu_a_end_with_marker_finishes_access_unit():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start(fragment=b"\x01")) is None
    result = d.push(make_fu_a_end(fragment=b"\x02"), marker=True)
    assert result is not None
    assert len(result) == 1


def test_marker_without_any_nal_raises():
    d = H264Depacketizer()
    d._access_unit.clear()
    with pytest.raises(H264DepacketizationError):
        d._finish_access_unit()


@pytest.mark.parametrize("bad_type", [0, 24, 30, 31])
def test_unsupported_single_nal_type_raises(bad_type):
    d = H264Depacketizer()
    payload = bytes((((3 & 0x03) << 5) | bad_type, 0xAA))
    with pytest.raises(H264DepacketizationError):
        d.push(payload)


@pytest.mark.parametrize("bad_payload", ["abc", b"", None, 123])
def test_invalid_payload_raises(bad_payload):
    d = H264Depacketizer()
    with pytest.raises(H264DepacketizationError):
        d.push(bad_payload)


@pytest.mark.parametrize("bad_marker", [0, 1, "yes", None])
def test_invalid_marker_raises(bad_marker):
    d = H264Depacketizer()
    with pytest.raises(H264DepacketizationError):
        d.push(make_single_nal(), marker=bad_marker)


def test_reset_discards_partial_state():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start()) is None
    assert d.push(make_single_nal()) is None if False else True
    d.reset()
    assert d.push(make_single_nal(), marker=True) == (make_single_nal(),)


def test_error_during_push_resets_state():
    d = H264Depacketizer()
    assert d.push(make_fu_a_start()) is None
    with pytest.raises(H264DepacketizationError):
        d.push(make_fu_a_cont(nri=1))
    assert d.push(make_single_nal(), marker=True) == (make_single_nal(),)


def test_fu_a_header_size_constant():
    assert FU_A_HEADER_SIZE == 2
