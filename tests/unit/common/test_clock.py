import pytest

from rtp_live_streaming.common import clock
from rtp_live_streaming.common.clock import (
    ClockError,
    RTPClock,
    elapsed_seconds,
    get_clock_rate,
    monotonic_ns,
    monotonic_seconds,
    rtp_ticks_to_seconds,
    seconds_to_rtp_ticks,
    timestamp_difference,
    timestamp_from_elapsed,
)
from rtp_live_streaming.common.payload_types import Codec


def test_get_clock_rate_known_codecs():
    assert get_clock_rate(Codec.H264) == 90_000
    assert get_clock_rate(Codec.VP8) == 90_000
    assert get_clock_rate(Codec.OPUS) == 48_000
    assert get_clock_rate(Codec.PCMU) == 8_000


@pytest.mark.parametrize(
    "codec,expected",
    [("h264", 90_000), ("OPUS", 48_000), ("Pcmu", 8_000), ("vp8", 90_000)],
)
def test_get_clock_rate_accepts_strings(codec, expected):
    assert get_clock_rate(codec) == expected


@pytest.mark.parametrize("codec", ["mp3", "aac", 123, None])
def test_get_clock_rate_unsupported_raises(codec):
    with pytest.raises(ClockError):
        get_clock_rate(codec)


def test_elapsed_seconds_basic():
    assert elapsed_seconds(0, 1_000_000_000) == 1.0
    assert elapsed_seconds(1_000_000_000, 2_500_000_000) == 1.5


def test_elapsed_seconds_zero():
    assert elapsed_seconds(5, 5) == 0.0


def test_elapsed_seconds_default_end_is_now():
    start = monotonic_ns()
    result = elapsed_seconds(start)
    assert result >= 0.0


def test_elapsed_seconds_end_before_start_raises():
    with pytest.raises(ClockError):
        elapsed_seconds(10, 5)


@pytest.mark.parametrize("bad", [1.5, "10", None, True])
def test_elapsed_seconds_invalid_start_raises(bad):
    with pytest.raises(ClockError):
        elapsed_seconds(bad, 10)


@pytest.mark.parametrize("bad", [1.5, "10", True])
def test_elapsed_seconds_invalid_end_raises(bad):
    with pytest.raises(ClockError):
        elapsed_seconds(0, bad)


def test_seconds_to_rtp_ticks_basic():
    assert seconds_to_rtp_ticks(1.0, 90_000) == 90_000
    assert seconds_to_rtp_ticks(0.5, 8_000) == 4_000
    assert seconds_to_rtp_ticks(0, 48_000) == 0


def test_seconds_to_rtp_ticks_rounding():
    assert seconds_to_rtp_ticks(0.00005, 48_000) == 2
    assert seconds_to_rtp_ticks(0.00006, 48_000) == 3


def test_seconds_to_rtp_ticks_accepts_int():
    assert seconds_to_rtp_ticks(2, 8_000) == 16_000


def test_seconds_to_rtp_ticks_negative_raises():
    with pytest.raises(ClockError):
        seconds_to_rtp_ticks(-0.1, 8_000)


@pytest.mark.parametrize("bad", [True, "1.0", None])
def test_seconds_to_rtp_ticks_invalid_seconds_raises(bad):
    with pytest.raises(ClockError):
        seconds_to_rtp_ticks(bad, 8_000)


@pytest.mark.parametrize("rate", [0, -1, True])
def test_seconds_to_rtp_ticks_invalid_clock_rate_raises(rate):
    with pytest.raises(ClockError):
        seconds_to_rtp_ticks(1.0, rate)


def test_rtp_ticks_to_seconds_basic():
    assert rtp_ticks_to_seconds(90_000, 90_000) == 1.0
    assert rtp_ticks_to_seconds(4_000, 8_000) == 0.5
    assert rtp_ticks_to_seconds(0, 48_000) == 0.0


def test_rtp_ticks_to_seconds_negative_raises():
    with pytest.raises(ClockError):
        rtp_ticks_to_seconds(-1, 8_000)


@pytest.mark.parametrize("bad", [True, 1.5, "1"])
def test_rtp_ticks_to_seconds_invalid_ticks_raises(bad):
    with pytest.raises(ClockError):
        rtp_ticks_to_seconds(bad, 8_000)


def test_rtp_ticks_to_seconds_invalid_clock_rate_raises():
    with pytest.raises(ClockError):
        rtp_ticks_to_seconds(1, 0)


def test_rtp_ticks_round_trip():
    ticks = seconds_to_rtp_ticks(1.25, 48_000)
    assert rtp_ticks_to_seconds(ticks, 48_000) == pytest.approx(1.25)


def test_timestamp_from_elapsed_zero():
    assert timestamp_from_elapsed(0, 90_000) == 0
    assert timestamp_from_elapsed(0, 90_000, initial_timestamp=123) == 123


def test_timestamp_from_elapsed_one_second():
    assert timestamp_from_elapsed(1_000_000_000, 90_000) == 90_000
    assert timestamp_from_elapsed(1_000_000_000, 48_000) == 48_000


def test_timestamp_from_elapsed_initial_offset():
    assert (
        timestamp_from_elapsed(1_000_000_000, 90_000, initial_timestamp=1000) == 91_000
    )


def test_timestamp_from_elapsed_wraparound():
    expected = 0

    assert (
        timestamp_from_elapsed(
            125_000,
            8_000,
            initial_timestamp=2**32 - 1,
        )
        == expected
    )


def test_timestamp_from_elapsed_negative_raises():
    with pytest.raises(ClockError):
        timestamp_from_elapsed(-1, 90_000)


def test_timestamp_from_elapsed_bool_elapsed_raises():
    with pytest.raises(ClockError):
        timestamp_from_elapsed(True, 90_000)


def test_timestamp_from_elapsed_bool_initial_raises():
    with pytest.raises(ClockError):
        timestamp_from_elapsed(0, 90_000, initial_timestamp=True)


@pytest.mark.parametrize("bad", [-1, 2**32, 2**32 + 1, 1.5])
def test_timestamp_from_elapsed_out_of_range_initial_raises(bad):
    with pytest.raises(ClockError):
        timestamp_from_elapsed(0, 90_000, initial_timestamp=bad)


def test_timestamp_difference_simple():
    assert timestamp_difference(200, 100) == 100


def test_timestamp_difference_zero():
    assert timestamp_difference(5, 5) == 0


def test_timestamp_difference_wraparound():
    assert timestamp_difference(100, 2**32 - 100) == 200


def test_timestamp_difference_backwards():
    assert timestamp_difference(100, 200) == 2**32 - 100


@pytest.mark.parametrize("newer,older", [(2**32, 0), (0, -1), (True, 0)])
def test_timestamp_difference_out_of_range_raises(newer, older):
    with pytest.raises(ClockError):
        timestamp_difference(newer, older)


@pytest.mark.parametrize("rate", [0, -1, True])
def test_rtp_clock_invalid_clock_rate_raises(rate):
    with pytest.raises(ClockError):
        RTPClock(rate)


@pytest.mark.parametrize("initial", [-1, 2**32])
def test_rtp_clock_invalid_initial_raises(initial):
    with pytest.raises(ClockError):
        RTPClock(90_000, initial_timestamp=initial)


def test_rtp_clock_initial_state():
    c = RTPClock(90_000, initial_timestamp=500)
    assert c.clock_rate == 90_000
    assert c.initial_timestamp == 500


def test_rtp_clock_for_codec():
    c = RTPClock.for_codec(Codec.OPUS)
    assert c.clock_rate == 48_000

    c2 = RTPClock.for_codec("h264", initial_timestamp=42)
    assert c2.clock_rate == 90_000
    assert c2.initial_timestamp == 42


def test_rtp_clock_timestamp_at_zero_elapsed():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clock, "monotonic_ns", lambda: 1000)
        c = RTPClock(8_000, initial_timestamp=7)
        assert c.timestamp() == 7


def test_rtp_clock_timestamp_after_elapsed():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clock, "monotonic_ns", lambda: 1000)
        c = RTPClock(8_000, initial_timestamp=0)
        mp.setattr(clock, "monotonic_ns", lambda: 1000 + 1_000_000_000)
        assert c.timestamp() == 8_000


def test_rtp_clock_timestamp_wraparound():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clock, "monotonic_ns", lambda: 0)
        c = RTPClock(8_000, initial_timestamp=0)
        mp.setattr(clock, "monotonic_ns", lambda: (2**32 // 8_000 + 2) * 1_000_000_000)
        ts = c.timestamp()
        assert 0 <= ts <= 2**32 - 1


def test_rtp_clock_tick_conversions():
    c = RTPClock(48_000)
    assert c.ticks_from_seconds(1.0) == 48_000
    assert c.seconds_from_ticks(24_000) == 0.5


def test_rtp_clock_timestamp_in_range():
    c = RTPClock.for_codec(Codec.H264)
    assert 0 <= c.timestamp() <= 2**32 - 1


def test_monotonic_ns_is_int():
    assert isinstance(monotonic_ns(), int)


def test_monotonic_seconds_is_float():
    assert isinstance(monotonic_seconds(), float)


def test_monotonic_ns_is_monotonic():
    a = monotonic_ns()
    b = monotonic_ns()
    assert b >= a
