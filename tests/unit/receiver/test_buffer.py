import pytest

from rtp_live_streaming.common.rtp_packet import RTPPacket
from rtp_live_streaming.receiver.jitter_buffer import (
    JitterBuffer,
    JitterBufferStats,
)


def make_packet(sequence_number=0, payload=b"data"):
    return RTPPacket(
        payload_type=96,
        sequence_number=sequence_number,
        timestamp=90000,
        ssrc=12345,
        payload=payload,
    )


def make_buffer(buffer_ms=50, max_packets=512):
    return JitterBuffer(buffer_ms=buffer_ms, max_packets=max_packets)


def test_constructor_negative_buffer_ms_raises():
    with pytest.raises(ValueError):
        JitterBuffer(buffer_ms=-1)


def test_constructor_zero_max_packets_raises():
    with pytest.raises(ValueError):
        JitterBuffer(max_packets=0)


def test_constructor_negative_max_packets_raises():
    with pytest.raises(ValueError):
        JitterBuffer(max_packets=-5)


def test_constructor_accepts_zero_buffer_ms():
    b = JitterBuffer(buffer_ms=0)
    assert b.buffer_ms == 0.0


def test_constructor_stores_parameters():
    b = make_buffer(buffer_ms=100, max_packets=10)
    assert b.buffer_ms == 100.0
    assert b.max_packets == 10


def test_initial_occupancy_zero():
    assert make_buffer().occupancy == 0


def test_initial_expected_sequence_none():
    assert make_buffer().expected_sequence is None


def test_initial_stats_all_zero():
    stats = make_buffer().stats
    assert stats == JitterBufferStats(
        received_packets=0,
        released_packets=0,
        reordered_packets=0,
        duplicate_packets=0,
        late_packets=0,
        lost_packets=0,
        overflow_packets=0,
        buffered_packets=0,
    )


def test_push_in_order_accepted():
    b = make_buffer()
    assert b.push(make_packet(0), arrival_time=0.0) is True
    assert b.occupancy == 1


def test_push_sets_expected_sequence_on_first_packet():
    b = make_buffer()
    b.push(make_packet(100), arrival_time=0.0)
    assert b.expected_sequence == 100


def test_push_increments_received():
    b = make_buffer()
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    assert b.stats.received_packets == 2


def test_push_accepts_arrival_time_none():
    b = make_buffer()
    assert b.push(make_packet(0)) is True


@pytest.mark.parametrize("sequence", [0, 1, 65534, 65535])
def test_push_sequence_boundaries(sequence):
    b = make_buffer()
    assert b.push(make_packet(sequence), arrival_time=0.0) is True


@pytest.mark.parametrize("bad_seq", [-1, 65536, "0", 1.5, True, None])
def test_push_invalid_sequence_raises(bad_seq):
    b = make_buffer()
    with pytest.raises((TypeError, ValueError)):
        b.push(make_packet(bad_seq), arrival_time=0.0)


def test_push_duplicate_rejected():
    b = make_buffer()
    b.push(make_packet(5), arrival_time=0.0)
    assert b.push(make_packet(5), arrival_time=0.0) is False
    assert b.stats.duplicate_packets == 1
    assert b.occupancy == 1


def test_push_duplicate_does_not_increment_reordered():
    b = make_buffer()
    b.push(make_packet(5), arrival_time=0.0)
    b.push(make_packet(5), arrival_time=0.0)
    assert b.stats.reordered_packets == 0


def test_push_out_of_order_accepted_and_counted():
    b = make_buffer()
    b.push(make_packet(1), arrival_time=0.0)
    b.push(make_packet(0), arrival_time=0.0)
    assert b.stats.reordered_packets == 1
    assert b.occupancy == 2


def test_push_in_order_not_counted_as_reordered():
    b = make_buffer()
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    assert b.stats.reordered_packets == 0


def test_push_handles_wraparound_ordering():
    b = make_buffer()
    b.push(make_packet(65535), arrival_time=0.0)
    b.push(make_packet(0), arrival_time=0.0)
    assert b.stats.reordered_packets == 0
    assert b.occupancy == 2


def test_push_late_packet_rejected():
    b = make_buffer()
    b.push(make_packet(10), arrival_time=0.0)
    b.push(make_packet(11), arrival_time=0.0)
    b.pop_ready(now=1.0)
    assert b.expected_sequence == 12
    assert b.push(make_packet(5), arrival_time=2.0) is False
    assert b.stats.late_packets == 1


def test_push_late_at_half_range_boundary_rejected():
    b = make_buffer()
    b.push(make_packet(0), arrival_time=0.0)
    assert b.push(make_packet(32768), arrival_time=0.0) is False
    assert b.stats.late_packets == 1


def test_push_overflow_rejected():
    b = make_buffer(max_packets=2)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    assert b.push(make_packet(2), arrival_time=0.0) is False
    assert b.stats.overflow_packets == 1
    assert b.occupancy == 2


def test_push_overflow_preserves_existing():
    b = make_buffer(max_packets=1)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    ready = b.pop_ready(now=1.0)
    assert [p.sequence_number for p in ready] == [0]


def test_pop_ready_empty_buffer_returns_empty():
    assert make_buffer().pop_ready(now=0.0) == []


def test_pop_ready_before_delay_returns_empty():
    b = make_buffer(buffer_ms=100)
    b.push(make_packet(0), arrival_time=0.0)
    assert b.pop_ready(now=0.05) == []


def test_pop_ready_at_delay_returns_packet():
    b = make_buffer(buffer_ms=100)
    b.push(make_packet(0), arrival_time=0.0)
    ready = b.pop_ready(now=0.1)
    assert [p.sequence_number for p in ready] == [0]


def test_pop_ready_after_delay_returns_packet():
    b = make_buffer(buffer_ms=100)
    b.push(make_packet(0), arrival_time=0.0)
    ready = b.pop_ready(now=5.0)
    assert [p.sequence_number for p in ready] == [0]


def test_pop_ready_in_order():
    b = make_buffer(buffer_ms=0)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    b.push(make_packet(2), arrival_time=0.0)
    ready = b.pop_ready(now=0.0)
    assert [p.sequence_number for p in ready] == [0, 1, 2]


def test_pop_ready_reorders_out_of_order():
    b = make_buffer(buffer_ms=0)
    b.push(make_packet(2), arrival_time=0.0)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    ready = b.pop_ready(now=0.0)
    assert [p.sequence_number for p in ready] == [0, 1, 2]


def test_pop_ready_declares_lost_after_delay():
    b = make_buffer(buffer_ms=100)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(2), arrival_time=0.0)
    ready = b.pop_ready(now=0.1)
    assert [p.sequence_number for p in ready] == [0, 2]
    assert b.stats.lost_packets == 1


def test_pop_ready_multiple_losses_counted():
    b = make_buffer(buffer_ms=100)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(4), arrival_time=0.0)
    ready = b.pop_ready(now=0.1)
    assert [p.sequence_number for p in ready] == [0, 4]
    assert b.stats.lost_packets == 3


def test_pop_ready_released_count_matches():
    b = make_buffer(buffer_ms=0)
    for i in range(5):
        b.push(make_packet(i), arrival_time=0.0)
    ready = b.pop_ready(now=0.0)
    assert len(ready) == 5
    assert b.stats.released_packets == 5


def test_pop_ready_updates_expected_sequence():
    b = make_buffer(buffer_ms=0)
    b.push(make_packet(10), arrival_time=0.0)
    b.pop_ready(now=0.0)
    assert b.expected_sequence == 11


def test_pop_ready_handles_wraparound():
    b = make_buffer(buffer_ms=0)
    b.push(make_packet(65534), arrival_time=0.0)
    b.push(make_packet(65535), arrival_time=0.0)
    b.push(make_packet(0), arrival_time=0.0)
    ready = b.pop_ready(now=0.0)
    assert [p.sequence_number for p in ready] == [65534, 65535, 0]


def test_pop_ready_wraparound_expected_after_release():
    b = make_buffer(buffer_ms=0)
    b.push(make_packet(65535), arrival_time=0.0)
    b.pop_ready(now=0.0)
    assert b.expected_sequence == 0


def test_pop_ready_returns_original_packet_objects():
    b = make_buffer(buffer_ms=0)
    packet = make_packet(0, payload=b"hello")
    b.push(packet, arrival_time=0.0)
    ready = b.pop_ready(now=0.0)
    assert ready[0] is packet


def test_pop_ready_partial_release():
    b = make_buffer(buffer_ms=100)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.5)
    ready = b.pop_ready(now=0.2)
    assert [p.sequence_number for p in ready] == [0]
    assert b.occupancy == 1


def test_pop_ready_default_now_uses_monotonic():
    b = make_buffer(buffer_ms=0)
    b.push(make_packet(0))
    assert len(b.pop_ready()) == 1


def test_flush_empty_returns_empty():
    assert make_buffer().flush() == []


def test_flush_releases_all_in_order():
    b = make_buffer(buffer_ms=1000)
    b.push(make_packet(2), arrival_time=0.0)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    ready = b.flush()
    assert [p.sequence_number for p in ready] == [0, 1, 2]
    assert b.occupancy == 0


def test_flush_counts_gaps_as_lost():
    b = make_buffer(buffer_ms=1000)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(3), arrival_time=0.0)
    ready = b.flush()
    assert [p.sequence_number for p in ready] == [0, 3]
    assert b.stats.lost_packets == 2


def test_flush_increments_released():
    b = make_buffer(buffer_ms=1000)
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    b.flush()
    assert b.stats.released_packets == 2


def test_flush_handles_wraparound():
    b = make_buffer(buffer_ms=1000)
    b.push(make_packet(65535), arrival_time=0.0)
    b.push(make_packet(0), arrival_time=0.0)
    ready = b.flush()
    assert [p.sequence_number for p in ready] == [65535, 0]


def test_reset_discards_packets():
    b = make_buffer()
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    b.reset()
    assert b.occupancy == 0
    assert b.expected_sequence is None


def test_reset_preserves_stats_by_default():
    b = make_buffer()
    b.push(make_packet(0), arrival_time=0.0)
    b.reset()
    assert b.stats.received_packets == 1


def test_reset_with_reset_stats_clears_counters():
    b = make_buffer()
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(0), arrival_time=0.0)
    b.reset(reset_stats=True)
    assert b.stats.received_packets == 0
    assert b.stats.duplicate_packets == 0


def test_reset_allows_new_sequence_space():
    b = make_buffer(buffer_ms=0)
    b.push(make_packet(1000), arrival_time=0.0)
    b.pop_ready(now=1.0)
    b.reset()
    assert b.push(make_packet(0), arrival_time=2.0) is True
    assert b.expected_sequence == 0


def test_stats_buffered_matches_occupancy():
    b = make_buffer()
    b.push(make_packet(0), arrival_time=0.0)
    b.push(make_packet(1), arrival_time=0.0)
    assert b.stats.buffered_packets == b.occupancy == 2


def test_stats_snapshot_is_consistent_type():
    b = make_buffer()
    assert isinstance(b.stats, JitterBufferStats)
