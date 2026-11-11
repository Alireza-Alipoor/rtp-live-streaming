"""Integration tests for the RTP-over-UDP sender and receiver.

Run from the repository root with:

    uv run pytest tests/unit/test_rtp_transport.py -v
"""

from __future__ import annotations

import socket
import time

import pytest

from rtp_live_streaming.common.payload_types import Codec
from rtp_live_streaming.common.rtp_packet import RTPPacket
from rtp_live_streaming.receiver.rtp_receiver import (
    RTPReceiver,
    RTPReceiverError,
)
from rtp_live_streaming.sender.rtp_sender import RTPSender, RTPSenderError


@pytest.fixture
def receiver() -> RTPReceiver:
    """Bind an ephemeral localhost UDP port for each test."""
    instance = RTPReceiver(
        bind_host="127.0.0.1",
        bind_port=0,
        jitter_buffer_ms=0,
        receive_timeout=0.01,
    )
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def sender(receiver: RTPReceiver) -> RTPSender:
    """Create a sender targeting the receiver fixture."""
    host, port = receiver.address
    instance = RTPSender(
        host,
        port,
        Codec.OPUS,
        ssrc=0x12345678,
        initial_sequence_number=100,
    )
    try:
        yield instance
    finally:
        instance.close()


def test_sender_receiver_round_trip(sender: RTPSender, receiver: RTPReceiver) -> None:
    sent = sender.send_payload(b"opus-frame", timestamp=48_000, marker=True)

    received = receiver.receive_once(timeout=0.1)

    assert len(received) == 1
    packet = received[0]
    assert packet.payload == b"opus-frame"
    assert packet.payload_type == sent.payload_type == 111
    assert packet.sequence_number == sent.sequence_number == 100
    assert packet.timestamp == sent.timestamp == 48_000
    assert packet.ssrc == sent.ssrc == 0x12345678
    assert packet.marker is True


def test_sender_increments_sequence_number_and_wraps(
    receiver: RTPReceiver,
) -> None:
    host, port = receiver.address
    sender = RTPSender(
        host,
        port,
        Codec.OPUS,
        ssrc=0xABCDEF01,
        initial_sequence_number=0xFFFF,
    )
    try:
        sender.send_payload(b"last-before-wrap", timestamp=3)
        sender.send_payload(b"first-after-wrap", timestamp=4)

        first = receiver.receive_once(timeout=0.1)[0]
        second = receiver.receive_once(timeout=0.1)[0]

        assert [first.sequence_number, second.sequence_number] == [0xFFFF, 0]
        assert sender.next_sequence_number == 1
    finally:
        sender.close()


def test_sender_rejects_payload_that_exceeds_datagram_limit(
    receiver: RTPReceiver,
) -> None:
    host, port = receiver.address
    sender = RTPSender(
        host,
        port,
        Codec.OPUS,
        initial_sequence_number=7,
        max_datagram_size=16,
    )
    try:
        with pytest.raises(RTPSenderError, match="Packetize the media payload"):
            sender.send_payload(b"12345", timestamp=0)

        # A failed send must not consume a sequence number.
        assert sender.next_sequence_number == 7
    finally:
        sender.close()


@pytest.mark.parametrize(
    ("host", "port", "codec", "kwargs", "exception"),
    [
        ("", 5004, Codec.OPUS, {}, ValueError),
        ("127.0.0.1", 0, Codec.OPUS, {}, ValueError),
        ("127.0.0.1", 5004, "not-a-codec", {}, ValueError),
        (
            "127.0.0.1",
            5004,
            Codec.OPUS,
            {"initial_sequence_number": 65_536},
            ValueError,
        ),
    ],
)
def test_sender_validates_constructor_arguments(
    host: str,
    port: int,
    codec: Codec | str,
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        RTPSender(host, port, codec, **kwargs)  # type: ignore[arg-type]


def test_sender_codec_switch_preserves_sequence_and_ssrc(
    sender: RTPSender, receiver: RTPReceiver
) -> None:
    original_ssrc = sender.ssrc
    sender.set_codec(Codec.PCMU)
    sent = sender.send_payload(b"\xff\x00", timestamp=8_000)

    received = receiver.receive_once(timeout=0.1)[0]

    assert sender.codec is Codec.PCMU
    assert sent.payload_type == received.payload_type == 0
    assert received.ssrc == original_ssrc
    assert received.sequence_number == 100


def test_receiver_rejects_malformed_datagram(
    receiver: RTPReceiver,
) -> None:
    raw_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        raw_socket.sendto(b"too-short", receiver.address)
    finally:
        raw_socket.close()

    assert receiver.receive_once(timeout=0.1) == []
    assert receiver.stats.datagrams_received == 1
    assert receiver.stats.malformed_datagrams == 1


def test_receiver_counts_unsupported_payload_type(
    receiver: RTPReceiver,
) -> None:
    raw_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = RTPPacket(
        payload_type=120,  # Valid 7-bit PT, but not registered by this project.
        sequence_number=1,
        timestamp=0,
        ssrc=55,
        payload=b"unknown-codec",
    )
    try:
        raw_socket.sendto(packet.to_bytes(), receiver.address)
    finally:
        raw_socket.close()

    assert receiver.receive_once(timeout=0.1) == []
    assert receiver.stats.unsupported_payload_types == 1


def test_receiver_reorders_packets_within_jitter_window(
    receiver: RTPReceiver,
) -> None:
    # Use a non-zero playout delay so both datagrams can enter the buffer
    # before the first packet is released.
    receiver.jitter_buffer_ms = 25
    host, port = receiver.address
    sender = RTPSender(
        host,
        port,
        Codec.OPUS,
        ssrc=99,
        initial_sequence_number=11,
    )
    try:
        sender.send_payload(b"sequence-11", timestamp=960)
        sender.send_payload(b"sequence-12", timestamp=1_920)

        assert receiver.receive_once(timeout=0.1) == []
        assert receiver.receive_once(timeout=0.1) == []

        time.sleep(0.035)
        released = receiver.receive_once(timeout=0.02)

        assert [packet.sequence_number for packet in released] == [11, 12]
        assert [packet.payload for packet in released] == [
            b"sequence-11",
            b"sequence-12",
        ]
    finally:
        sender.close()


def test_receiver_invokes_packet_callback() -> None:
    received_packets: list[RTPPacket] = []
    receiver = RTPReceiver(
        bind_host="127.0.0.1",
        bind_port=0,
        jitter_buffer_ms=0,
        on_packet=received_packets.append,
    )
    host, port = receiver.address
    sender = RTPSender(host, port, Codec.OPUS, initial_sequence_number=5)
    try:
        sender.send_payload(b"callback-data", timestamp=123)
        returned = receiver.receive_once(timeout=0.1)

        assert len(returned) == 1
        assert received_packets == returned
    finally:
        sender.close()
        receiver.close()


def test_receiver_reports_parse_errors_to_callback() -> None:
    errors: list[Exception] = []
    receiver = RTPReceiver(
        bind_host="127.0.0.1",
        bind_port=0,
        on_error=errors.append,
    )
    raw_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        raw_socket.sendto(b"x", receiver.address)
        receiver.receive_once(timeout=0.1)
    finally:
        raw_socket.close()
        receiver.close()

    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


def test_receiver_flushes_buffered_packets_on_request(
    receiver: RTPReceiver, sender: RTPSender
) -> None:
    receiver.jitter_buffer_ms = 10_000
    sender.send_payload(b"buffered", timestamp=10)

    # The packet is accepted but is not yet eligible for playout.
    assert receiver.receive_once(timeout=0.1) == []

    flushed = receiver.stop(flush=True)
    assert len(flushed) == 1
    assert flushed[0].payload == b"buffered"
    assert receiver.stats.packets_delivered == 1


def test_closed_sender_and_receiver_reject_further_operations(
    receiver: RTPReceiver,
) -> None:
    host, port = receiver.address
    sender = RTPSender(host, port, Codec.OPUS)
    sender.close()

    with pytest.raises(RTPSenderError, match="closed"):
        sender.send_payload(b"data", timestamp=0)

    receiver.close()
    with pytest.raises(RTPReceiverError, match="closed"):
        receiver.receive_once()
