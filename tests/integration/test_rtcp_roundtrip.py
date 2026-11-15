"""End-to-end UDP tests for the RTCP sender and receiver components."""

from __future__ import annotations

import socket

import pytest

from rtp_live_streaming.common.rtcp import RawRTCPPacket, SenderReport
from rtp_live_streaming.common.rtp_packet import RTPPacket
from rtp_live_streaming.receiver.rtcp_receiver import RTCPReceiver
from rtp_live_streaming.sender.rtcp_sender import RTCPSender


@pytest.fixture
def rtcp_receiver():
    """Bind an RTCP receiver to an ephemeral localhost UDP port."""
    receiver = RTCPReceiver(
        bind_host="127.0.0.1",
        bind_port=0,
        receive_timeout=0.05,
    )
    try:
        yield receiver
    finally:
        receiver.close()


def test_sender_report_round_trip_over_udp(rtcp_receiver):
    """RTCPSender's compound SR+SDES datagram is parsed by RTCPReceiver."""
    ssrc = 0x10203040
    received_callbacks = []
    rtcp_receiver.on_packet = received_callbacks.append

    sender = RTCPSender(
        destination_host=rtcp_receiver.address[0],
        destination_port=rtcp_receiver.address[1],
        ssrc=ssrc,
        clock_rate=90_000,
        cname="integration-test",
    )

    media_packet = RTPPacket(
        payload_type=96,
        sequence_number=7,
        timestamp=90_000,
        ssrc=ssrc,
        payload=b"encoded-media-payload",
        marker=True,
    )
    sender.record_packet(media_packet)

    try:
        expected_report = sender.send_report(
            rtp_timestamp=90_000,
            unix_time=1_700_000_000,
        )
        received = rtcp_receiver.receive_once(timeout=1.0)
    finally:
        sender.close()

    assert len(received) == 2
    report, sdes = received

    assert isinstance(report, SenderReport)
    assert report == expected_report
    assert report.ssrc == ssrc
    assert report.sender_packet_count == 1
    assert report.sender_octet_count == len(media_packet.payload)

    # SDES is currently preserved as a raw RTCP packet by the parser.
    assert isinstance(sdes, RawRTCPPacket)
    assert sdes.packet_type == 202
    assert len(received_callbacks) == 2
    assert received_callbacks == list(received)

    stats = rtcp_receiver.stats
    assert stats.datagrams_received == 1
    assert stats.packets_received == 2
    assert stats.sender_reports == 1
    assert stats.other_packets == 1
    assert stats.malformed_datagrams == 0
    assert stats.active_sources == 1
    assert rtcp_receiver.last_peer is not None


def test_receiver_counts_malformed_rtcp_datagram(rtcp_receiver):
    """Invalid RTCP input is counted and reported without crashing receiver."""
    errors = []
    rtcp_receiver.on_error = errors.append

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as raw_sender:
        raw_sender.sendto(b"\x00\x01\x02", rtcp_receiver.address)

    received = rtcp_receiver.receive_once(timeout=1.0)

    assert received == ()
    assert rtcp_receiver.stats.datagrams_received == 1
    assert rtcp_receiver.stats.malformed_datagrams == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
