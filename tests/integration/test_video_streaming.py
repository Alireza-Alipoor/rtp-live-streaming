"""End-to-end H.264-over-RTP integration test.

Set ``RTP_TEST_VIDEO`` to the path of a video file before running this test:

    RTP_TEST_VIDEO=/path/to/video.mp4
    uv run pytest tests/integration/test_video_streaming.py

The test uses localhost UDP sockets and does not open a display window.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from rtp_live_streaming.common.config import VideoConfig
from rtp_live_streaming.receiver.player import VideoPlayer
from rtp_live_streaming.sender.capture.video import VideoFileCapture
from rtp_live_streaming.sender.encoding.video_encoder import VideoEncoder
from rtp_live_streaming.sender.rtp_sender import RTPSender


@pytest.fixture
def video_path() -> Path:
    """Return the integration-test video supplied through the environment."""
    configured_path = os.environ.get("RTP_TEST_VIDEO")
    if not configured_path:
        pytest.skip("Set RTP_TEST_VIDEO to a local video file to run this test")

    path = Path(configured_path).expanduser()
    if not path.is_file():
        pytest.fail(f"RTP_TEST_VIDEO does not point to a file: {path}")
    return path


def test_video_file_is_encoded_sent_and_decoded_over_rtp(video_path: Path) -> None:
    """Stream a short file through localhost UDP and decode at least one frame."""
    pytest.importorskip("av", reason="PyAV is required for the video pipeline")

    decoded_frame = threading.Event()
    decoded_frame_count = 0
    callback_lock = threading.Lock()

    def on_frame(_frame: object) -> None:
        nonlocal decoded_frame_count
        with callback_lock:
            decoded_frame_count += 1
        decoded_frame.set()

    player = VideoPlayer(
        bind_host="127.0.0.1",
        bind_port=0,
        jitter_buffer_ms=20,
        on_frame=on_frame,
        display=False,
        receive_timeout=0.01,
    )

    sender: RTPSender | None = None
    try:
        player.start()
        sender = RTPSender(
            "127.0.0.1",
            player.address[1],
            codec="h264",
            max_datagram_size=1200,
        )

        # Keep the test bounded even when the supplied media is long.
        config = VideoConfig(codec="h264", width=640, height=480, fps=15)
        with VideoFileCapture(video_path, realtime=False) as capture:
            with VideoEncoder(config) as encoder:
                for frame_index, frame in enumerate(capture.frames()):
                    for encoded_packet in encoder.encode(frame):
                        sender.send_h264_access_unit(
                            encoded_packet.data,
                            encoded_packet.timestamp,
                        )

                    if frame_index >= 59:
                        break

                # Flush delayed encoder output so the final frames are sent.
                for encoded_packet in encoder.flush():
                    sender.send_h264_access_unit(
                        encoded_packet.data,
                        encoded_packet.timestamp,
                    )

        assert decoded_frame.wait(timeout=5), (
            "The receiver did not decode a video frame within 5 seconds; "
            f"player stats: {player.stats}"
        )

        assert decoded_frame_count >= 1
        assert player.stats.packets_received > 0
        assert player.stats.access_units_completed > 0
        assert player.stats.frames_decoded >= 1
    finally:
        if sender is not None:
            sender.close()
        player.close()
