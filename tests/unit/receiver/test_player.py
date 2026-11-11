from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rtp_live_streaming.common.payload_types import Codec
from rtp_live_streaming.common.rtp_packet import RTPPacket
from rtp_live_streaming.receiver.player import (
    UnsupportedVideoCodecError,
    VideoPlayer,
    VideoPlayerError,
    VideoPlayerStats,
)


def make_packet(
    payload_type=96, sequence_number=0, ssrc=12345, marker=False, payload=b"\x65\xaa"
):
    return RTPPacket(
        payload_type=payload_type,
        sequence_number=sequence_number,
        timestamp=90000,
        ssrc=ssrc,
        payload=payload,
        marker=marker,
    )


def install_fake_dependencies(monkeypatch):
    import rtp_live_streaming.receiver.player as module

    fake_receiver = MagicMock()
    fake_receiver.address = ("127.0.0.1", 5004)
    fake_receiver.codec_for_payload_type = MagicMock(return_value=Codec.H264)
    fake_receiver_class = MagicMock(return_value=fake_receiver)

    fake_depacketizer = MagicMock()
    fake_depacketizer.push = MagicMock(return_value=None)
    fake_depacketizer.reset = MagicMock()
    fake_depacketizer_class = MagicMock(return_value=fake_depacketizer)

    fake_decoder = MagicMock()
    fake_decoder.decode_access_unit = MagicMock(return_value=[])
    fake_decoder.flush = MagicMock(return_value=[])
    fake_decoder.close = MagicMock()
    fake_decoder_class = MagicMock(return_value=fake_decoder)

    monkeypatch.setattr(module, "RTPReceiver", fake_receiver_class)
    monkeypatch.setattr(module, "H264Depacketizer", fake_depacketizer_class)
    monkeypatch.setattr(module, "VideoDecoder", fake_decoder_class)

    return SimpleNamespace(
        receiver=fake_receiver,
        receiver_class=fake_receiver_class,
        depacketizer=fake_depacketizer,
        depacketizer_class=fake_depacketizer_class,
        decoder=fake_decoder,
        decoder_class=fake_decoder_class,
    )


def make_player(monkeypatch, **overrides):
    deps = install_fake_dependencies(monkeypatch)
    overrides.setdefault("bind_port", 0)
    player = VideoPlayer(**overrides)
    return player, deps


def install_fake_cv2(monkeypatch):
    import rtp_live_streaming.receiver.player as module

    fake_cv2 = SimpleNamespace(
        imshow=MagicMock(),
        waitKey=MagicMock(),
        destroyWindow=MagicMock(),
    )
    monkeypatch.setattr(
        module.VideoPlayer, "_import_cv2", staticmethod(lambda: fake_cv2)
    )
    return fake_cv2


@pytest.mark.parametrize("bad", [123, "yes", object()])
def test_constructor_invalid_on_frame_raises(monkeypatch, bad):
    install_fake_dependencies(monkeypatch)
    with pytest.raises(TypeError):
        VideoPlayer(on_frame=bad)


@pytest.mark.parametrize("bad", [123, "yes", object()])
def test_constructor_invalid_on_error_raises(monkeypatch, bad):
    install_fake_dependencies(monkeypatch)
    with pytest.raises(TypeError):
        VideoPlayer(on_error=bad)


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_constructor_invalid_display_raises(monkeypatch, bad):
    install_fake_dependencies(monkeypatch)
    with pytest.raises(TypeError):
        VideoPlayer(display=bad)


@pytest.mark.parametrize("bad", ["", None, 123])
def test_constructor_invalid_window_name_raises(monkeypatch, bad):
    install_fake_dependencies(monkeypatch)
    with pytest.raises(ValueError):
        VideoPlayer(window_name=bad)


@pytest.mark.parametrize("bad", ["not a dict", 123, []])
def test_constructor_invalid_decoder_options_raises(monkeypatch, bad):
    install_fake_dependencies(monkeypatch)
    with pytest.raises(TypeError):
        VideoPlayer(decoder_options=bad)


def test_constructor_receiver_error_wrapped(monkeypatch):
    import rtp_live_streaming.receiver.player as module

    def failing_receiver(*args, **kwargs):
        raise OSError("bind failed")

    monkeypatch.setattr(module, "RTPReceiver", failing_receiver)
    with pytest.raises(VideoPlayerError):
        VideoPlayer()


def test_constructor_display_imports_cv2(monkeypatch):
    install_fake_dependencies(monkeypatch)
    cv2 = install_fake_cv2(monkeypatch)
    player = VideoPlayer(display=True, window_name="Test")
    try:
        assert player._cv2 is cv2
        assert player.window_name == "Test"
    finally:
        player.close()


def test_constructor_display_missing_cv2_raises(monkeypatch):
    import rtp_live_streaming.receiver.player as module

    install_fake_dependencies(monkeypatch)

    def missing():
        raise VideoPlayerError("OpenCV is required")

    monkeypatch.setattr(module.VideoPlayer, "_import_cv2", staticmethod(missing))
    with pytest.raises(VideoPlayerError):
        VideoPlayer(display=True)


def test_initial_stats_zero(monkeypatch):
    player, _ = make_player(monkeypatch)
    try:
        assert player.stats == VideoPlayerStats(
            packets_received=0,
            access_units_completed=0,
            frames_decoded=0,
            decode_errors=0,
            callback_errors=0,
        )
    finally:
        player.close()


def test_initial_is_closed_false(monkeypatch):
    player, _ = make_player(monkeypatch)
    try:
        assert player.is_closed is False
    finally:
        player.close()


def test_address_delegates_to_receiver(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        assert player.address == ("127.0.0.1", 5004)
    finally:
        player.close()


def test_get_source_creates_h264_source(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        packet = make_packet()
        source = player._get_source(packet)
        assert source is not None
        assert deps.depacketizer_class.call_count == 1
        assert deps.decoder_class.call_count == 1
    finally:
        player.close()


def test_get_source_reuses_source_for_same_ssrc(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        packet = make_packet(ssrc=1111)
        s1 = player._get_source(packet)
        s2 = player._get_source(packet)
        assert s1 is s2
        assert deps.depacketizer_class.call_count == 1
    finally:
        player.close()


def test_get_source_creates_separate_sources_per_ssrc(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        player._get_source(make_packet(ssrc=1111))
        player._get_source(make_packet(ssrc=2222))
        assert deps.depacketizer_class.call_count == 2
        assert deps.decoder_class.call_count == 2
    finally:
        player.close()


@pytest.mark.parametrize("codec", [Codec.OPUS, Codec.PCMU, Codec.VP8])
def test_get_source_unsupported_codec_raises(monkeypatch, codec):
    player, deps = make_player(monkeypatch)
    try:
        deps.receiver.codec_for_payload_type = MagicMock(return_value=codec)
        with pytest.raises(UnsupportedVideoCodecError):
            player._get_source(make_packet())
    finally:
        player.close()


def test_get_source_decoder_error_propagates(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        from rtp_live_streaming.receiver.decoding.video_decoder import VideoDecoderError

        deps.decoder_class.side_effect = VideoDecoderError("no decoder")
        with pytest.raises(VideoDecoderError):
            player._get_source(make_packet())
    finally:
        player.close()


def test_get_source_passes_decoder_options(monkeypatch):
    player, deps = make_player(monkeypatch, decoder_options={"threads": "4"})
    try:
        player._get_source(make_packet())
        assert deps.decoder_class.call_args.kwargs == {"options": {"threads": "4"}}
    finally:
        player.close()


def test_handle_packet_increments_packets_received(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        player._handle_packet(make_packet())
        assert player.stats.packets_received == 1
    finally:
        player.close()


def test_handle_packet_incomplete_access_unit_returns(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        deps.depacketizer.push = MagicMock(return_value=None)
        player._handle_packet(make_packet())
        assert player.stats.access_units_completed == 0
        assert player.stats.frames_decoded == 0
    finally:
        player.close()


def test_handle_packet_complete_access_unit_decodes(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        frame = object()
        deps.depacketizer.push = MagicMock(return_value=(b"\x65\xaa",))
        deps.decoder.decode_access_unit = MagicMock(return_value=[frame])
        received = []
        player.on_frame = received.append

        player._handle_packet(make_packet(marker=True))

        assert player.stats.access_units_completed == 1
        assert player.stats.frames_decoded == 1
        assert received == [frame]
    finally:
        player.close()


def test_handle_packet_passes_timestamp_to_decoder(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        deps.depacketizer.push = MagicMock(return_value=(b"\x65",))
        deps.decoder.decode_access_unit = MagicMock(return_value=[])
        packet = make_packet()
        player._handle_packet(packet)
        kwargs = deps.decoder.decode_access_unit.call_args.kwargs
        assert kwargs["timestamp"] == packet.timestamp
    finally:
        player.close()


def test_handle_packet_depacketization_error_counted(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        from rtp_live_streaming.receiver.depacketization.h264 import (
            H264DepacketizationError,
        )

        deps.depacketizer.push = MagicMock(side_effect=H264DepacketizationError("bad"))
        errors = []
        player.on_error = errors.append

        player._handle_packet(make_packet())

        assert player.stats.decode_errors == 1
        assert len(errors) == 1
        deps.depacketizer.reset.assert_called_once()
    finally:
        player.close()


def test_handle_packet_decoder_error_counted(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        from rtp_live_streaming.receiver.decoding.video_decoder import VideoDecoderError

        deps.depacketizer.push = MagicMock(return_value=(b"\x65",))
        deps.decoder.decode_access_unit = MagicMock(
            side_effect=VideoDecoderError("bad")
        )
        errors = []
        player.on_error = errors.append

        player._handle_packet(make_packet())

        assert player.stats.decode_errors == 1
        assert len(errors) == 1
    finally:
        player.close()


def test_handle_packet_unsupported_codec_counted(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        deps.receiver.codec_for_payload_type = MagicMock(return_value=Codec.OPUS)
        errors = []
        player.on_error = errors.append

        player._handle_packet(make_packet())

        assert player.stats.decode_errors == 1
        assert len(errors) == 1
    finally:
        player.close()


def test_handle_packet_unexpected_exception_counted(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        deps.depacketizer.push = MagicMock(side_effect=RuntimeError("boom"))
        errors = []
        player.on_error = errors.append

        player._handle_packet(make_packet())

        assert player.stats.decode_errors == 1
        assert len(errors) == 1
        assert isinstance(errors[0], VideoPlayerError)
    finally:
        player.close()


def test_deliver_frame_calls_on_frame(monkeypatch):
    player, _ = make_player(monkeypatch)
    try:
        received = []
        player.on_frame = received.append
        frame = object()
        player._deliver_frame(frame)
        assert received == [frame]
    finally:
        player.close()


def test_deliver_frame_callback_error_counted(monkeypatch):
    player, _ = make_player(monkeypatch)
    try:
        errors = []

        def bad(frame):
            raise RuntimeError("boom")

        player.on_frame = bad
        player.on_error = errors.append

        player._deliver_frame(object())

        assert player.stats.callback_errors == 1
        assert len(errors) == 1
    finally:
        player.close()


def test_deliver_frame_display_error_counted(monkeypatch):
    player, _ = make_player(monkeypatch, display=True)
    try:
        errors = []
        player.on_error = errors.append
        frame = MagicMock()
        frame.to_ndarray = MagicMock(side_effect=RuntimeError("no ndarray"))

        player._deliver_frame(frame)

        assert player.stats.callback_errors == 1
        assert len(errors) == 1
    finally:
        player.close()


def test_start_delegates_to_receiver(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        player.start(daemon=False)
        deps.receiver.start.assert_called_once_with(daemon=False)
    finally:
        player.close()


def test_start_on_closed_raises(monkeypatch):
    player, _ = make_player(monkeypatch)
    player.close()
    with pytest.raises(VideoPlayerError):
        player.start()


def test_start_receiver_error_wrapped(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        from rtp_live_streaming.receiver.rtp_receiver import RTPReceiverError

        deps.receiver.start = MagicMock(side_effect=RTPReceiverError("bad"))
        with pytest.raises(VideoPlayerError):
            player.start()
    finally:
        player.close()


def test_stop_sets_event_and_stops_receiver(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        player.stop()
        assert player._run_stop_event.is_set()
        deps.receiver.stop.assert_called_once_with(flush=False)
    finally:
        player.close()


def test_stop_when_closed_noop(monkeypatch):
    player, deps = make_player(monkeypatch)
    player.close()
    deps.receiver.stop.reset_mock()
    player.stop()
    deps.receiver.stop.assert_not_called()


def test_stop_receiver_error_reported(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        errors = []
        player.on_error = errors.append
        deps.receiver.stop = MagicMock(side_effect=RuntimeError("boom"))
        player.stop()
        assert len(errors) == 1
    finally:
        player.close()


def test_run_starts_and_closes(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:

        def fake_wait(timeout):
            return True

        player._run_stop_event.wait = fake_wait
        player.run()
        deps.receiver.start.assert_called_once()
        assert player.is_closed is True
    finally:
        player.close()


def test_run_handles_keyboard_interrupt(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:

        def fake_wait(timeout):
            raise KeyboardInterrupt()

        player._run_stop_event.wait = fake_wait
        player.run()
        assert player.is_closed is True
    finally:
        player.close()


def test_close_stops_receiver_and_flushes(monkeypatch):
    player, deps = make_player(monkeypatch)
    player.close()
    deps.receiver.stop.assert_called_once_with(flush=True)
    deps.receiver.close.assert_called_once()
    assert player.is_closed is True


def test_close_flushes_decoders(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        frame = object()
        deps.decoder.flush = MagicMock(return_value=[frame])
        received = []
        player.on_frame = received.append

        player._handle_packet(make_packet())
        player.close()

        deps.decoder.flush.assert_called_once()
        deps.decoder.close.assert_called_once()
        assert received == [frame]
        assert player.stats.frames_decoded == 1
    finally:
        player.close()


def test_close_decoder_flush_error_reported(monkeypatch):
    player, deps = make_player(monkeypatch)
    try:
        from rtp_live_streaming.receiver.decoding.video_decoder import VideoDecoderError

        errors = []
        player.on_error = errors.append
        deps.decoder.flush = MagicMock(side_effect=VideoDecoderError("bad"))

        player._handle_packet(make_packet())
        player.close()

        assert len(errors) == 1
        assert player.is_closed is True
    finally:
        player.close()


def test_close_is_idempotent(monkeypatch):
    player, deps = make_player(monkeypatch)
    player.close()
    deps.receiver.close.reset_mock()
    player.close()
    deps.receiver.close.assert_not_called()


def test_close_window_destroy_error_swallowed(monkeypatch):
    player, _ = make_player(monkeypatch, display=True)
    cv2 = player._cv2
    cv2.destroyWindow = MagicMock(side_effect=RuntimeError("boom"))
    frame = MagicMock()
    frame.to_ndarray = MagicMock(return_value="image")
    player._deliver_frame(frame)

    player.close()
    assert player.is_closed is True


def test_report_error_calls_on_error(monkeypatch):
    player, _ = make_player(monkeypatch)
    try:
        errors = []
        player.on_error = errors.append
        exc = RuntimeError("boom")
        player._report_error(exc)
        assert errors == [exc]
    finally:
        player.close()


def test_report_error_callback_exception_swallowed(monkeypatch):
    player, _ = make_player(monkeypatch)
    try:

        def bad(exc):
            raise RuntimeError("boom")

        player.on_error = bad
        player._report_error(RuntimeError("original"))
    finally:
        player.close()


def test_context_manager_closes(monkeypatch):
    player, deps = make_player(monkeypatch)
    with player as p:
        assert p is player
        assert p.is_closed is False
    assert player.is_closed is True
    deps.receiver.close.assert_called_once()


def test_context_manager_enter_when_closed_raises(monkeypatch):
    player, _ = make_player(monkeypatch)
    player.close()
    with pytest.raises(VideoPlayerError):
        player.__enter__()
