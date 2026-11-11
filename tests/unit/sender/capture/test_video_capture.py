import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rtp_live_streaming.sender.capture.video import (
    VideoCaptureError,
    VideoFileCapture,
)


class FakeAVModule(SimpleNamespace):
    def __init__(self, container):
        super().__init__()
        self.open = MagicMock(return_value=container)


class FakeFrame(SimpleNamespace):
    def __init__(self, pts=None, time_base=None):
        super().__init__()
        self.pts = pts
        self.time_base = time_base


def make_stream(
    width=640,
    height=480,
    average_rate=None,
    duration=None,
    time_base=None,
):
    return SimpleNamespace(
        codec_context=SimpleNamespace(width=width, height=height),
        average_rate=average_rate,
        duration=duration,
        time_base=time_base,
        thread_type=None,
    )


def make_container(streams, frames):
    container = MagicMock()
    container.streams.video = streams
    container.decode = MagicMock(return_value=iter(frames))
    return container


def make_file(tmp_path, name="video.mp4"):
    path = tmp_path / name
    path.write_bytes(b"fake video")
    return path


def install_fake_av(monkeypatch, container):
    import rtp_live_streaming.sender.capture.video as module

    fake_av = FakeAVModule(container)
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setattr(
        module.VideoFileCapture, "_import_av", staticmethod(lambda: fake_av)
    )
    return fake_av


@pytest.mark.parametrize("bad", [123, 1.5, None, ["a"]])
def test_constructor_invalid_source_type_raises(tmp_path, bad):
    with pytest.raises(TypeError):
        VideoFileCapture(bad)


@pytest.mark.parametrize("bad", [1.5, "0", True, None])
def test_constructor_invalid_stream_index_type_raises(tmp_path, bad, monkeypatch):
    path = make_file(tmp_path)
    install_fake_av(monkeypatch, make_container([], []))
    with pytest.raises(TypeError):
        VideoFileCapture(path, stream_index=bad)


def test_constructor_negative_stream_index_raises(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    install_fake_av(monkeypatch, make_container([], []))
    with pytest.raises(ValueError):
        VideoFileCapture(path, stream_index=-1)


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_constructor_invalid_realtime_raises(tmp_path, bad, monkeypatch):
    path = make_file(tmp_path)
    install_fake_av(monkeypatch, make_container([], []))
    with pytest.raises(TypeError):
        VideoFileCapture(path, realtime=bad)


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_constructor_invalid_loop_raises(tmp_path, bad, monkeypatch):
    path = make_file(tmp_path)
    install_fake_av(monkeypatch, make_container([], []))
    with pytest.raises(TypeError):
        VideoFileCapture(path, loop=bad)


def test_constructor_missing_file_raises(tmp_path, monkeypatch):
    install_fake_av(monkeypatch, make_container([], []))
    with pytest.raises(FileNotFoundError):
        VideoFileCapture(tmp_path / "missing.mp4")


def test_constructor_pyav_missing_raises(tmp_path, monkeypatch):
    import rtp_live_streaming.sender.capture.video as module

    path = make_file(tmp_path)

    def missing_import():
        raise module.VideoCaptureError("PyAV is required")

    monkeypatch.setattr(
        module.VideoFileCapture,
        "_import_av",
        staticmethod(missing_import),
    )

    with pytest.raises(module.VideoCaptureError, match="PyAV is required"):
        module.VideoFileCapture(path)


def test_open_stream_index_out_of_range_raises(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    container = make_container([make_stream()], [])
    install_fake_av(monkeypatch, container)
    with pytest.raises(VideoCaptureError):
        VideoFileCapture(path, stream_index=1)
    container.close.assert_called_once()


def test_open_failure_wrapped_as_video_capture_error(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    fake_av = FakeAVModule(MagicMock())
    fake_av.open = MagicMock(side_effect=RuntimeError("boom"))

    import rtp_live_streaming.sender.capture.video as module

    monkeypatch.setattr(
        module.VideoFileCapture, "_import_av", staticmethod(lambda: fake_av)
    )
    with pytest.raises(VideoCaptureError):
        VideoFileCapture(path)


def test_properties_after_open(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    from fractions import Fraction

    stream = make_stream(
        width=1280,
        height=720,
        average_rate=Fraction(30, 1),
        duration=90,
        time_base=Fraction(1, 90_000),
    )
    install_fake_av(monkeypatch, make_container([stream], []))
    capture = VideoFileCapture(path)
    try:
        assert capture.source == path
        assert capture.width == 1280
        assert capture.height == 720
        assert float(capture.average_rate) == 30.0
        assert capture.duration == pytest.approx(90 / 90_000)
        assert capture.is_closed is False
    finally:
        capture.close()


def test_duration_none_when_no_time_base(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    stream = make_stream(duration=10, time_base=None)
    install_fake_av(monkeypatch, make_container([stream], []))
    capture = VideoFileCapture(path)
    try:
        assert capture.duration is None
    finally:
        capture.close()


def test_duration_none_when_no_duration(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    stream = make_stream(duration=None, time_base=1)
    install_fake_av(monkeypatch, make_container([stream], []))
    capture = VideoFileCapture(path)
    try:
        assert capture.duration is None
    finally:
        capture.close()


@pytest.mark.parametrize("prop", ["width", "height", "average_rate", "duration"])
def test_properties_raise_when_closed(tmp_path, monkeypatch, prop):
    path = make_file(tmp_path)
    install_fake_av(monkeypatch, make_container([make_stream()], []))
    capture = VideoFileCapture(path)
    capture.close()
    with pytest.raises(VideoCaptureError):
        getattr(capture, prop)


def test_read_returns_frames_then_none(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    frames = [FakeFrame(), FakeFrame()]
    install_fake_av(monkeypatch, make_container([make_stream()], frames))
    capture = VideoFileCapture(path)
    try:
        assert capture.read() is frames[0]
        assert capture.read() is frames[1]
        assert capture.read() is None
    finally:
        capture.close()


def test_frames_iterates_all(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    frames = [FakeFrame(), FakeFrame(), FakeFrame()]
    install_fake_av(monkeypatch, make_container([make_stream()], frames))
    capture = VideoFileCapture(path)
    try:
        assert list(capture.frames()) == frames
    finally:
        capture.close()


def test_iter_returns_frames_iterator(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    frames = [FakeFrame()]
    install_fake_av(monkeypatch, make_container([make_stream()], frames))
    capture = VideoFileCapture(path)
    try:
        assert list(capture) == frames
    finally:
        capture.close()


def test_read_when_closed_raises(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    install_fake_av(monkeypatch, make_container([make_stream()], []))
    capture = VideoFileCapture(path)
    capture.close()
    with pytest.raises(VideoCaptureError):
        capture.read()


def test_frames_stops_when_closed(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    frames = [FakeFrame(), FakeFrame()]
    install_fake_av(monkeypatch, make_container([make_stream()], frames))
    capture = VideoFileCapture(path)
    try:
        gen = capture.frames()
        next(gen)
        capture.close()
        with pytest.raises(StopIteration):
            next(gen)
    finally:
        capture.close()


def test_decode_error_wrapped(tmp_path, monkeypatch):
    path = make_file(tmp_path)

    def bad_iter():
        raise RuntimeError("decode failure")
        yield  # pragma: no cover

    container = MagicMock()
    container.streams.video = [make_stream()]
    container.decode = MagicMock(return_value=bad_iter())

    import rtp_live_streaming.sender.capture.video as module

    fake_av = FakeAVModule(container)
    monkeypatch.setattr(
        module.VideoFileCapture, "_import_av", staticmethod(lambda: fake_av)
    )
    capture = VideoFileCapture(path)
    try:
        with pytest.raises(VideoCaptureError):
            capture.read()
    finally:
        capture.close()


def test_loop_reopens_on_eof(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    frame = FakeFrame()
    container1 = make_container([make_stream()], [frame])
    container2 = make_container([make_stream()], [frame])

    import rtp_live_streaming.sender.capture.video as module

    fake_av = FakeAVModule(container1)
    fake_av.open = MagicMock(side_effect=[container1, container2])
    monkeypatch.setattr(
        module.VideoFileCapture, "_import_av", staticmethod(lambda: fake_av)
    )

    capture = VideoFileCapture(path, loop=True)
    try:
        assert capture.read() is frame
        assert capture.read() is frame
    finally:
        capture.close()


def test_loop_empty_stream_does_not_loop_forever(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    install_fake_av(monkeypatch, make_container([make_stream()], []))
    capture = VideoFileCapture(path, loop=True)
    try:
        assert capture.read() is None
    finally:
        capture.close()


def test_loop_decode_error_on_reopen_wrapped(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    empty_container = make_container([make_stream()], [])

    def bad_iter():
        raise RuntimeError("bad")
        yield  # pragma: no cover

    bad_container = MagicMock()
    bad_container.streams.video = [make_stream()]
    bad_container.decode = MagicMock(return_value=bad_iter())

    import rtp_live_streaming.sender.capture.video as module

    fake_av = FakeAVModule(empty_container)
    fake_av.open = MagicMock(side_effect=[empty_container, bad_container])
    monkeypatch.setattr(
        module.VideoFileCapture, "_import_av", staticmethod(lambda: fake_av)
    )

    capture = VideoFileCapture(path, loop=True)
    try:
        with pytest.raises(VideoCaptureError):
            capture.read()
    finally:
        capture.close()


def test_realtime_pacing_first_frame_no_sleep(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    frame = FakeFrame(pts=0, time_base=1)
    install_fake_av(monkeypatch, make_container([make_stream()], [frame]))

    import rtp_live_streaming.sender.capture.video as module

    sleeps = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)

    capture = VideoFileCapture(path, realtime=True)
    try:
        assert capture.read() is frame
        assert sleeps == []
    finally:
        capture.close()


def test_realtime_pacing_second_frame_sleeps(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    frames = [
        FakeFrame(pts=0, time_base=1),
        FakeFrame(pts=1, time_base=1),
    ]
    install_fake_av(monkeypatch, make_container([make_stream()], frames))

    import rtp_live_streaming.sender.capture.video as module

    sleeps = []
    times = iter([100.0, 100.1])
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))

    capture = VideoFileCapture(path, realtime=True)
    try:
        capture.read()
        capture.read()
        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(0.9)
    finally:
        capture.close()


def test_realtime_pacing_no_sleep_when_ahead(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    frames = [
        FakeFrame(pts=0, time_base=1),
        FakeFrame(pts=1, time_base=1),
    ]
    install_fake_av(monkeypatch, make_container([make_stream()], frames))

    import rtp_live_streaming.sender.capture.video as module

    sleeps = []
    times = iter([100.0, 105.0])
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))

    capture = VideoFileCapture(path, realtime=True)
    try:
        capture.read()
        capture.read()
        assert sleeps == []
    finally:
        capture.close()


def test_media_time_falls_back_to_average_rate(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    from fractions import Fraction

    stream = make_stream(average_rate=Fraction(10, 1))
    frames = [FakeFrame(pts=None, time_base=None), FakeFrame(pts=None, time_base=None)]
    install_fake_av(monkeypatch, make_container([stream], frames))

    import rtp_live_streaming.sender.capture.video as module

    sleeps = []
    times = iter([100.0, 100.0])
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))

    capture = VideoFileCapture(path, realtime=True)
    try:
        capture.read()
        capture.read()
        assert sleeps == [pytest.approx(0.1)]
    finally:
        capture.close()


def test_media_time_no_timestamps_no_rate_returns_zero(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    stream = make_stream(average_rate=None)
    frames = [FakeFrame(pts=None, time_base=None), FakeFrame(pts=None, time_base=None)]
    install_fake_av(monkeypatch, make_container([stream], frames))

    capture = VideoFileCapture(path, realtime=True)
    try:
        assert capture._media_time(frames[0]) == 0.0
    finally:
        capture.close()


def test_close_is_idempotent(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    container = make_container([make_stream()], [])
    install_fake_av(monkeypatch, container)
    capture = VideoFileCapture(path)
    capture.close()
    capture.close()
    assert capture.is_closed is True
    assert container.close.call_count == 1


def test_close_failure_wrapped(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    container = make_container([make_stream()], [])
    container.close = MagicMock(side_effect=RuntimeError("close failed"))
    install_fake_av(monkeypatch, container)
    capture = VideoFileCapture(path)
    with pytest.raises(VideoCaptureError):
        capture.close()
    assert capture.is_closed is True


def test_context_manager_closes(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    container = make_container([make_stream()], [])
    install_fake_av(monkeypatch, container)
    with VideoFileCapture(path) as capture:
        assert capture.is_closed is False
    assert capture.is_closed is True
    assert container.close.call_count == 1


def test_context_manager_enter_when_closed_raises(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    install_fake_av(monkeypatch, make_container([make_stream()], []))
    capture = VideoFileCapture(path)
    capture.close()
    with pytest.raises(VideoCaptureError):
        capture.__enter__()
