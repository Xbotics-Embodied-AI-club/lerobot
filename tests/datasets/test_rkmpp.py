"""MPP 硬件编码（h264_rkmpp）与边录边编的统计抽样（Xbotics）。

真机上 MPP 走 GStreamer 的 mpph264enc，CI 上没有这套东西：编码器的选择逻辑与 GStreamer 命令单测，
两条编码路径（录完再编 / 边录边编）用假的写入器替掉子进程，只校验喂进去的帧与收尾。
"""

import queue
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

from lerobot.configs import VALID_VIDEO_CODECS, DepthEncoderConfig, RGBEncoderConfig  # noqa: E402
from lerobot.datasets import rkmpp, video_utils  # noqa: E402
from lerobot.datasets.image_writer import write_image  # noqa: E402

RKMPP = rkmpp.RKMPP_VCODEC


@pytest.fixture
def on_rockchip(monkeypatch):
    """装作在带 MPP 编码器的端侧计算平台上。"""
    monkeypatch.setattr(rkmpp, "rkmpp_available", lambda: True)
    monkeypatch.delenv("LEROBOT_RKMPP", raising=False)


@pytest.fixture
def not_on_rockchip(monkeypatch):
    monkeypatch.setattr(rkmpp, "rkmpp_available", lambda: False)


class FakeWriter:
    """顶替 GStreamer 子进程：记下写入的帧与是否正常收尾。"""

    instances: list["FakeWriter"] = []

    def __init__(self, video_path, width, height, fps, qp, gop):
        self.video_path, self.size, self.fps, self.qp, self.gop = video_path, (width, height), fps, qp, gop
        self.frames: list[np.ndarray] = []
        self.closed = False
        FakeWriter.instances.append(self)

    def write(self, rgb: np.ndarray) -> None:
        assert not self.closed
        self.frames.append(rgb.copy())

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.closed = True


@pytest.fixture
def fake_writer(monkeypatch):
    FakeWriter.instances = []
    monkeypatch.setattr(video_utils, "RkmppH264Writer", FakeWriter)
    return FakeWriter


# ---------------------------------------------------------------- 选编码器


def test_rkmpp_is_a_valid_codec_name():
    assert RKMPP in VALID_VIDEO_CODECS


def test_default_software_av1_switches_to_mpp_on_rockchip(on_rockchip):
    """板子上 libsvtav1 软件编码只有约 11 fps，默认配置在板子上自动改走 MPP。"""
    cfg = RGBEncoderConfig()
    assert cfg.vcodec == RKMPP
    assert cfg.preset is None


def test_auto_prefers_mpp_on_rockchip(on_rockchip):
    assert RGBEncoderConfig(vcodec="auto").vcodec == RKMPP


def test_switch_can_be_turned_off(on_rockchip, monkeypatch):
    monkeypatch.setenv("LEROBOT_RKMPP", "0")
    assert RGBEncoderConfig().vcodec == "libsvtav1"


def test_explicit_other_codec_is_kept_on_rockchip(on_rockchip):
    assert RGBEncoderConfig(vcodec="h264").vcodec == "h264"


def test_depth_encoder_never_switches(on_rockchip):
    assert DepthEncoderConfig().vcodec == "hevc"


def test_default_unchanged_off_rockchip(not_on_rockchip):
    assert RGBEncoderConfig().vcodec == "libsvtav1"


def test_explicit_mpp_without_mpp_raises(not_on_rockchip):
    with pytest.raises(ValueError, match=RKMPP):
        RGBEncoderConfig(vcodec=RKMPP)


def test_mpp_codec_options_map_crf_to_fixed_qp(on_rockchip):
    cfg = RGBEncoderConfig(vcodec=RKMPP, g=2, crf=30)
    assert cfg.get_codec_options(as_strings=True) == {"g": "2", "qp": "30"}


def test_gst_command_and_env():
    cmd = rkmpp.gst_command("/tmp/out.mp4", width=640, height=480, fps=30, qp=30, gop=2)
    pipeline = " ".join(cmd)
    assert cmd[0] == "gst-launch-1.0"
    assert (
        "fdsrc fd=0 blocksize=921600 ! rawvideoparse width=640 height=480 format=rgb framerate=30/1"
        in pipeline
    )
    assert "mpph264enc rc-mode=fixqp qp-init=30 gop=2" in pipeline
    assert "mp4mux trak-timescale=12000" in pipeline
    assert "filesink location=/tmp/out.mp4" in pipeline
    # 板子上有落在 4 GB 以上的物理内存，MPP 插件默认借 RGA 搬帧会写坏内存、整机挂死
    assert rkmpp.gst_env()["GST_MPP_NO_RGA"] == "1"


def _write_episode(path, fps: int, timescale: int, n_frames: int = 6) -> None:
    """按给定时间单位写一段 MP4，模拟 MPP（mp4mux）写出的一条。

    FFmpeg 的 MP4 封装器默认把低于 10000 的时间单位逐次乘 2，``video_track_timescale`` 让它照写不改，
    与 mp4mux 一样。
    """
    import av

    with av.open(str(path), "w", options={"video_track_timescale": str(timescale)}) as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, timescale)
        stream.codec_context.time_base = Fraction(1, fps)
        for i in range(n_frames):
            rgb = np.full((48, 64, 3), i * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _append_three_episodes(tmp_path, fps: int, timescale: int) -> int:
    """照录制时的存法：第一条直接当文件，第二、三条依次拼到它后面；返回拼完的帧数。"""
    import av

    target = tmp_path / "file-000.mp4"
    _write_episode(target, fps, timescale)
    for k in range(2):
        episode = tmp_path / f"ep{k}.mp4"
        _write_episode(episode, fps, timescale)
        video_utils.concatenate_video_files([target, episode], target)
    with av.open(str(target)) as container:
        return sum(1 for p in container.demux(video=0) if p.dts is not None)


@pytest.mark.parametrize("fps", [10, 15, 20, 25, 30, 50, 60])
def test_mpp_episodes_append_three_times_into_one_file(fps, tmp_path):
    """MPP 段的时间单位等于拼接后文件的，同一文件里连拼三条不报时间戳倒退。"""
    assert rkmpp.mp4_timescale(fps) % fps == 0
    assert _append_three_episodes(tmp_path, fps, rkmpp.mp4_timescale(fps)) == 18


def test_mp4mux_default_timescale_breaks_the_third_episode(tmp_path):
    """反例：mp4mux 自动取的 1/3000（30 fps）拼第三条时报错，正是板子上的现场。"""
    with pytest.raises(ValueError):
        _append_three_episodes(tmp_path, 30, 3000)


def test_writer_keeps_gst_output_off_the_terminal(monkeypatch, tmp_path):
    """标准输出是终端时，gst-launch 每秒刷好多行管道位置（``0:00:01.2 / 99:99:99.``），``-q`` 关不掉；
    网页终端里录数据会被它刷屏。"""
    seen = {}

    class FakeProc:
        stdin = None

        def __init__(self, cmd, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(rkmpp.subprocess, "Popen", FakeProc)
    rkmpp.RkmppH264Writer(tmp_path / "out.mp4", width=640, height=480, fps=30, qp=30, gop=2)
    assert seen["stdout"] is subprocess.DEVNULL


@pytest.mark.skipif(not hasattr(__import__("fcntl"), "F_GETPIPE_SZ"), reason="只有 Linux 能调管道大小")
def test_writer_pipe_holds_a_whole_frame(monkeypatch, tmp_path):
    """管道开到 1 MB：640×480 RGB 一帧 900 KB，默认 64 KB 的管道要拆十几次写、编码进程读两百多次。"""
    import fcntl

    monkeypatch.setattr(rkmpp, "gst_command", lambda *a, **k: ["cat"])
    writer = rkmpp.RkmppH264Writer(tmp_path / "out.mp4", width=640, height=480, fps=30, qp=30, gop=2)
    try:
        size = fcntl.fcntl(writer._proc.stdin.fileno(), fcntl.F_GETPIPE_SZ)
        assert size >= 640 * 480 * 3
    finally:
        writer.abort()


# ---------------------------------------------------------------- 两条编码路径


def test_encode_video_frames_feeds_png_frames_to_mpp(on_rockchip, fake_writer, tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(5):
        write_image(np.full((48, 64, 3), i * 40, dtype=np.uint8), imgs / f"frame-{i:06d}.png")
    out = tmp_path / "out.mp4"

    video_utils.encode_video_frames(
        imgs, out, fps=30, video_encoder=RGBEncoderConfig(vcodec=RKMPP, g=2, crf=28)
    )

    (writer,) = fake_writer.instances
    assert writer.size == (64, 48) and writer.qp == 28 and writer.gop == 2
    assert [int(f[0, 0, 0]) for f in writer.frames] == [0, 40, 80, 120, 160]
    assert writer.closed


def _run_encoder_thread(tmp_path, cfg, frames):
    frame_queue: queue.Queue = queue.Queue(maxsize=len(frames) + 1)
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    thread = video_utils._CameraEncoderThread(
        video_path=tmp_path / "cam.mp4",
        fps=30,
        video_encoder=cfg,
        frame_queue=frame_queue,
        result_queue=result_queue,
        stop_event=threading.Event(),
    )
    thread.start()
    for frame in frames:
        frame_queue.put(frame)
    frame_queue.put(None)
    thread.join(timeout=60)
    return result_queue.get(timeout=5)


def test_streaming_encoder_thread_feeds_mpp(on_rockchip, fake_writer, tmp_path):
    frames = [np.full((48, 64, 3), i, dtype=np.uint8) for i in range(12)]
    status, stats = _run_encoder_thread(tmp_path, RGBEncoderConfig(vcodec=RKMPP), frames)

    assert status == "ok" and stats is not None
    (writer,) = fake_writer.instances
    assert len(writer.frames) == 12 and writer.closed


# ---------------------------------------------------------------- 主线程直写（不开编码线程）

KEYS = ["observation.images.top", "observation.images.wrist"]


def _episode_frames(n: int, offset: int = 0) -> list[np.ndarray]:
    return [np.full((48, 64, 3), (i * 23 + offset) % 256, dtype=np.uint8) for i in range(n)]


def _expected_stats(frames: list[np.ndarray], every: int) -> dict:
    from lerobot.datasets.compute_stats import RunningQuantileStats

    tracker = RunningQuantileStats()
    for i, frame in enumerate(frames):
        if i < 2 or i % every == 0:
            tracker.update(video_utils._stats_batch(frame))
    return tracker.get_statistics()


def test_mpp_streaming_writes_from_the_caller_without_threads(on_rockchip, fake_writer, tmp_path):
    """板子上 MPP 编码不开编码线程：送进来的帧按顺序写进编码子进程，统计与同样抽样规则算的逐位相同。"""
    encoder = video_utils.StreamingVideoEncoder(
        fps=30, rgb_encoder=RGBEncoderConfig(vcodec=RKMPP, stats_every=10)
    )
    encoder.start_episode(video_keys=KEYS, temp_dir=tmp_path)
    frames = _episode_frames(25)
    for frame in frames:
        for key in KEYS:
            encoder.feed_frame(key, frame)
    assert encoder._threads == {}
    results = encoder.finish_episode()

    first_two = fake_writer.instances[:2]
    assert [w.closed for w in first_two] == [True, True]
    for writer in first_two:
        assert [int(f[0, 0, 0]) for f in writer.frames] == [int(f[0, 0, 0]) for f in frames]
    expected = _expected_stats(frames, every=10)
    for key in KEYS:
        path, stats = results[key]
        assert path.name == f"{key}_streaming.mp4"
        for name, value in expected.items():
            np.testing.assert_array_equal(stats[name], value)
    encoder.close()


def test_mpp_streaming_prestarts_the_next_episode(on_rockchip, fake_writer, tmp_path):
    """存完一条就按画面尺寸启动下一条的编码子进程；下一条直接用它，第一帧时不再新开。"""
    encoder = video_utils.StreamingVideoEncoder(fps=30, rgb_encoder=RGBEncoderConfig(vcodec=RKMPP))
    encoder.start_episode(video_keys=KEYS, temp_dir=tmp_path)
    for frame in _episode_frames(3):
        for key in KEYS:
            encoder.feed_frame(key, frame)
    first = encoder.finish_episode()
    assert len(fake_writer.instances) == 4  # 两路这一条的 + 两路预先启动的
    spares = fake_writer.instances[2:]
    assert all(w.size == (64, 48) and not w.closed and not w.frames for w in spares)

    encoder.start_episode(video_keys=KEYS, temp_dir=tmp_path)
    for frame in _episode_frames(3, offset=7):
        for key in KEYS:
            encoder.feed_frame(key, frame)
    second = encoder.finish_episode()
    assert len(spares[0].frames) == 3 and len(spares[1].frames) == 3
    for key in KEYS:
        assert second[key][0].parent != first[key][0].parent
    encoder.close()


def test_mpp_streaming_cancel_and_close_clean_up(on_rockchip, fake_writer, tmp_path):
    """重录（取消这一条）停掉写入器、删掉它的临时目录；收尾时还没用上的预先启动的也停掉、删掉。

    存完的那几条的临时目录留给数据集存盘时搬走，这里不动。
    """

    def record_one() -> set:
        encoder.start_episode(video_keys=KEYS, temp_dir=tmp_path)
        for frame in _episode_frames(3):
            for key in KEYS:
                encoder.feed_frame(key, frame)
        return {path.parent for path, _ in encoder.finish_episode().values()}

    encoder = video_utils.StreamingVideoEncoder(fps=30, rgb_encoder=RGBEncoderConfig(vcodec=RKMPP))
    finished_dirs = record_one()
    spares = fake_writer.instances[-2:]
    spare_dirs = {Path(w.video_path).parent for w in spares}
    assert len(spare_dirs) == 2 and all(d.is_dir() for d in spare_dirs)

    # 第二条用上了预先启动的编码子进程，中途取消
    encoder.start_episode(video_keys=KEYS, temp_dir=tmp_path)
    encoder.feed_frame(KEYS[0], _episode_frames(1)[0])
    encoder.cancel_episode()
    assert all(w.closed for w in spares)
    assert not any(d.exists() for d in spare_dirs)

    # 第三条录完又预先启动了两个，收尾时它们被停掉、删掉
    finished_dirs |= record_one()
    last_spares = fake_writer.instances[-2:]
    last_spare_dirs = {Path(w.video_path).parent for w in last_spares}
    encoder.close()
    assert all(w.closed for w in last_spares)
    assert not any(d.exists() for d in last_spare_dirs)
    assert {d for d in tmp_path.iterdir() if d.is_dir()} == finished_dirs


def test_stats_worker_imports_no_lerobot():
    """统计进程按路径加载 RunningQuantileStats，不导入 lerobot 包（端侧计算平台上 lerobot.datasets 一导入就几十秒）。"""
    import subprocess
    import sys

    from lerobot.datasets import stats_worker

    probe = (
        "import runpy, sys; m = runpy.run_path(sys.argv[1], run_name='probe'); m['load_tracker_class'](); "
        "print(sorted(k for k in sys.modules if k == 'lerobot' or k.startswith('lerobot.')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe, stats_worker.__file__], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_stats_process_matches_in_process_stats():
    """统计进程算出的与本进程里依次 update 的逐位相同；两个键互不干扰；出错在取结果时报出来。"""
    from lerobot.datasets.compute_stats import RunningQuantileStats

    rng = np.random.default_rng(0)
    batches = {k: [rng.integers(0, 256, (300, 3), dtype=np.uint8) for _ in range(5)] for k in ("a", "b")}
    stats = video_utils.StatsProcess()
    try:
        for i in range(5):
            for key in ("a", "b"):
                stats.update(key, batches[key][i])
        for key in ("a", "b"):
            expected = RunningQuantileStats()
            for batch in batches[key]:
                expected.update(batch)
            got = stats.get(key)
            for name, value in expected.get_statistics().items():
                np.testing.assert_array_equal(got[name], value)
        stats.update("c", np.zeros((3, 3), dtype=np.uint8))
        stats.update("c", np.zeros((3, 4), dtype=np.uint8))  # 列数不一致，原类会报错
        with pytest.raises(RuntimeError, match="does not match"):
            stats.get("c")
    finally:
        stats.close()


def test_mpp_streaming_writer_error_surfaces_on_feed(on_rockchip, monkeypatch, tmp_path):
    class BrokenWriter(FakeWriter):
        def write(self, rgb):
            raise BrokenPipeError("gst exited")

    monkeypatch.setattr(video_utils, "RkmppH264Writer", BrokenWriter)
    encoder = video_utils.StreamingVideoEncoder(fps=30, rgb_encoder=RGBEncoderConfig(vcodec=RKMPP))
    encoder.start_episode(video_keys=KEYS[:1], temp_dir=tmp_path)
    with pytest.raises(RuntimeError, match="gst exited"):
        encoder.feed_frame(KEYS[0], _episode_frames(1)[0])
    encoder.close()


# ---------------------------------------------------------------- 统计抽样


def test_stats_every_defaults_by_platform(monkeypatch):
    monkeypatch.setattr(rkmpp.platform, "machine", lambda: "aarch64")
    assert RGBEncoderConfig(vcodec="h264").resolved_stats_every() == 10
    monkeypatch.setattr(rkmpp.platform, "machine", lambda: "x86_64")
    assert RGBEncoderConfig(vcodec="h264").resolved_stats_every() == 1
    assert RGBEncoderConfig(vcodec="h264", stats_every=4).resolved_stats_every() == 4


def test_streaming_stats_sample_every_nth_frame(tmp_path):
    """每 N 帧算一次统计（前两帧总算）：结果等于只拿被抽中的帧算的统计。"""
    pytest.importorskip("av")
    from lerobot.datasets.compute_stats import RunningQuantileStats

    frames = [np.full((48, 64, 3), (i * 9) % 256, dtype=np.uint8) for i in range(25)]
    status, stats = _run_encoder_thread(tmp_path, RGBEncoderConfig(vcodec="h264", stats_every=10), frames)
    assert status == "ok"

    expected = RunningQuantileStats()
    for i in (0, 1, 10, 20):
        expected.update(frames[i].reshape(-1, 3))
    np.testing.assert_allclose(stats["mean"], expected.get_statistics()["mean"])
    assert int(stats["count"][0]) == int(expected.get_statistics()["count"][0])
