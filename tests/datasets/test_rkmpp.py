"""瑞芯微 MPP 硬件编码（h264_rkmpp）与边录边编的统计抽样（Xbotics）。

真机上 MPP 走 GStreamer 的 mpph264enc，CI 上没有这套东西：编码器的选择逻辑与 GStreamer 命令单测，
两条编码路径（录完再编 / 边录边编）用假的写入器替掉子进程，只校验喂进去的帧与收尾。
"""

import queue
import subprocess
import threading

import numpy as np
import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

from lerobot.configs import VALID_VIDEO_CODECS, DepthEncoderConfig, RGBEncoderConfig  # noqa: E402
from lerobot.datasets import rkmpp, video_utils  # noqa: E402
from lerobot.datasets.image_writer import write_image  # noqa: E402

RKMPP = rkmpp.RKMPP_VCODEC


@pytest.fixture
def on_rockchip(monkeypatch):
    """装作在接了 MPP 的瑞芯微板子上。"""
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
    assert "rawvideoparse width=640 height=480 format=rgb framerate=30/1" in pipeline
    assert "mpph264enc rc-mode=fixqp qp-init=30 gop=2" in pipeline
    assert "filesink location=/tmp/out.mp4" in pipeline
    # 板子上有落在 4 GB 以上的物理内存，MPP 插件默认借 RGA 搬帧会写坏内存、整机挂死
    assert rkmpp.gst_env()["GST_MPP_NO_RGA"] == "1"


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
