"""相机采集放在单独的进程里（Xbotics）：共享内存环形缓冲、采集进程的连接与报错、OpenCVCamera 的接口不变。

CI 上没有相机，拿一段生成的小视频文件当相机（采集进程读到结尾从头再来，按帧率放慢），
用 ``LEROBOT_CAMERA_PROCESS=1`` 强制走采集进程。
"""

import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from lerobot.cameras import jpeg_passthrough  # noqa: E402
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig  # noqa: E402
from lerobot.cameras.opencv.capture_process import SLOTS, FrameRing  # noqa: E402
from lerobot.utils.errors import DeviceNotConnectedError  # noqa: E402

W, H, FPS = 64, 48, 30


@pytest.fixture
def video(tmp_path) -> Path:
    """60 帧、每帧灰度不同的小视频（MJPEG 编码，逐帧解码后的灰度差得开）。"""
    path = tmp_path / "camera.avi"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mjpeg", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuvj420p"
        for i in range(60):
            rgb = np.full((H, W, 3), (i * 4) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuvj420p")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


@pytest.fixture
def forced(monkeypatch):
    monkeypatch.setenv("LEROBOT_CAMERA_PROCESS", "1")


def _camera(path, **kwargs) -> OpenCVCamera:
    cfg = {"width": W, "height": H, "fps": FPS, "warmup_s": 0.5, **kwargs}
    return OpenCVCamera(OpenCVCameraConfig(index_or_path=path, **cfg))


def test_frame_ring_round_trip_and_overwrite_detection():
    shape, cap = (4, 5, 3), 16
    shm = shared_memory.SharedMemory(create=True, size=FrameRing.size(shape, cap))
    try:
        ring = FrameRing(shm, shape, cap)
        ring._seq[0] = 0
        assert ring.seq == 0
        ring.write(np.full(shape, 7, np.uint8), b"\xff\xd8ab", timestamp=1.5)
        frame, jpeg, ts = ring.read(1)
        assert ring.seq == 1 and int(frame[0, 0, 0]) == 7 and jpeg == b"\xff\xd8ab" and ts == 1.5
        ring.write(np.zeros(shape, np.uint8), b"x" * (cap + 1), timestamp=2.0)  # JPEG 太大：不带 JPEG
        assert ring.read(2)[1] is None
        # 写的一方又往前走了一整圈：旧序号的槽位已被覆盖，读出来不算数
        for k in range(SLOTS):
            ring.write(np.full(shape, k, np.uint8), None, timestamp=3.0 + k)
        assert ring.read(2) is None
        del ring
    finally:
        shm.close()
        shm.unlink()


def test_latest_returns_the_same_array_until_a_new_frame():
    """同一帧反复取返回同一个数组（网页预览按对象查原始 JPEG）；来了新帧才换。"""
    from lerobot.cameras.opencv.capture_process import CaptureProcess

    shape, cap = (4, 5, 3), 16
    shm = shared_memory.SharedMemory(create=True, size=FrameRing.size(shape, cap))
    try:
        ring = FrameRing(shm, shape, cap)
        ring._seq[0] = 0
        reader = CaptureProcess.__new__(CaptureProcess)
        reader.ring, reader._cache = ring, None
        assert reader.latest() is None
        ring.write(np.full(shape, 1, np.uint8), b"\xff\xd8a", timestamp=1.0)
        first, again = reader.latest(), reader.latest()
        assert first[0] is again[0] and first[1] == b"\xff\xd8a"
        ring.write(np.full(shape, 2, np.uint8), None, timestamp=2.0)
        newer = reader.latest()
        assert newer[0] is not first[0] and int(newer[0][0, 0, 0]) == 2
        del reader, ring, first, again, newer
    finally:
        shm.close()
        shm.unlink()


def test_capture_process_serves_frames_through_the_camera_api(video, forced):
    camera = _camera(video)
    camera.connect()
    try:
        assert camera._capture is not None and camera.thread is None
        first = camera.async_read(timeout_ms=2000)
        assert first.shape == (H, W, 3) and first.dtype == np.uint8
        second = camera.async_read(timeout_ms=2000)
        assert second is not first
        latest = camera.read_latest()
        assert latest.shape == (H, W, 3)
        assert jpeg_passthrough.jpeg_of(latest) is None  # 视频文件给的是解好的图，没有原始 JPEG
        shm_name = camera._capture.shm.name
    finally:
        camera.disconnect()
    assert not camera.is_connected
    assert not (Path("/dev/shm") / shm_name).exists()


def test_capture_process_reports_open_failure(tmp_path, forced):
    camera = _camera(tmp_path / "missing.avi")
    with pytest.raises(ConnectionError):
        camera.connect()
    assert not camera.is_connected


def test_capture_process_reports_wrong_size(video, forced):
    camera = _camera(video, width=W * 2)
    with pytest.raises(RuntimeError, match="capture_width"):
        camera.connect()
    assert not camera.is_connected


def test_dead_capture_process_means_disconnected(video, forced):
    camera = _camera(video)
    camera.connect()
    camera._capture._proc.kill()
    camera._capture._proc.wait()
    time.sleep(0.1)
    assert not camera.is_connected
    with pytest.raises(DeviceNotConnectedError):
        camera.read_latest()
    camera._cleanup_resources()


def test_not_used_without_size_or_when_forced_off(video, monkeypatch):
    monkeypatch.setenv("LEROBOT_CAMERA_PROCESS", "1")
    assert not OpenCVCamera(OpenCVCameraConfig(index_or_path=video))._wants_capture_process()
    monkeypatch.setenv("LEROBOT_CAMERA_PROCESS", "0")
    assert not _camera(video)._wants_capture_process()
