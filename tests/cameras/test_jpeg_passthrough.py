"""相机原始 JPEG 直通网页预览（Xbotics）：OpenCV 交出原始 MJPG 时相机自己解码并登记原始 JPEG，
预览按同一个数组对象查回它、原样转发，不再自己压图。"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from lerobot.cameras import jpeg_passthrough
from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils import rerun_web_visualization as web

W, H = 64, 48


def _jpeg(value: int) -> bytes:
    bgr = np.zeros((H, W, 3), dtype=np.uint8)
    bgr[:, : W // 2] = (value, 0, 255 - value)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


class RawMjpgCapture:
    """顶替 cv2.VideoCapture：像关掉 CONVERT_RGB 的 V4L2 MJPG 相机那样，read() 给 1×N 的原始 JPEG。"""

    props: dict[int, float] = {}

    def __init__(self, *args, **kwargs):
        self.n = 0
        RawMjpgCapture.props = {}

    def isOpened(self):  # noqa: N802
        return True

    def getBackendName(self):  # noqa: N802
        return "V4L2"

    def set(self, prop, value):
        RawMjpgCapture.props[prop] = value
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: W, cv2.CAP_PROP_FRAME_HEIGHT: H, cv2.CAP_PROP_FPS: 30}.get(prop, 0)

    def read(self):
        self.n += 1
        return True, np.frombuffer(_jpeg((self.n * 40) % 256), np.uint8).reshape(1, -1)

    def release(self):
        pass


@pytest.fixture(autouse=True)
def fresh_registry():
    jpeg_passthrough.clear()
    yield
    jpeg_passthrough.clear()


def _camera(**kwargs) -> OpenCVCamera:
    config = OpenCVCameraConfig(index_or_path="/dev/fake", width=W, height=H, fps=30, warmup_s=0, **kwargs)
    return OpenCVCamera(config)


def test_raw_mjpg_frame_is_decoded_and_its_jpeg_kept():
    with patch(f"{OpenCVCamera.__module__}.cv2.VideoCapture", new=RawMjpgCapture):
        camera = _camera(fourcc="MJPG")
        camera.connect(warmup=False)
        try:
            frame = camera.async_read(timeout_ms=2000)
            assert frame.shape == (H, W, 3)
            jpeg = jpeg_passthrough.jpeg_of(frame)
            assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
            decoded = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            np.testing.assert_array_equal(decoded, frame)
        finally:
            camera.disconnect()
    assert RawMjpgCapture.props[cv2.CAP_PROP_CONVERT_RGB] == 0


def test_rotated_frames_are_not_registered():
    with patch(f"{OpenCVCamera.__module__}.cv2.VideoCapture", new=RawMjpgCapture):
        camera = _camera(fourcc="MJPG", rotation=Cv2Rotation.ROTATE_180)
        camera.connect(warmup=False)
        try:
            assert jpeg_passthrough.jpeg_of(camera.async_read(timeout_ms=2000)) is None
        finally:
            camera.disconnect()


def test_lookup_is_by_object_identity():
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    jpeg_passthrough.remember(frame, b"\xff\xd8jpeg")
    assert jpeg_passthrough.jpeg_of(frame) == b"\xff\xd8jpeg"
    assert jpeg_passthrough.jpeg_of(frame.copy()) is None  # 内容一样、不是同一帧：不能冒用


def test_registry_keeps_only_recent_frames():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(jpeg_passthrough._MAX_ENTRIES + 3)]
    for i, f in enumerate(frames):
        jpeg_passthrough.remember(f, bytes([i]))
    assert jpeg_passthrough.jpeg_of(frames[0]) is None
    assert jpeg_passthrough.jpeg_of(frames[-1]) == bytes([len(frames) - 1])


def test_preview_forwards_the_camera_jpeg_instead_of_copying():
    camera_frame = np.zeros((H, W, 3), dtype=np.uint8)
    other_frame = np.ones((H, W, 3), dtype=np.uint8)
    jpeg_passthrough.remember(camera_frame, b"\xff\xd8raw")
    item = web.snapshot({"top": camera_frame, "wrist": other_frame}, {})
    assert item.jpegs == {"observation.top": b"\xff\xd8raw"}
    assert list(item.images) == ["observation.wrist"]
