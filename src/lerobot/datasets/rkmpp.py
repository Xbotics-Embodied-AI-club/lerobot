"""瑞芯微 MPP 硬件 H.264 编码（Xbotics）：录数据时用 RK3566 / RK3576 这类板子芯片自带的编码器。

PyAV 自带的 FFmpeg 不含 rkmpp，所以经 GStreamer：RGB 原图从标准输入喂给
``gst-launch-1.0 fdsrc ! rawvideoparse ! mpph264enc ! h264parse ! mp4mux ! filesink``。
板子上的实测（两路 640×480，30 fps）：软件 libsvtav1 约 11 fps，MPP 29 fps。

两条要点：

- ``GST_MPP_NO_RGA=1``：插件默认借 RGA 把普通内存里的画面搬给编码器，而板子有一段物理内存落在 4 GB 以上，
  RGA 碰到那段页就写坏内存、整机挂死（内核报 ``RGA_MMU unsupported memory larger than 4G``）。
  编码器本身直接吃 RGB，不需要 RGA。
- 码率控制用固定 QP（``rc-mode=fixqp``），QP 取编码配置里的 ``crf``：默认 30，与 LeRobot 默认 crf 30 对齐。

板子上的默认行为（见 :meth:`lerobot.configs.video.VideoEncoderConfig.resolve_vcodec`）：RGB 相机默认的
libsvtav1 自动换成 MPP；设环境变量 ``LEROBOT_RKMPP=0`` 可关掉这个切换。
"""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np

from lerobot.configs.video import RKMPP_VCODEC

__all__ = [
    "RKMPP_VCODEC",
    "RkmppH264Writer",
    "default_stats_every",
    "gst_command",
    "gst_env",
    "rkmpp_available",
]

# 弱 CPU 板子上，边录边编时逐帧算图像统计是录制的瓶颈（RK3566 上每路每帧约 15 ms）。
# 每 10 帧算一次，均值差 ≤0.00025、q99 差 ≤0.012，录制从 13–22 fps 回到相机上限 29 fps。
ARM_STATS_EVERY = 10
_ARM_MACHINES = ("aarch64", "arm64", "armv7l")


def is_arm() -> bool:
    return platform.machine().lower() in _ARM_MACHINES


def default_stats_every() -> int:
    """边录边编时每几帧算一次图像统计：ARM 板子 10，其余 1（逐帧，同上游）。"""
    return ARM_STATS_EVERY if is_arm() else 1


@functools.cache
def rkmpp_available() -> bool:
    """当前机器能否用 MPP 编码：瑞芯微芯片 + gst-launch-1.0 + GStreamer 的 mpph264enc 插件。"""
    if not is_arm() or shutil.which("gst-launch-1.0") is None or shutil.which("gst-inspect-1.0") is None:
        return False
    try:
        compatible = Path("/proc/device-tree/compatible").read_bytes()
    except OSError:
        return False
    if b"rockchip" not in compatible:
        return False
    probe = subprocess.run(["gst-inspect-1.0", "mpph264enc"], capture_output=True, timeout=30, check=False)
    return probe.returncode == 0


def gst_command(video_path: str | Path, width: int, height: int, fps: int, qp: int, gop: int) -> list[str]:
    """把标准输入里的 RGB 原图编成 H.264 MP4 的 GStreamer 命令。

    每个词单独一个参数：gst-launch 不会把含空格的参数拆开，整段 ``rawvideoparse width=…`` 当一个参数会报
    ``erroneous pipeline: syntax error``。
    """
    pipeline = (
        "fdsrc fd=0 ! "
        f"rawvideoparse width={width} height={height} format=rgb framerate={fps}/1 ! "
        f"mpph264enc rc-mode=fixqp qp-init={qp} gop={gop} ! h264parse ! mp4mux ! filesink"
    )
    return ["gst-launch-1.0", "-q", *pipeline.split(), f"location={video_path}"]


def gst_env() -> dict[str, str]:
    return {**os.environ, "GST_MPP_NO_RGA": "1"}


class RkmppH264Writer:
    """一个 MP4 文件的 MPP 编码会话：逐帧 :meth:`write` RGB 原图，:meth:`close` 收尾并检查产物。"""

    def __init__(self, video_path: str | Path, width: int, height: int, fps: int, qp: int, gop: int):
        self.video_path = Path(video_path)
        self.shape = (height, width, 3)
        # gst-launch 把各参数拼起来再解析管道，不经 shell
        self._proc = subprocess.Popen(  # nosec B603
            gst_command(self.video_path, width, height, fps, qp, gop),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=gst_env(),
        )

    def write(self, rgb: np.ndarray) -> None:
        if rgb.shape != self.shape or rgb.dtype != np.uint8:
            raise ValueError(f"MPP 编码要 {self.shape} uint8 RGB，收到 {rgb.shape} {rgb.dtype}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(rgb).data)

    def close(self) -> None:
        # communicate 自己关标准输入（EOF 让 gst 收尾、写完 mp4 的 moov），再等进程退出
        _, stderr = self._proc.communicate(timeout=120)
        if self._proc.returncode != 0 or not self.video_path.exists() or self.video_path.stat().st_size == 0:
            raise OSError(
                f"MPP 编码失败（退出码 {self._proc.returncode}）：{self.video_path}\n{stderr.decode(errors='replace')}"
            )

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
