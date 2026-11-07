"""相机采集放在单独的进程里（Xbotics）：读相机、解码、转色彩都在采集进程里做，画面经共享内存交给录制进程。

端侧计算平台上录数据，每路相机一个读取线程时，录制主线程读舵机、写编码管道都会被拖住：同一进程里的线程
来回拿放 GIL，读舵机的线程要隔几毫秒才能接着读。实测（两路相机、MPP 编码、从臂真读）：去掉相机线程后，
一分钟的 slower 警告从 4 条降到 0 条（只剩第一拍），写编码管道的 p95 从 7.3 ms 降到 2.1 ms。

三部分：

- :class:`FrameRing`：共享内存里的环形缓冲（默认 4 个槽位），每个槽位放一帧画面和它的原始 JPEG；
  写的一方写完一个槽位再把序号加一，读的一方按序号取槽位、拷出来后再核对一次序号（没被覆盖才算数）。
- 本文件直接运行时是采集进程：只导入标准库、numpy 与 cv2（不导入 lerobot 包，端侧计算平台上 lerobot 一导入就要十几秒），
  按参数打开相机、照 :class:`OpenCVCamera` 的规则设置并核对尺寸与帧率，在标准输出回一行 ``ready`` 或 ``error``，然后一直采集。
  标准输入读到 EOF（录制进程退出）就结束。
- :class:`CaptureProcess`：录制进程这一侧，起采集进程、等它就绪、从共享内存取帧。
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

SLOTS = 4
_HEADER = 8  # int64 序号：已写完的最新一帧（0 = 还没有）


class FrameRing:
    """共享内存里的环形缓冲：槽位 = 画面（H×W×C uint8）+ 原始 JPEG（不超过 jpeg_capacity 字节）+ 时间戳 + JPEG 长度。"""

    def __init__(self, shm: shared_memory.SharedMemory, shape: tuple[int, int, int], jpeg_capacity: int):
        self.shm = shm
        self.shape = shape
        self.jpeg_capacity = jpeg_capacity
        frame_bytes = int(np.prod(shape))
        buf = shm.buf
        self._seq = np.ndarray((1,), dtype=np.int64, buffer=buf, offset=0)
        meta_offset = _HEADER
        self._ts = np.ndarray((SLOTS,), dtype=np.float64, buffer=buf, offset=meta_offset)
        self._jpeg_len = np.ndarray((SLOTS,), dtype=np.int64, buffer=buf, offset=meta_offset + 8 * SLOTS)
        frames_offset = meta_offset + 16 * SLOTS
        self._frames = np.ndarray((SLOTS, *shape), dtype=np.uint8, buffer=buf, offset=frames_offset)
        jpeg_offset = frames_offset + SLOTS * frame_bytes
        self._jpegs = np.ndarray((SLOTS, jpeg_capacity), dtype=np.uint8, buffer=buf, offset=jpeg_offset)

    @staticmethod
    def size(shape: tuple[int, int, int], jpeg_capacity: int) -> int:
        return _HEADER + 16 * SLOTS + SLOTS * int(np.prod(shape)) + SLOTS * jpeg_capacity

    @property
    def seq(self) -> int:
        return int(self._seq[0])

    def write(self, frame: np.ndarray, jpeg: bytes | None, timestamp: float) -> None:
        seq = self.seq + 1
        slot = seq % SLOTS
        self._frames[slot] = frame
        if jpeg is not None and len(jpeg) <= self.jpeg_capacity:
            self._jpegs[slot, : len(jpeg)] = np.frombuffer(jpeg, np.uint8)
            self._jpeg_len[slot] = len(jpeg)
        else:
            self._jpeg_len[slot] = 0
        self._ts[slot] = timestamp
        self._seq[0] = seq  # 最后才发布：读的一方看到这个序号时，槽位已经写完

    def read(self, seq: int) -> tuple[np.ndarray, bytes | None, float] | None:
        """拷出第 seq 帧；拷的过程中被新帧覆盖了（写的一方又往前走了一整圈）就返回 None。"""
        slot = seq % SLOTS
        frame = self._frames[slot].copy()
        length = int(self._jpeg_len[slot])
        jpeg = self._jpegs[slot, :length].tobytes() if length else None
        timestamp = float(self._ts[slot])
        if self.seq - seq >= SLOTS - 1:
            return None
        return frame, jpeg, timestamp


# ---------------------------------------------------------------- 采集进程（本文件直接运行）


def _fourcc_text(code: float) -> str:
    value = int(code)
    return "".join(chr((value >> 8 * i) & 0xFF) for i in range(4))


def _configure(cap: Any, cfg: dict) -> dict:
    """照 OpenCVCamera._configure_capture_settings 设置并核对；设不上时抛 RuntimeError（同原来的报错）。"""
    import cv2

    if cfg["fourcc"] is not None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
        actual = _fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        if actual != cfg["fourcc"]:
            print(
                f"warning: failed to set fourcc={cfg['fourcc']} (actual={actual})",
                file=sys.stderr,
                flush=True,
            )
    width, height = cfg["capture_width"], cfg["capture_height"]
    if width is None or height is None:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    elif (int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))) != (
        width,
        height,
    ):
        # 已经是要求的尺寸就不再设（视频文件当相机时设不了，测试用）；否则照 OpenCVCamera 设并核对
        ok_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        ok_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        actual_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not ok_w or actual_w != width:
            raise RuntimeError(f"failed to set capture_width={width} ({actual_w=}, width_success={ok_w}).")
        if not ok_h or actual_h != height:
            raise RuntimeError(f"failed to set capture_height={height} ({actual_h=}, height_success={ok_h}).")
    fps = cfg["fps"]
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    elif not math.isclose(fps, cap.get(cv2.CAP_PROP_FPS), rel_tol=1e-3):
        ok = cap.set(cv2.CAP_PROP_FPS, float(fps))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if not ok or not math.isclose(fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(f"failed to set fps={fps} ({actual_fps=}).")
    if cfg["fourcc"] == "MJPG" and cap.getBackendName() == "V4L2":
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # 交出原始 JPEG，自己解码，原始 JPEG 留给网页预览
    return {"capture_width": width, "capture_height": height, "fps": fps}


def _postprocess(frame: np.ndarray, cfg: dict, size: dict) -> np.ndarray:
    """照 OpenCVCamera._postprocess_image：核对尺寸与通道、按需转 RGB、旋转。"""
    import cv2

    h, w, c = frame.shape
    if h != size["capture_height"] or w != size["capture_width"]:
        raise RuntimeError(
            f"frame width={w} or height={h} do not match configured width={size['capture_width']} "
            f"or height={size['capture_height']}."
        )
    if c != 3:
        raise RuntimeError(f"frame channels={c} do not match expected 3 channels (RGB/BGR).")
    if cfg["color_mode"] == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if cfg["rotation"] is not None:
        frame = cv2.rotate(frame, cfg["rotation"])
    return frame


def _watch_parent() -> None:
    """录制进程退出（标准输入读到 EOF）时结束采集进程。"""
    sys.stdin.buffer.read()
    os._exit(0)


def capture_main(cfg: dict) -> None:
    import cv2

    cv2.setNumThreads(1)
    threading.Thread(target=_watch_parent, daemon=True).start()

    def reply(message: dict) -> None:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()

    cap = cv2.VideoCapture(cfg["index_or_path"], cfg["backend"])
    if not cap.isOpened():
        reply({"status": "connection_error"})
        return
    try:
        size = _configure(cap, cfg)
    except RuntimeError as e:
        reply({"status": "runtime_error", "message": str(e)})
        return

    out_h, out_w = size["capture_height"], size["capture_width"]
    if cfg["rotation"] in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        out_h, out_w = out_w, out_h
    shm = shared_memory.SharedMemory(name=cfg["shm_name"])
    # 共享内存归录制进程管（它负责 unlink）；采集进程只借用，不让本进程的 resource_tracker 去清理
    from multiprocessing import resource_tracker

    resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    ring = FrameRing(shm, (out_h, out_w, 3), cfg["jpeg_capacity"])
    reply({"status": "ready", **size})

    is_file = Path(str(cfg["index_or_path"])).is_file()
    failures = 0
    while True:
        ret, frame = cap.read()
        if not ret and is_file:
            # 拿视频文件当相机（测试用）：读到结尾就从头再来，并按帧率放慢
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        try:
            if not ret:
                raise RuntimeError(f"read failed (status={ret}).")
            jpeg = None
            if frame.ndim == 1 or (frame.ndim == 2 and frame.shape[0] == 1):  # 原始 JPEG
                jpeg = frame.tobytes()
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"could not decode the MJPG frame ({len(jpeg)} bytes).")
            frame = _postprocess(frame, cfg, size)
            ring.write(frame, jpeg if cfg["rotation"] is None else None, time.perf_counter())
            failures = 0
        except RuntimeError as e:
            failures += 1
            print(f"warning: error reading frame: {e}", file=sys.stderr, flush=True)
            if failures > 10:
                print("error: exceeded maximum consecutive read failures.", file=sys.stderr, flush=True)
                return
        if is_file and size["fps"]:
            time.sleep(1.0 / float(size["fps"]))


# ---------------------------------------------------------------- 录制进程这一侧


class CaptureProcess:
    """起一路相机的采集进程，等它就绪，从共享内存取帧。配置里必须给出采集的宽、高（共享内存按它开）。"""

    def __init__(self, cfg: dict, ready_timeout_s: float = 30.0):
        self.cfg = cfg
        frame_h, frame_w = int(cfg["capture_height"]), int(cfg["capture_width"])
        self.shm: shared_memory.SharedMemory | None = None
        self.ring: FrameRing | None = None
        self._last_seq = 0
        self._cache: tuple[int, np.ndarray, bytes | None, float] | None = None
        self._proc: subprocess.Popen | None = None
        self._start(frame_h, frame_w, ready_timeout_s)

    def _start(self, frame_h: int, frame_w: int, ready_timeout_s: float) -> None:
        import cv2

        out_h, out_w = frame_h, frame_w
        if self.cfg["rotation"] in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            out_h, out_w = out_w, out_h
        jpeg_capacity = frame_h * frame_w * 3
        shape = (out_h, out_w, 3)
        self.shm = shared_memory.SharedMemory(create=True, size=FrameRing.size(shape, jpeg_capacity))
        np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf)[0] = 0
        args = {**self.cfg, "shm_name": self.shm.name, "jpeg_capacity": jpeg_capacity}
        self._proc = subprocess.Popen(  # nosec B603 —— 本包自带的脚本
            [sys.executable, str(Path(__file__)), json.dumps(args)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._proc.stdout is not None
        line = self._read_line(ready_timeout_s)
        message = json.loads(line) if line else {"status": "connection_error"}
        status = message["status"]
        if status == "connection_error":
            self.close()
            raise ConnectionError("failed to open the camera")
        if status == "runtime_error":
            self.close()
            raise RuntimeError(message["message"])
        self.fps = message["fps"]
        self.ring = FrameRing(self.shm, shape, jpeg_capacity)

    def _read_line(self, timeout_s: float) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        result: list[str] = []
        reader = threading.Thread(target=lambda: result.append(self._proc.stdout.readline()), daemon=True)
        reader.start()
        reader.join(timeout_s)
        return result[0].strip() if result else ""

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def seq(self) -> int:
        return self.ring.seq if self.ring is not None else 0

    def latest(self) -> tuple[np.ndarray, bytes | None, float] | None:
        """最新一帧（画面、原始 JPEG、采集时刻）；还没有帧返回 None。同一帧反复取返回同一个数组。"""
        assert self.ring is not None
        while True:
            seq = self.ring.seq
            if seq == 0:
                return None
            if self._cache is not None and self._cache[0] == seq:
                return self._cache[1:]
            got = self.ring.read(seq)
            if got is not None:
                self._cache = (seq, *got)
                return got

    def wait_new(self, timeout_s: float) -> tuple[np.ndarray, bytes | None, float] | None:
        """等一帧上次取走之后的新画面；超时返回 None。"""
        deadline = time.perf_counter() + timeout_s
        while self.seq <= self._last_seq:
            if time.perf_counter() > deadline or not self.alive:
                return None
            time.sleep(0.001)
        self._last_seq = self.seq
        return self.latest()

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.stdin is not None:
                with contextlib.suppress(OSError):
                    self._proc.stdin.close()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        self.ring = None
        self._cache = None
        if self.shm is not None:
            self.shm.close()
            with contextlib.suppress(FileNotFoundError):
                self.shm.unlink()
            self.shm = None


if __name__ == "__main__":
    capture_main(json.loads(sys.argv[1]))
