"""板子出网页的 rerun 预览（Xbotics）：录数据时学员只用浏览器看画面，不用装任何东西。

板子没有桌面、也扛不住原生查看器，所以由板子自己出网页，渲染放在学员的浏览器里做：

- 浏览器只需打开 ``http://<板子>:9090``。那是一个跳转页：按浏览器访问板子用的主机名（请求的 Host 头）
  拼出数据地址，跳到 9091 上的 rerun 网页查看器。查看器页自己不知道数据在哪 —— ``serve_grpc``
  回给它的是板子的 127.0.0.1，别的机器上打开就会去连学员自己的电脑。
- rerun 0.3x 的数据服务默认只接受 localhost 页面的跨源请求，别的机器上打开会被浏览器拦下（服务端照样回 200），
  所以放行「任何主机、查看器那个端口」。0.26 没有这个参数，它本来就不拦。
- 推送放进子进程：两张 JPEG 的 ``rr.log`` 在 RK3566 上要十几毫秒，放在录制进程里会和录制抢 GIL；
  主进程每 0.1 秒取一次最新的画面（默认 10 次/秒），后台线程压 JPEG 后交给子进程。
  实测录制 28.7 fps 不受影响，预览 10 次/秒。

注意：没有显卡加速的浏览器（软件光栅）里，rerun 0.33 的网页查看器一收到数据就卡死，0.26.2 正常；
板子环境钉 ``rerun-sdk==0.26.2``（``viz`` 组声明的范围内）。
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import multiprocessing
import os
import queue
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import quote

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package

logger = logging.getLogger(__name__)

DEFAULT_WEB_PORT = 9090  # 跳转页（学员敲的地址）；查看器在 +1
DEFAULT_GRPC_PORT = 9876  # rerun 数据服务
DEFAULT_HZ = 10.0
DEFAULT_JPEG_QUALITY = 80
DEFAULT_MEMORY_LIMIT = "64MB"  # 数据服务为晚连上的浏览器缓冲的上限；命令行版默认是系统内存的 25%


def viewer_url(host_header: str, grpc_port: int, viewer_port: int) -> str:
    """按浏览器访问板子用的主机名，拼出查看器地址（带上数据服务地址）。"""
    host = host_header
    if host.startswith("["):  # IPv6：[addr]:port
        host = host[: host.index("]") + 1]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    data = quote(f"rerun+http://{host}:{grpc_port}/proxy", safe="")
    return f"http://{host}:{viewer_port}/?url={data}"


def cors_kwargs(serve_grpc: Callable[..., Any], viewer_port: int) -> dict[str, list[str]]:
    """rerun 有 ``cors_allow_origin`` 时放行查看器端口的任何主机；没有（0.26）就不传。"""
    if "cors_allow_origin" in inspect.signature(serve_grpc).parameters:
        return {"cors_allow_origin": [f"http://*:{viewer_port}"]}
    return {}


def local_address() -> str:
    """板子对外的地址，只用来打印提示（UDP connect 不发包）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
        except OSError:
            return socket.gethostname()


class LatestSlot:
    """只放最新的一份：发得慢时旧的被新的顶掉；``due`` 按间隔限频。"""

    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self._item: Any = None
        self._last_put: float | None = None
        self._lock = threading.Lock()
        self.ready = threading.Event()

    def due(self, now: float) -> bool:
        return self._last_put is None or now - self._last_put >= self.interval_s

    def put(self, item: Any, now: float) -> None:
        with self._lock:
            self._item, self._last_put = item, now
            self.ready.set()

    def take(self) -> Any:
        with self._lock:
            item, self._item = self._item, None
            self.ready.clear()
            return item


@dataclass
class PreviewItem:
    images: dict[str, np.ndarray] = field(default_factory=dict)
    observation_scalars: dict[str, float] = field(default_factory=dict)
    action_scalars: dict[str, float] = field(default_factory=dict)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) or (isinstance(v, np.ndarray) and v.ndim == 0)


def snapshot(observation: RobotObservation | None, action: RobotAction | None) -> PreviewItem:
    """取一份要推的数据：画面复制（相机驱动可能复用缓冲区），标量转 float，键名同 log_rerun_data。"""
    item = PreviewItem()
    for k, v in (observation or {}).items():
        key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"
        if _is_scalar(v):
            item.observation_scalars[key] = float(v)
        elif isinstance(v, np.ndarray) and v.ndim == 3 and v.dtype == np.uint8 and v.shape[-1] == 3:
            item.images[key] = v.copy()
    for k, v in (action or {}).items():
        key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"
        if _is_scalar(v):
            item.action_scalars[key] = float(v)
    return item


def _serve(jobs: Any, web_port: int, grpc_port: int, memory_limit: str) -> None:
    """子进程：起 rerun 数据服务、网页查看器与跳转页，把收到的 JPEG 与标量推出去。"""
    with contextlib.suppress(OSError):
        os.nice(-5)  # 录制线程余量大；预览让一点优先级，否则 4 个核都忙时它抢不到 CPU
    import rerun as rr

    viewer_port = web_port + 1
    rr.init("recording")
    rr.serve_grpc(
        grpc_port=grpc_port, server_memory_limit=memory_limit, **cors_kwargs(rr.serve_grpc, viewer_port)
    )
    rr.serve_web_viewer(web_port=viewer_port, open_browser=False)

    class ToViewer(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", viewer_url(self.headers.get("Host") or "", grpc_port, viewer_port))
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("0.0.0.0", web_port), ToViewer)  # nosec B104 —— 就是要局域网能连
    threading.Thread(target=server.serve_forever, daemon=True).start()

    while (job := jobs.get()) is not None:
        jpegs, obs_scalars, act_scalars = job
        for key, data in jpegs.items():
            rr.log(key, rr.EncodedImage(contents=data, media_type="image/jpeg"), static=True)
        # 关节读数与动作各合成一条推（逐个值推一条在 RK3566 上要 7 ms，合并后 0.8 ms）
        if obs_scalars:
            rr.log(f"{OBS_STR}.state", rr.Scalars(list(obs_scalars.values())))
        if act_scalars:
            rr.log(ACTION, rr.Scalars(list(act_scalars.values())))
    server.shutdown()


class WebPreview:
    """录制进程这一侧：限频取最新的一份，后台压 JPEG，交给子进程推出去。"""

    def __init__(
        self,
        web_port: int = DEFAULT_WEB_PORT,
        grpc_port: int = DEFAULT_GRPC_PORT,
        hz: float = DEFAULT_HZ,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
    ):
        ctx = multiprocessing.get_context("spawn")
        self._jobs = ctx.Queue(maxsize=2)
        self._proc = ctx.Process(
            target=_serve, args=(self._jobs, web_port, grpc_port, memory_limit), daemon=True
        )
        self._proc.start()
        self._slot = LatestSlot(1.0 / hz)
        self._quality = jpeg_quality
        self._stop = threading.Event()
        self._pump = threading.Thread(target=self._run, name="rerun-web-preview", daemon=True)
        self._pump.start()

    def offer(self, observation: RobotObservation | None, action: RobotAction | None) -> None:
        now = time.monotonic()
        if self._slot.due(now):
            self._slot.put(snapshot(observation, action), now)

    def _run(self) -> None:
        import cv2

        while not self._stop.is_set():
            if not self._slot.ready.wait(timeout=0.5):
                continue
            item = self._slot.take()
            if item is None:
                continue
            jpegs = {}
            for key, rgb in item.images.items():
                ok, buf = cv2.imencode(
                    ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self._quality]
                )
                if ok:
                    jpegs[key] = buf.tobytes()
            # 子进程忙：这一帧不推，下一帧更新
            with contextlib.suppress(queue.Full):
                self._jobs.put_nowait((jpegs, item.observation_scalars, item.action_scalars))

    def close(self) -> None:
        self._stop.set()
        self._pump.join(timeout=2)
        with contextlib.suppress(Exception):
            self._jobs.put(None, timeout=2)
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()


_preview: WebPreview | None = None


def init_rerun_web(web_port: int | None = None, grpc_port: int | None = None) -> None:
    """在板子上起网页预览，并打印浏览器要打开的地址。"""
    require_package("rerun-sdk", extra="viz", import_name="rerun")
    global _preview
    if _preview is not None:
        return
    web_port = web_port or int(os.getenv("LEROBOT_WEB_PREVIEW_PORT", DEFAULT_WEB_PORT))
    grpc_port = grpc_port or DEFAULT_GRPC_PORT
    hz = float(os.getenv("LEROBOT_WEB_PREVIEW_HZ", DEFAULT_HZ))
    _preview = WebPreview(web_port=web_port, grpc_port=grpc_port, hz=hz)
    logger.info(f"网页预览：浏览器打开 http://{local_address()}:{web_port}")


def log_rerun_web_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """把这一拍的画面与关节交给网页预览（限频取最新，不等推送）。``compress_images`` 无意义：总是 JPEG。"""
    if _preview is not None:
        _preview.offer(observation, action)


def shutdown_rerun_web() -> None:
    global _preview
    if _preview is not None:
        _preview.close()
        _preview = None
