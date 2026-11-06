"""网页预览的推送进程（Xbotics）：起 rerun 数据服务、网页查看器与跳转页，把录制进程送来的画面与关节推给浏览器。

由 :mod:`lerobot.utils.rerun_web_visualization` 以 ``python <本文件> <参数>`` 单独起一个解释器运行，
**只导入标准库与 rerun，不导入 lerobot 包**：用 multiprocessing 的 spawn 起子进程会把录制程序整个重新导入一遍
（torch 等，端侧计算平台上约 30 秒单核 CPU），正好压在第一条录制上；``lerobot`` 包的 ``utils`` 一导入就要 12 秒。
本文件独立运行时只要导入 rerun 的 3 秒。

录制进程经标准输入一条条送 pickle：``(jpegs, scalars)``，``jpegs`` 是 ``{实体名: JPEG 字节}``，
``scalars`` 是 ``{实体名: [浮点数, …]}``；标准输入关闭即退出。
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import os
import pickle
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import quote


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


def serve(web_port: int, grpc_port: int, memory_limit: str) -> None:
    with contextlib.suppress(OSError):
        os.nice(5)  # 让着录制：四个核都忙时先保录制不掉帧，预览慢一点无妨
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

    jobs = sys.stdin.buffer
    while True:
        try:
            jpegs, scalars = pickle.load(jobs)  # nosec B301 —— 只读父进程（录制进程）经管道送来的数据
        except EOFError:
            break
        for key, data in jpegs.items():
            rr.log(key, rr.EncodedImage(contents=data, media_type="image/jpeg"), static=True)
        for key, values in scalars.items():
            rr.log(key, rr.Scalars(values))
    server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web-port", type=int, required=True)
    parser.add_argument("--grpc-port", type=int, required=True)
    parser.add_argument("--memory-limit", required=True)
    args = parser.parse_args()
    serve(args.web_port, args.grpc_port, args.memory_limit)


if __name__ == "__main__":
    main()
