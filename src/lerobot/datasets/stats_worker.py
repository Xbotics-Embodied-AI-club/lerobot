"""边录边编的图像统计进程（Xbotics）：在单独的进程里跑 RunningQuantileStats，录制进程只管发画面。

由 :class:`lerobot.datasets.video_utils.StatsProcess` 以 ``python <本文件>`` 起一个解释器运行，
只导入标准库与 numpy，按文件路径加载 :mod:`lerobot.datasets.running_quantile_stats`，不导入 lerobot 包
（端侧计算平台上 ``lerobot.datasets`` 一导入就要几十秒）。

为什么要单独的进程：一次统计在端侧计算平台上约 13 ms；放在录制主线程里，两路相机同一拍各算一次，
这一拍就超时；放在同一进程的线程里，又和读舵机的主线程争 GIL，读一次舵机被拖到十几毫秒。

协议：标准输入上一条条 pickle。``("update", 键, 数组)`` 把一批向量（N×C）喂给这个键的统计，不回话；
``("get", 键)`` 回 ``("ok", 统计字典)`` 或 ``("error", 说明)``，并丢弃这个键；``("drop", 键)`` 丢弃，不回话。
标准输入关闭即退出。同一个键的 update 按收到的顺序执行，结果与在本进程里依次 update 逐位相同。
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any, BinaryIO


def load_tracker_class() -> type:
    path = Path(__file__).with_name("running_quantile_stats.py")
    spec = importlib.util.spec_from_file_location("_xbotics_running_quantile_stats", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RunningQuantileStats


def serve(inp: BinaryIO, out: BinaryIO) -> None:
    tracker_class = load_tracker_class()
    trackers: dict[str, Any] = {}
    errors: dict[str, str] = {}
    while True:
        try:
            message = pickle.load(inp)  # nosec B301 —— 只读父进程（录制进程）经管道送来的数据
        except EOFError:
            return
        op, key = message[0], message[1]
        if op == "update":
            if key in errors:
                continue
            try:
                trackers.setdefault(key, tracker_class()).update(message[2])
            except Exception as e:  # 记下来，取结果时再报给录制进程
                errors[key] = f"{type(e).__name__}: {e}"
        elif op == "get":
            tracker = trackers.pop(key, None)
            if key in errors:
                reply: tuple = ("error", errors.pop(key))
            elif tracker is None:
                reply = ("error", f"no frames were sent for {key}")
            else:
                try:
                    reply = ("ok", tracker.get_statistics())
                except Exception as e:
                    reply = ("error", f"{type(e).__name__}: {e}")
            pickle.dump(reply, out)
            out.flush()
        elif op == "drop":
            trackers.pop(key, None)
            errors.pop(key, None)


if __name__ == "__main__":
    serve(sys.stdin.buffer, sys.stdout.buffer)
