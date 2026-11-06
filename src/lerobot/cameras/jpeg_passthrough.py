"""相机原始 JPEG 直通预览（Xbotics）：MJPG 相机送来的本来就是 JPEG，预览直接转发它，由浏览器解码。

录数据要原图（编码进数据集），所以相机线程照样把 JPEG 解成原图；同时把这一帧的原始 JPEG 记在这里。
网页预览拿到观测里的画面时按「是不是同一个数组对象」查回它的 JPEG，查到就原样推给浏览器，
板子上不用为预览再压一遍 JPEG（端侧计算平台上两路 640×480 每秒 10 次，压图约占录制进程三成个核）。

画面在相机之后被改过（旋转、裁剪、换了数组）就查不到，预览照旧自己压。
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

# 每路相机只需要最近几帧：预览每 0.1 秒取一次最新的
_MAX_ENTRIES = 8

_lock = threading.Lock()
_entries: OrderedDict[int, tuple[np.ndarray, bytes]] = OrderedDict()


def remember(frame: np.ndarray, jpeg: bytes) -> None:
    """记下 ``frame`` 是由 ``jpeg`` 解出来的。保留数组引用，id 在登记期间不会被别的对象复用。"""
    with _lock:
        _entries[id(frame)] = (frame, jpeg)
        while len(_entries) > _MAX_ENTRIES:
            _entries.popitem(last=False)


def jpeg_of(frame: np.ndarray) -> bytes | None:
    """``frame`` 若正是相机登记过的那个数组，返回它的原始 JPEG；否则 None。"""
    with _lock:
        entry = _entries.get(id(frame))
    if entry is None or entry[0] is not frame:
        return None
    return entry[1]


def clear() -> None:
    with _lock:
        _entries.clear()
