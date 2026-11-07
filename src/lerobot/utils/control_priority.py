"""录制控制循环的线程改用实时调度（Xbotics）。

端侧计算平台只有 4 个 A55 核：每来一帧，两路相机解码、两路 MPP 编码子进程、图像统计进程与录制主线程同时要 CPU，
主线程排在后面，读舵机、写编码管道就被拖过一拍（33 ms），出现「Record loop is running slower」。
主线程改用 SCHED_FIFO 后，它一醒就先跑；其余工作照常在其他时间片里做完。

实测（两路 640×480 相机、MPP 边录边编、从臂真读，三条各 20 秒）：一分钟 slower 警告 16 条 → 0 条，每条 597 帧（满额 600）。

只改调用这一个线程；``SCHED_RESET_ON_FORK`` 让它之后新建的线程与子进程（每条的编码子进程等）恢复普通调度。
主线程每拍只忙十来毫秒、其余时间睡着，不会占住一个核不放；内核对实时任务另有 95% 的占用上限兜底。
"""

from __future__ import annotations

import gc
import logging
import os
import platform

logger = logging.getLogger(__name__)

DEFAULT_PRIORITY = 10


def _configured_priority() -> int | None:
    """``LEROBOT_LOOP_RT`` 与是否在端侧计算平台上，定下要不要用实时调度、优先级多少；不用返回 None。"""
    setting = os.environ.get("LEROBOT_LOOP_RT")
    if setting == "0":
        return None
    if setting is not None:
        return int(setting)
    if platform.system() != "Linux":
        return None
    from lerobot.datasets.rkmpp import rkmpp_available

    return DEFAULT_PRIORITY if rkmpp_available() else None


def raise_control_loop_priority() -> bool:
    """让调用线程改用实时调度；成功返回 True。

    端侧计算平台（带 MPP 的瑞芯微板子）上默认开启；``LEROBOT_LOOP_RT=0`` 关闭，``LEROBOT_LOOP_RT=<1–99>`` 指定优先级
    （其他机器上也可这样打开）。没有权限（不是 root、没有 CAP_SYS_NICE）时记一条警告，照常录制。
    """
    priority = _configured_priority()
    if priority is None:
        return False
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO | os.SCHED_RESET_ON_FORK, os.sched_param(priority))
    except (AttributeError, OSError) as e:
        logger.warning(
            f"Could not switch the control loop to real-time scheduling ({e}); recording continues."
        )
        return False
    logger.info(f"Control loop runs with real-time scheduling (SCHED_FIFO, priority {priority}).")
    return True


# ---------------------------------------------------------------- 垃圾回收不在录制中途做

_gc_held = False


def hold_gc() -> bool:
    """录制开始前：完整回收一次、冻结已有对象，然后关掉自动回收，回收改在两条之间做（:func:`collect_gc_between_episodes`）。

    自动回收在哪个线程触发，所有 Python 线程都要停下；端侧计算平台上录制时实测单次 36–53 ms，一拍就超时。
    对象主要靠引用计数释放，关掉的只是成环垃圾的回收，一条（几十秒）攒下的很少。与实时调度同一个开关（``LEROBOT_LOOP_RT``）。
    """
    global _gc_held
    if _configured_priority() is None:
        return False
    gc.collect()
    gc.freeze()
    gc.disable()
    _gc_held = True
    return True


def collect_gc_between_episodes() -> None:
    """两条之间（存盘后，录制循环不在跑）做一次完整回收并重新冻结。"""
    if _gc_held:
        gc.collect()
        gc.freeze()


def release_gc() -> None:
    """录制结束：恢复自动回收。"""
    global _gc_held
    if _gc_held:
        gc.unfreeze()
        gc.enable()
        _gc_held = False
