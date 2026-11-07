"""录制控制循环的线程改用实时调度（Xbotics）。

端侧计算平台只有 4 个 A55 核：每来一帧，两路相机解码、两路 MPP 编码子进程、图像统计进程与录制主线程同时要 CPU，
主线程排在后面，读舵机、写编码管道就被拖过一拍（33 ms），出现「Record loop is running slower」。
主线程改用 SCHED_FIFO 后，它一醒就先跑；其余工作照常在其他时间片里做完。

实测（两路 640×480 相机、MPP 边录边编、从臂真读，三条各 20 秒）：一分钟 slower 警告 16 条 → 0 条，每条 597 帧（满额 600）。

只改调用这一个线程；``SCHED_RESET_ON_FORK`` 让它之后新建的线程与子进程（每条的编码子进程等）恢复普通调度。
主线程每拍只忙十来毫秒、其余时间睡着，不会占住一个核不放；内核对实时任务另有 95% 的占用上限兜底。
"""

from __future__ import annotations

import logging
import os
import platform

logger = logging.getLogger(__name__)

DEFAULT_PRIORITY = 10


def raise_control_loop_priority() -> bool:
    """让调用线程改用实时调度；成功返回 True。

    端侧计算平台（带 MPP 的瑞芯微板子）上默认开启；``LEROBOT_LOOP_RT=0`` 关闭，``LEROBOT_LOOP_RT=<1–99>`` 指定优先级
    （其他机器上也可这样打开）。没有权限（不是 root、没有 CAP_SYS_NICE）时记一条警告，照常录制。
    """
    setting = os.environ.get("LEROBOT_LOOP_RT")
    if setting == "0":
        return False
    if setting is None:
        if platform.system() != "Linux":
            return False
        from lerobot.datasets.rkmpp import rkmpp_available

        if not rkmpp_available():
            return False
        priority = DEFAULT_PRIORITY
    else:
        priority = int(setting)
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO | os.SCHED_RESET_ON_FORK, os.sched_param(priority))
    except (AttributeError, OSError) as e:
        logger.warning(
            f"Could not switch the control loop to real-time scheduling ({e}); recording continues."
        )
        return False
    logger.info(f"Control loop runs with real-time scheduling (SCHED_FIFO, priority {priority}).")
    return True
