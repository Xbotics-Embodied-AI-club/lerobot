"""带 Web 控制台的异步推理客户端（Xbotics）：``lerobot-inference-console``。

在维护版 ``robot_client`` 之上多了这些，服务端还是 ``python -m lerobot.async_inference.policy_server``：

- **RTC（Real-Time Chunking）**：按推理延迟滑动窗口自适应决定何时要下一个动作块；新旧动作块重叠的开头几步线性融合；
  执行层 EMA 平滑（夹爪不平滑）；可选把每个动作插值成几个子步（推理慢时控制循环照样跑高频）。
- **Web 控制台**（默认开，``http://<板子>:8007``）：相机画面、动作队列里新 / 旧 / 融合各占哪几步、
  推理延迟与 chunk 使用率；网页上改任务文本、选预设任务、启停任务、热调 RTC 参数，下一个控制周期生效。
- **两种传输**（``--transport``）：``grpc``（默认，连 LeRobot 的 policy_server，相机画面压 JPEG 再发，
  与 ``robot_client`` 同一实现）；``websocket``（连 LingBot-VLA 官方推理服务，msgpack 协议）。

移植自 gitee ``zhuzhuangtian/lerobot-v0.5.1-custom`` 的 ``lerobot_tools/core/lerobot_robot_client.py``；
改动：gRPC 默认发 JPEG、控制台默认开在 0.0.0.0:8007、命令行入口与插件注册、去掉 so102 机型的可选导入。

```shell
lerobot-inference-console \
    --robot.type=so101_follower \
    --robot.port=/dev/xbotics/follower \
    --robot.id=follower \
    --robot.cameras="{top: {type: opencv, index_or_path: /dev/xbotics/top, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/xbotics/wrist, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --task="Grab the cube" \
    --policy_type=act \
    --pretrained_name_or_path=<模型目录或 HF repo> \
    --policy_device=cuda \
    --server_host=127.0.0.1 \
    --server_port=8080 \
    --actions_per_chunk=50 \
    --fps=30
```
"""

import io
import logging
import math
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import torch
from PIL import Image

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.constants import DEFAULT_FPS
from lerobot.async_inference.helpers import (
    FPSTracker,
    RawObservation,
    TimedAction,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins

from .transport import create_transport

DEFAULT_WEBUI_PORT = 8007


class _LoopStats:
    """控制循环分段耗时统计（指数滑动平均），用于定位帧率瓶颈。"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.action_ms = 0.0  # 执行动作（含串口写）耗时
        self.obs_ms = 0.0  # 采集观测（串口读+相机+打包）耗时
        self.cycle_ms = 0.0  # 完整循环周期（含 sleep）
        self.obs_interval_ms = 0.0  # 相邻两次观测采集的间隔
        self._last_obs_t: float | None = None

    def update(
        self,
        action_ms: float | None,
        obs_ms: float | None,
        cycle_ms: float,
    ) -> None:
        """action_ms/obs_ms 传 None 表示本迭代未执行该事件（不参与平均）。"""
        a = self.alpha
        if action_ms is not None:
            self.action_ms = a * action_ms + (1 - a) * self.action_ms
        if obs_ms is not None:
            self.obs_ms = a * obs_ms + (1 - a) * self.obs_ms
        self.cycle_ms = a * cycle_ms + (1 - a) * self.cycle_ms
        if obs_ms is not None:
            now = time.perf_counter()
            if self._last_obs_t is not None:
                interval = (now - self._last_obs_t) * 1000
                self.obs_interval_ms = a * interval + (1 - a) * self.obs_interval_ms
            self._last_obs_t = now

    @property
    def measured_fps(self) -> float:
        return 1000 / self.cycle_ms if self.cycle_ms > 0 else 0.0

    @property
    def obs_capture_fps(self) -> float:
        return 1000 / self.obs_interval_ms if self.obs_interval_ms > 0 else 0.0


class ChunkUsageStats:
    """Chunk 使用率统计：评估每个推理 chunk 实际被执行了多少，用于衡量动作新鲜度。

    物理含义
    --------
    服务端为 chunk 打的 timestep 是**绝对执行位置**（`i_0` 取观测时刻的
    `latest_action`），因此一个 chunk 覆盖 [start, start+C-1] 这段执行区间。
    下一个 chunk 到达后会覆盖其中尚未执行的部分（重叠区 = 融合段），
    这些被覆盖的动作**永远执行不到**。

    因此对第 k 个 chunk：

        used(k)  = C - remaining_at_next_arrival     ← 真正被执行掉的动作数
        used_ratio = used / C                         ← 不含融合段的使用率

    其中 `remaining_at_next_arrival` 是**下一个 chunk 到达前**队列里还剩多少动作
    （即被覆盖掉、永远执行不到的那部分长度）。

    怎么读（方向必须记牢，容易说反）
    ------------------------------
    - `used_ratio` 低（→0）：新 chunk 很快就把旧块替换掉，动作一直跟着最新推理走。
      **对应 `lookahead_ratio` 调大** → 触发更早 → 老块用得更少 → **动作更新鲜**。
    - `used_ratio` 高（→1）：旧块接近用完才被替换，执行的是很长的开环预测尾部。
      **对应 `lookahead_ratio` 调小** → 触发更晚 → 老块用得更多 → **尾部预测误差更大**。
    - `refresh_depth` = 1 - used_ratio：新 chunk 对旧队列的替换深度（融合段占比），
      稳态下它 ≈ 触发点 / C，因此可直接用来验证 `lookahead_ratio` 是否生效。
    - 注意不要用"动作年龄"来判断新鲜度：无论 `used_ratio` 高低，被执行的动作都是
      `L`（推理延迟）之前那次观测推出来的，两者相同。真正的差别是**开环执行了多长**：
      `used_ratio` 高意味着靠同一次观测外推了更多步，累计漂移更大。

    实现说明
    --------
    只在"chunk 到达"这一个事件上采样（入队前读一次队列长度），
    不做逐动作记账，因此无锁竞争、无边界归因歧义，且天然满足
    0 <= used_ratio <= 1。代价是该 chunk 的读数要等到下一个 chunk 到达才结算
    （在此之前 `chunk_usage_current` 给出进行中的近似值）。
    """

    def __init__(self, window: int = 20):
        self._lock = threading.Lock()
        self._window = max(1, int(window))
        self._history: list[dict] = []
        self.last: dict | None = None
        self._pending: dict | None = None  # 当前 chunk（等待下一个 chunk 结算）
        self._executed_total = 0  # 本计量窗口内执行掉的动作数

    def settle_previous(self, queue_size_before_merge: int, nominal_size: int) -> None:
        """新 chunk 到达时调用：用【合并前】的队列长度结算上一个 chunk，并登记本 chunk。

        `queue_size_before_merge` 必须在 `_aggregate_action_queues` **之前**采样：
        合并会覆盖旧队列，合并后就再也读不到"旧 chunk 还剩多少"了。
        旧 chunk 的未执行长度就是它被执行掉的部分的反面，故
        old_used = old_C - queue_size_before_merge。
        """
        with self._lock:
            if self._pending is not None and self._pending["nominal_size"]:
                nominal = self._pending["nominal_size"]
                remaining = max(0, int(queue_size_before_merge))
                used = max(0, min(nominal, nominal - remaining))
                entry = {
                    "arrived_at": self._pending["arrived_at"],
                    "settled_at": time.time(),
                    "nominal_size": nominal,
                    "remaining_at_next_arrival": remaining,
                    "used_steps": used,
                    "used_ratio": used / nominal,
                    "refresh_depth": 1.0 - used / nominal,
                    "executed_total": self._executed_total,
                }
                self.last = entry
                self._history.append(entry)
                if len(self._history) > self._window:
                    self._history.pop(0)
            self._pending = {"arrived_at": time.time(), "nominal_size": max(1, int(nominal_size))}
            self._executed_total = 0

    def on_action_executed(self, timestep: int | None = None) -> None:
        """控制循环每执行一个队列动作调用一次（仅用于进行中的近似读数）。

        `timestep` 仅为兼容调用点保留；本实现不依赖 timestep 归因。
        """
        with self._lock:
            self._executed_total += 1

    def _mean(self, key: str) -> float | None:
        values = [e[key] for e in self._history if e.get(key) is not None]
        return sum(values) / len(values) if values else None

    def snapshot(self) -> dict:
        """供 /api/status 与进度条读取的不可变快照。"""
        with self._lock:
            current = None
            if self._pending is not None and self._pending["nominal_size"]:
                nominal = self._pending["nominal_size"]
                executed = self._executed_total
                # 进行中的近似值：已执行数 / chunk 长度（未结算，仅供参考）
                current = {
                    "arrived_at": self._pending["arrived_at"],
                    "nominal_size": nominal,
                    "used_steps": min(executed, nominal),
                    "used_ratio": min(executed, nominal) / nominal,
                    "provisional": True,
                }
            settled = self.last["used_ratio"] if self.last else None
            return {
                "chunk_usage_ratio": settled,
                # 尚未产生已结算样本时回退到进行中的近似值，避免刚启动就没有读数
                "chunk_usage_ratio_display": (
                    settled if settled is not None else (current["used_ratio"] if current else None)
                ),
                "chunk_usage_ratio_avg": self._mean("used_ratio"),
                # 新 chunk 对旧队列的替换深度 = 1 - 使用率（稳态下 ≈ 触发点/C）
                "chunk_refresh_depth": self.last["refresh_depth"] if self.last else None,
                "chunk_refresh_depth_avg": self._mean("refresh_depth"),
                "chunk_usage_current": current,
                "chunk_usage_samples": len(self._history),
                "chunk_usage_last_size": self.last["nominal_size"] if self.last else None,
                "chunk_usage_last_steps": self.last["used_steps"] if self.last else None,
                "chunk_usage_last_remaining": (self.last["remaining_at_next_arrival"] if self.last else None),
            }


class _FixedLineQueueBar:
    """固定在终端最后一行刷新的单行进度条，可与日志输出共存。

    - 行宽严格受限（不超过终端宽度且封顶），绝不换行；
    - 每次刷新用 \\r 回到行首并用空格覆写清除残留，再重绘（兼容不支持 ANSI 的终端）；
    - 配合 _BarAwareStreamHandler：日志输出前先清除进度条行，日志写完再在新行重绘，
      进度条始终钉在最后一行，日志正常向上滚动。
    """

    # 单行总宽度上限（避免过宽终端下行尾换行）
    MAX_LINE_WIDTH = 100

    _active: "_FixedLineQueueBar | None" = None
    _lock = threading.RLock()

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stderr
        self._last_line = ""
        _FixedLineQueueBar._active = self
        _install_bar_aware_handlers()

    def _render(
        self, queue_size: int, chunk_size: int, step: int, infer_ms: float, usage_ratio: float | None = None
    ) -> str:
        chunk_size = max(chunk_size, 1)
        ratio = min(queue_size / chunk_size, 1.0)
        usage = f" | used={usage_ratio * 100:.0f}%" if usage_ratio is not None else ""
        info = f" {queue_size}/{chunk_size} | step={step} | infer={infer_ms:.0f}ms{usage}"
        prefix = "Queue ["
        # 终端宽度与 MAX_LINE_WIDTH 取小，并留 2 字符余量，保证单行不换行
        usable = min(shutil.get_terminal_size(fallback=(80, 24)).columns, self.MAX_LINE_WIDTH) - 2
        bar_width = max(8, usable - len(prefix) - len(info) - 1)
        filled = int(bar_width * ratio)
        return f"{prefix}{'#' * filled}{'-' * (bar_width - filled)}]{info}"

    def update(
        self, queue_size: int, chunk_size: int, step: int, infer_ms: float, usage_ratio: float | None = None
    ) -> None:
        line = self._render(queue_size, chunk_size, step, infer_ms, usage_ratio)
        with self._lock:
            self._stream.write("\r" + " " * len(self._last_line) + "\r" + line)
            self._stream.flush()
            self._last_line = line

    def clear_line(self) -> None:
        """清除进度条所在行（供日志 handler 在写日志前调用）。"""
        if self._last_line:
            self._stream.write("\r" + " " * len(self._last_line) + "\r")

    def redraw(self) -> None:
        """在当前行重绘进度条（供日志 handler 在写完日志后调用）。"""
        if self._last_line:
            self._stream.write(self._last_line)
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.write("\n")
            self._stream.flush()
            if _FixedLineQueueBar._active is self:
                _FixedLineQueueBar._active = None


class _BarAwareStreamHandler(logging.StreamHandler):
    """写日志前先清除进度条行，写完日志后在新行重绘进度条，避免两者混在一行。"""

    def emit(self, record):
        bar = _FixedLineQueueBar._active
        if bar is None:
            super().emit(record)
            return
        with bar._lock:
            bar.clear_line()
            super().emit(record)
            bar.redraw()


def _install_bar_aware_handlers() -> None:
    """将 root logger 的控制台 handler 包装为进度条感知版本（幂等）。"""
    root = logging.getLogger()
    for i, h in enumerate(list(root.handlers)):
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, (logging.FileHandler, _BarAwareStreamHandler)
        ):
            wrapped = _BarAwareStreamHandler(stream=h.stream)
            wrapped.setLevel(h.level)
            wrapped.setFormatter(h.formatter)
            root.handlers[i] = wrapped


def _pid_listening_on_port(port: int) -> int | None:
    """返回监听该端口的进程 PID（找不到或无权限时返回 None）。"""
    import re
    import subprocess

    try:
        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if f":{port} " not in line and not line.rstrip().endswith(f":{port}"):
            continue
        m = re.search(r"pid=(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def _ensure_port_free(host: str, port: int, timeout: float = 5.0) -> None:
    """预检端口是否可绑定；被占用时抛出带处理建议的明确错误（Web 端口冲突时
    浏览器会拿到异常响应表现为白屏，启动前拦截并提示）。

    Windows 允许不带 SO_EXCLUSIVEADDRUSE 的通配地址（0.0.0.0）与已绑定的
    具体地址重叠——预检必须通过 SO_EXCLUSIVEADDRUSE 才能检出这种占用。

    占用处理策略：
    1. 先宽限 timeout 秒（上一次运行可能正在退出、端口尚未释放）；
    2. 仍被占用且占用者是本客户端的残留进程（cmdline 含 inference_console）
       时自动终止并等待其退出，避免上次没退干净导致无法重启；
    3. 其他进程占用则抛出原错误，不越权清理。
    """
    import socket
    import sys
    import time

    def _try_bind() -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform == "win32":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # 与真实服务（uvicorn 等）保持一致设 SO_REUSEADDR：上次运行留下的
            # TIME_WAIT 连接（内核持有 60s，无进程 pid）不应阻止重新绑定
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    deadline = time.monotonic() + timeout
    while True:
        if _try_bind():
            return
        if time.monotonic() < deadline:
            time.sleep(0.2)
            continue

        # 宽限期后仍占用：尝试识别并清理本客户端的残留进程
        pid = _pid_listening_on_port(port)
        leftover_killed = False
        if pid is not None and pid != os.getpid():
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                cmdline = ""
            if "inference_console" in cmdline or "lerobot-inference-console" in cmdline:
                print(f"WebUI 端口 {port} 被上次未退干净的客户端进程（PID {pid}）占用，正在终止它…")
                os.kill(pid, signal.SIGTERM)
                kill_deadline = time.monotonic() + 5.0
                while time.monotonic() < kill_deadline:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        leftover_killed = True
                        break
                    time.sleep(0.2)
                if not leftover_killed:
                    os.kill(pid, signal.SIGKILL)
                    leftover_killed = True
                # 等端口真正释放
                release_deadline = time.monotonic() + 3.0
                while time.monotonic() < release_deadline:
                    if _try_bind():
                        print(f"已清理残留进程，端口 {port} 释放。")
                        return
                    time.sleep(0.2)

        raise RuntimeError(
            f"WebUI 端口 {port}（监听地址 {host}）已被占用"
            + ("（检测到残留客户端进程但清理后仍被占用）" if leftover_killed else "")
            + "。请用 --webui_port 换一个端口，或释放该端口后重试。"
        )


class InferenceLatencyTracker:
    """滑动窗口推理延迟追踪器，用于 RTC 自适应提前触发."""

    def __init__(self, window_size: int = 5, init_latency_ms: float = 100.0):
        self.window_size = window_size
        self._history: list[float] = [init_latency_ms]
        self._lock = threading.Lock()

    def update(self, latency_ms: float):
        with self._lock:
            self._history.append(latency_ms)
            if len(self._history) > self.window_size:
                self._history.pop(0)

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            return sum(self._history) / len(self._history)

    @property
    def max_latency_ms(self) -> float:
        """窗口内最大延迟，比均值更能覆盖抖动（用于触发点估计更保守）。"""
        with self._lock:
            return max(self._history)


@dataclass
class PolicyRobotClientConfig:
    """Configuration for PolicyRobotClient.

    大部分参数与 LeRobot 的 RobotClientConfig 一致，另加 RTC / Web 控制台配置，
    以及传输选择（transport=grpc/websocket）与 WebSocket（LingBot-VLA）配置段。
    """

    robot: RobotConfig = field(metadata={"help": "Robot configuration"})
    task: str = field(
        default="", metadata={"help": "Task instruction for the robot (e.g. 'Pick up the red cube')"}
    )
    device_name: str = field(
        default="",
        metadata={
            "help": "Device name displayed in the web UI to distinguish multiple deployed robots "
            "(e.g. 'RK3566-1')."
        },
    )
    task_presets_file: str = field(
        default="",
        metadata={
            "help": "Path to a YAML config file whose 'tasks' section defines preset tasks "
            "({name: text}) selectable in the web UI."
        },
    )

    # 推理传输选择：grpc（默认，LeRobot policy_server，画面压 JPEG）| websocket（LingBot-VLA 官方推理服务）
    transport: str = field(
        default="grpc",
        metadata={
            "help": "Inference transport: grpc (LeRobot original policy server, default) | "
            "websocket (LingBot-VLA msgpack server)."
        },
    )

    # gRPC policy server configuration
    server_host: str = field(default="localhost", metadata={"help": "Policy server host"})
    server_port: int = field(default=8080, metadata={"help": "Policy server port"})
    # 与 robot_client 的同名参数一致：相机画面压 JPEG 再发，服务端解回数组（grpc 传输）
    observation_jpeg_quality: int = field(
        default=90,
        metadata={"help": "grpc transport: JPEG quality for camera frames sent to the server (0 = send raw)"},
    )

    # WebSocket (LingBot-VLA) configuration
    ws_host: str = field(
        default="",
        metadata={
            "help": "LingBot-VLA websocket server host. Supports ws:// or wss:// prefix, "
            "or a bare host (with ws_use_ssl / ws_port)."
        },
    )
    ws_port: int | None = field(default=None, metadata={"help": "LingBot-VLA websocket server port."})
    ws_path: str = field(default="", metadata={"help": "LingBot-VLA websocket URL path (e.g. /ws)."})
    ws_use_ssl: bool = field(default=False, metadata={"help": "Use wss:// when ws_host has no prefix."})
    ws_api_key: str = field(
        default="", metadata={"help": "API key sent as 'Authorization: Api-Key <key>' header."}
    )
    ws_robo_name: str = field(
        default="",
        metadata={"help": "Server-side robot config name, sent with the reset message at connect time."},
    )
    ws_image_codec: str = field(
        default="jpeg",
        metadata={"help": "Image transport encoding for websocket transport: raw | jpeg."},
    )
    ws_obs_image_size: int = field(
        default=224,
        metadata={"help": "Resize observation images to this square size before sending (0 = no resize)."},
    )
    ws_jpeg_quality: int = field(
        default=70,
        metadata={"help": "JPEG encoding quality (1-100) when ws_image_codec=jpeg."},
    )
    action_output_keys: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Generic action key order for websocket transport (bi-manual robots use the "
            "built-in arm+effector reassembly and don't need this). "
            "Default: sorted action keys."
        },
    )

    # Policy configuration（SendPolicyInstructions 携带的 RemotePolicyConfig）
    policy_type: str = field(
        default="act",
        metadata={"help": "Policy type: act/smolvla/diffusion/tdmpc/vqbet/pi0/pi05/groot"},
    )
    pretrained_name_or_path: str = field(
        default="",
        metadata={"help": "HF repo id or local path of the pretrained policy, loaded by the server"},
    )
    actions_per_chunk: int = field(
        default=25,
        metadata={"help": "Number of actions to use per chunk"},
    )
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference on the server"})
    client_device: str = field(
        default="cpu",
        metadata={"help": "Device to move actions to after receiving from server"},
    )

    # Control behavior configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Control frequency (Hz)"})
    chunk_size_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Threshold for chunk size control. When queue_size/chunk_size > threshold, stop sending new observations. Fallback when RTC is disabled."
        },
    )
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={
            "help": "Action aggregation function when overlapping chunks are received (only used when RTC is disabled)."
        },
    )

    # Real-Time Chunking (RTC) configuration
    rtc_enabled: bool = field(
        default=True,
        metadata={
            "help": "Enable Real-Time Chunking (RTC) for smoother robot motion. "
            "When enabled, uses adaptive lookahead and progressive blending."
        },
    )
    rtc_lookahead_ratio: float = field(
        default=0.4,
        metadata={
            "help": "RTC: floor for the adaptive trigger point, as a fraction of chunk_size. "
            "The client triggers when remaining queue steps / chunk_size <= this ratio. "
            "Larger = earlier trigger = LESS of each chunk gets executed before being "
            "replaced (fresher actions, more margin against latency jitter, but more "
            "frequent inference); smaller = later trigger = MORE of each chunk is used "
            "(staler open-loop tail). Only takes effect when it exceeds the adaptive "
            "latency-based trigger point; otherwise the adaptive term binds. "
            "Only used when rtc_enabled=True. Range: [0, 1]"
        },
    )
    rtc_blend_steps: int = field(
        default=5,
        metadata={
            "help": "RTC: number of overlapping steps to perform progressive linear blending "
            "when a new chunk arrives. 0 = disable blending."
        },
    )
    rtc_action_ema_alpha: float = field(
        default=0.3,
        metadata={
            "help": "RTC: EMA smoothing factor for executed actions (0=heavy smooth, 1=no smooth). "
            "Helps suppress high-frequency jitter at execution layer."
        },
    )
    rtc_ema_exclude_gripper: bool = field(
        default=True,
        metadata={
            "help": "RTC: skip EMA smoothing for gripper joints (keys containing 'gripper'). "
            "Gripper open/close needs crisp response; smoothing would smear it."
        },
    )
    rtc_safety_margin_steps: int = field(
        default=2,
        metadata={
            "help": "RTC: extra safety margin in control steps when calculating adaptive trigger point. "
            "NOTE: this also determines the overlap length between old/new chunks "
            "(old queue remaining at chunk arrival ≈ this value), i.e. the room available "
            "for blending. Set it >= rtc_blend_steps for a smooth transition."
        },
    )
    rtc_latency_window: int = field(
        default=5,
        metadata={"help": "RTC: sliding window size for inference latency moving average."},
    )

    # 动作插值细分：每个队列动作拆成 N 个子步线性插值执行。
    # 适合推理慢导致控制 FPS 受限的场景：控制循环跑高 FPS（动作平滑），
    # 队列仍以 FPS/N 的速率消耗（等效低频，推理能跟上）。1 = 不启用。
    action_interp_steps: int = field(
        default=1,
        metadata={
            "help": "Split each queued action into N linearly interpolated sub-steps. "
            "Control loop runs at high FPS (smooth motion) while the action queue "
            "drains at FPS/N (matching slow inference). 1 = disabled."
        },
    )

    # 停止任务时机械臂回到初始位置（home）
    home_action: dict[str, float] = field(
        default_factory=dict,
        metadata={
            "help": "Joint targets sent to the robot directly (one-shot) when the task stops, "
            "as {motor_name: value}, e.g. {shoulder_pan.pos: 0.0, gripper.pos: 2.5}. "
            "Units follow the robot config (so101: degrees, gripper 0-100). "
            "Servo Acceleration register provides the ramp. Empty = stay in place."
        },
    )

    # Web 控制台：浏览器查看运行状态 / 相机画面 / 动作队列融合可视化，支持运行时修改任务文本
    webui_enabled: bool = field(
        default=True,
        metadata={
            "help": "Enable a lightweight web UI (Flask) showing run status, "
            "camera feeds, action-queue/chunk-fusion visualization, and allowing "
            "runtime task text updates."
        },
    )
    webui_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Web UI bind host. Use 0.0.0.0 to access from other machines (no auth!)."},
    )
    webui_port: int = field(
        default=DEFAULT_WEBUI_PORT,
        metadata={"help": "Web UI port."},
    )
    webui_video_enabled: bool = field(
        default=True,
        metadata={
            "help": "Push camera frames to the web UI (MJPEG streams). Disable to save CPU "
            "on low-power boards (frame capture/JPEG encoding is significant). "
            "Can also be toggled at runtime via POST /api/video."
        },
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize action queue size after stopping"}
    )
    display_queue_bar: bool = field(
        default=True,
        metadata={
            "help": "Display a fixed-position progress bar in the terminal showing the "
            "remaining actions in the action queue (queue size / chunk size)."
        },
    )

    @property
    def server_address(self) -> str:
        return f"{self.server_host}:{self.server_port}"

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        # websocket 传输由服务端自己定模型，不需要这一项
        if self.transport == "grpc" and not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")
        if not self.client_device:
            raise ValueError("client_device cannot be empty")
        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

        # Validate transport params
        if self.transport not in ("grpc", "websocket"):
            raise ValueError(f"transport must be 'grpc' or 'websocket', got {self.transport!r}")
        if not 0 <= self.observation_jpeg_quality <= 100:
            raise ValueError(
                f"observation_jpeg_quality must be between 0 and 100, got {self.observation_jpeg_quality}"
            )
        if self.transport == "websocket":
            if not self.ws_host:
                raise ValueError("ws_host is required when transport=websocket")
            if not self.ws_robo_name:
                raise ValueError("ws_robo_name is required when transport=websocket")
            if self.ws_image_codec not in ("raw", "jpeg"):
                raise ValueError(f"ws_image_codec must be 'raw' or 'jpeg', got {self.ws_image_codec!r}")
            if self.ws_obs_image_size < 0:
                raise ValueError(f"ws_obs_image_size must be >= 0, got {self.ws_obs_image_size}")
            if not (1 <= self.ws_jpeg_quality <= 100):
                raise ValueError(f"ws_jpeg_quality must be in [1, 100], got {self.ws_jpeg_quality}")

        # Validate RTC params
        if self.rtc_lookahead_ratio < 0 or self.rtc_lookahead_ratio > 1:
            raise ValueError(f"rtc_lookahead_ratio must be in [0, 1], got {self.rtc_lookahead_ratio}")
        if self.rtc_blend_steps < 0:
            raise ValueError(f"rtc_blend_steps must be >= 0, got {self.rtc_blend_steps}")
        if not (0.0 <= self.rtc_action_ema_alpha <= 1.0):
            raise ValueError(f"rtc_action_ema_alpha must be in [0, 1], got {self.rtc_action_ema_alpha}")
        if self.action_interp_steps < 1:
            raise ValueError(f"action_interp_steps must be >= 1, got {self.action_interp_steps}")


class PolicyRobotClient:
    """Robot client with pluggable inference transport (gRPC / WebSocket LingBot-VLA).

    Architecture:
    - Main thread (control_loop): executes actions from the local queue and submits
      observations to the transport (non-blocking).
    - Transport (transport.py): owns the sending thread / blocking inference,
      and turns protocol-specific messages into list[TimedAction] chunks.
    - Background thread (receive_actions): loops transport.receive_chunk() and merges
      incoming action chunks into the local queue.
    """

    prefix = "grpc_policy_robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyRobotClientConfig):
        """Initialize the client: connect robot and create the inference transport."""
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        self.transport = create_transport(config, self.robot, self.logger)

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # RTC: inference latency tracker for adaptive lookahead
        self.latency_tracker = InferenceLatencyTracker(
            window_size=config.rtc_latency_window,
            init_latency_ms=100.0,  # 默认假设 100ms 初始延迟
        )
        # RTC: 观测采集耗时追踪（发生在延迟戳之前，不计入 latency_tracker，但必须算进触发点）
        self._obs_overhead_tracker = InferenceLatencyTracker(
            window_size=config.rtc_latency_window,
            init_latency_ms=50.0,
        )

        # RTC: action EMA smoothing state
        self._ema_action_lock = threading.Lock()
        self._ema_prev_action: torch.Tensor | None = None
        # 夹爪不平滑的逐维掩码（True=平滑，False=透传）
        self._ema_mask: torch.Tensor | None = None
        if config.rtc_ema_exclude_gripper:
            keys = list(self.robot.action_features)
            mask = ["gripper" not in k.lower() for k in keys]
            if not all(mask):
                excluded = [k for k, m in zip(keys, mask, strict=False) if not m]
                self._ema_mask = torch.tensor(mask, dtype=torch.bool)
                self.logger.info(f"EMA smoothing excluded for gripper joints: {excluded}")

        # 动作插值细分状态
        self._interp_last_target: torch.Tensor | None = None  # 上一个从队列取出的目标动作
        self._interp_target: torch.Tensor | None = None  # 当前正在逼近的目标动作
        self._interp_start: torch.Tensor | None = None  # 当前插值段起点
        self._interp_remaining: int = 0  # 当前插值段剩余子步数

        # Web 控制台状态
        self._task_lock = threading.Lock()
        self._current_task = config.task  # 运行时可被 webui 动态修改
        self._task_active_lock = threading.Lock()
        self._task_active = True  # 任务运行/停止（Web/API 可切换）

        # 预设任务列表（从 YAML 配置文件的 tasks 节加载）
        self._task_presets: dict[str, str] = {}
        if config.task_presets_file:
            p = Path(config.task_presets_file)
            if p.exists():
                try:
                    import yaml

                    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    self._task_presets = {str(k): str(v) for k, v in (data.get("tasks") or {}).items()}
                    self.logger.info(f"Loaded {len(self._task_presets)} task presets from {p}")
                except Exception as e:
                    self.logger.warning(f"Failed to load task presets from {p}: {e}")
            else:
                self.logger.warning(f"Task presets file not found: {p}")

        self.actions_sent_count = 0
        # Chunk 使用率 / 动作新鲜度统计（见 ChunkUsageStats 文档）
        self.chunk_usage = ChunkUsageStats(window=20)
        self._queue_viz: list[dict] = []  # 队列组成快照（old/new/agg/blend+alpha）
        self._webui_frames: dict[str, bytes] = {}
        self._webui_frames_lock = threading.Lock()
        # 停止任务回初始位置：校验关节名合法性
        if config.home_action:
            unknown = set(config.home_action) - set(self.robot.action_features)
            if unknown:
                raise ValueError(
                    f"home_action contains unknown joints: {sorted(unknown)}; "
                    f"valid: {list(self.robot.action_features)}"
                )

        # 控制循环分段耗时统计（帧率瓶颈定位）
        self._loop_stats = _LoopStats()

        self._webui = None
        if config.webui_enabled:
            from .webui import WebUIServer

            _ensure_port_free(config.webui_host, config.webui_port)
            self._webui = WebUIServer(self, host=config.webui_host, port=config.webui_port)
            self._webui.start()
            self.logger.info(f"Web UI available at {self._webui.address}")
            # 相机帧采集/JPEG 编码在独立线程中进行（~10Hz），不占用控制循环时间
            self._webui_capture_thread = threading.Thread(
                target=self._webui_capture_loop, daemon=True, name="webui_capture"
            )
            self._webui_capture_thread.start()
            if not config.webui_video_enabled:
                self.logger.info("Web UI video push disabled (webui_video_enabled=False)")

        if config.rtc_enabled:
            self.logger.info(
                f"RTC enabled | lookahead_ratio={config.rtc_lookahead_ratio} | "
                f"blend_steps={config.rtc_blend_steps} | ema_alpha={config.rtc_action_ema_alpha} | "
                f"safety_margin={config.rtc_safety_margin_steps} | latency_window={config.rtc_latency_window}"
            )
        else:
            self.logger.info("RTC disabled, using legacy fixed-threshold mode")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        self.logger.info("Robot connected and ready")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def current_task(self) -> str:
        with self._task_lock:
            return self._current_task

    def set_task(self, task: str) -> None:
        """运行时动态修改任务文本（Web 控制台 / API 调用入口）。"""
        with self._task_lock:
            self._current_task = task
        self.logger.info(f"Task updated at runtime: '{task}'")

    @property
    def task_active(self) -> bool:
        with self._task_active_lock:
            return self._task_active

    def set_task_active(self, active: bool) -> dict:
        """启动/停止任务（Web 控制台 / API 调用入口）。

        停止时：暂停动作执行、清空动作队列与插值状态（防止恢复后执行过期动作）；
        但**观测采集与上传服务器保持运行**（服务器持续收到实时画面流，
        返回的动作块不合并入队，直接丢弃）。
        启动时：队列已清空会立即触发新观测（must_go），新 chunk 从当前状态重新规划。
        """
        with self._task_active_lock:
            self._task_active = bool(active)
        if not active:
            with self.action_queue_lock:
                self.action_queue = Queue()
                self._queue_viz = []
            self._interp_remaining = 0
            with self._ema_action_lock:
                self._ema_prev_action = None
            # 回初始位置：直接发送目标角度（一步到位，舵机加速度寄存器提供斜坡）
            if self.config.home_action:
                try:
                    self.robot.send_action(dict(self.config.home_action))
                    self.logger.info(f"Moved to home position: {self.config.home_action}")
                except Exception as e:
                    self.logger.error(f"Failed to move to home position: {e}")
        else:
            # 恢复执行：队列已清空，下一个观测必须送服务器重新规划
            self.must_go.set()
        self.logger.info(f"Task {'started' if active else 'stopped'} at runtime")
        return {"task_active": self.task_active}

    def get_task_presets(self) -> dict[str, str]:
        """预设任务列表 {任务名称: 任务文本}。"""
        return dict(self._task_presets)

    # ---------------- Web 控制台数据接口 ----------------

    # 允许运行时通过 Web/API 动态调整的 RTC 相关参数及其类型
    _RTC_TUNEABLES: dict = {
        "fps": int,
        "rtc_enabled": bool,
        "rtc_lookahead_ratio": float,
        "rtc_blend_steps": int,
        "rtc_action_ema_alpha": float,
        "rtc_ema_exclude_gripper": bool,
        "rtc_safety_margin_steps": int,
        "rtc_latency_window": int,
        "action_interp_steps": int,
    }

    def get_webui_rtc(self) -> dict:
        return {k: getattr(self.config, k) for k in self._RTC_TUNEABLES}

    def set_webui_rtc(self, params: dict) -> dict:
        """运行时动态修改运行/RTC 参数（下一个控制周期即生效）。

        支持部分字段更新；非法参数名或取值越界会抛出 ValueError，且一个都不改（先全部校验，再一起生效）。
        fps 修改会同步更新控制周期与 FPS 统计目标。
        """
        unknown = set(params) - set(self._RTC_TUNEABLES)
        if unknown:
            raise ValueError(f"unknown params: {sorted(unknown)}; valid: {sorted(self._RTC_TUNEABLES)}")
        validated = {}
        for key, caster in self._RTC_TUNEABLES.items():
            if key not in params:
                continue
            value = params[key]
            if caster is bool:
                if isinstance(value, str):
                    value = value.strip().lower() in ("1", "true", "yes", "on")
                value = bool(value)
            else:
                value = caster(value)
            # 范围校验（与配置 __post_init__ 一致）
            if key == "fps" and value < 1:
                raise ValueError(f"fps must be >= 1, got {value}")
            if key in ("rtc_lookahead_ratio", "rtc_action_ema_alpha") and not (0.0 <= value <= 1.0):
                raise ValueError(f"{key} must be in [0, 1], got {value}")
            if key in ("rtc_blend_steps", "rtc_safety_margin_steps") and value < 0:
                raise ValueError(f"{key} must be >= 0, got {value}")
            if key in ("rtc_latency_window", "action_interp_steps") and value < 1:
                raise ValueError(f"{key} must be >= 1, got {value}")
            validated[key] = value

        for key, value in validated.items():
            setattr(self.config, key, value)
            if key == "rtc_latency_window":
                self.latency_tracker.window_size = value
            if key == "fps":
                self.fps_tracker.target_fps = value

        self.logger.info(f"RTC params updated at runtime: {params}")
        return self.get_webui_rtc()

    def get_webui_status(self) -> dict:
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        with self.latest_action_lock:
            latest_action = self.latest_action
        with self._webui_frames_lock:
            cams = list(self._webui_frames.keys())
        return {
            "running": self.running,
            "task_active": self.task_active,
            "device_name": self.config.device_name,
            "task": self.current_task,
            "queue_size": queue_size,
            "chunk_size": self.action_chunk_size,
            "latest_action": latest_action,
            "actions_sent": self.actions_sent_count,
            "infer_ms": self.latency_tracker.avg_latency_ms,
            "fps": self.config.fps,
            "measured_fps": self._loop_stats.measured_fps,
            "obs_capture_fps": self._loop_stats.obs_capture_fps,
            "action_ms": self._loop_stats.action_ms,
            "obs_ms": self._loop_stats.obs_ms,
            "rtc": self.config.rtc_enabled,
            "transport": {
                "type": self.config.transport,
                "address": self.transport.peer,
            },
            "server": self.config.server_address,
            "cams": cams,
            "video_enabled": self.config.webui_video_enabled,
            # Chunk 使用率 / 动作新鲜度（used_ratio 越低越新鲜；≈ 1 - 触发点/C）
            **self.chunk_usage.snapshot(),
        }

    def get_webui_queue(self) -> dict:
        """返回当前动作队列组成（含新旧 chunk 来源与混合系数），用于融合可视化。"""
        with self.latest_action_lock:
            latest_action = self.latest_action
        with self.action_queue_lock:
            entries = [e for e in self._queue_viz if e["ts"] > latest_action]
        return {"latest": latest_action, "chunk_size": self.action_chunk_size, "entries": entries}

    def get_webui_frame(self, cam: str) -> bytes | None:
        with self._webui_frames_lock:
            return self._webui_frames.get(cam)

    def set_webui_video(self, enabled: bool) -> dict:
        """运行时开关相机图像推流（关闭后立即停止采集编码，节省 CPU）。"""
        self.config.webui_video_enabled = bool(enabled)
        if not enabled:
            with self._webui_frames_lock:
                self._webui_frames.clear()
        self.logger.info(f"Web UI video push {'enabled' if enabled else 'disabled'} at runtime")
        return {"video_enabled": self.config.webui_video_enabled}

    def _webui_capture_loop(self) -> None:
        """后台线程：~10Hz 抓取相机帧并编码（可被 webui_video_enabled 动态开关）。"""
        while self.running:
            if self.config.webui_video_enabled:
                self._capture_webui_frames()
            time.sleep(0.1)

    def _capture_webui_frames(self) -> None:
        """抓取各相机最新帧并编码为 JPEG（供 webui 画面流使用，非阻塞）。"""
        frames = {}
        for cam_key, cam in getattr(self.robot, "cameras", {}).items():
            try:
                with cam.frame_lock:
                    img = cam.latest_frame
                if img is None:
                    continue
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=70)
                frames[cam_key] = buf.getvalue()
            except Exception:
                continue
        if frames:
            with self._webui_frames_lock:
                self._webui_frames.update(frames)

    def start(self):
        """Start the robot client: transport handshake + policy instructions/reset."""
        ok = self.transport.setup()
        if ok:
            self.shutdown_event.clear()
        return ok

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        if self._webui is not None:
            self._webui.stop()

        self.transport.close()
        self.logger.debug("Client stopped")

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Merge incoming actions into the local action queue, aggregating overlapping timesteps.

        Same base logic as LeRobot's robot_client._aggregate_action_queues.
        When RTC is enabled and blend_steps > 0, uses progressive linear blending
        in the overlap region instead of plain aggregation, and records a queue
        composition snapshot (old/new/blend/agg + alpha) for the web UI.
        """
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        # --- RTC: progressive blending setup ---
        use_blending = self.config.rtc_enabled and self.config.rtc_blend_steps > 0
        blend_steps = self.config.rtc_blend_steps if use_blending else 0

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = list(self.action_queue.queue)

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        # Determine overlap region for blending
        if use_blending and current_action_queue and incoming_actions:
            incoming_timesteps = sorted({a.get_timestep() for a in incoming_actions})
            overlap = [t for t in incoming_timesteps if t in current_action_queue]
            # 在重叠区的【开头】（新 chunk 到达后马上要执行的几步）做渐进混合，
            # 让旧计划平滑过渡到新计划。
            blend_region = set(overlap[:blend_steps]) if len(overlap) > 0 else set()
        else:
            blend_region = set()

        # alpha = (idx + 1) / L：新动作权重线性递增，且在混合区【最后一步】恰好达到 1.0，
        # 与后续的纯新动作无缝衔接。
        # 注意不能用 (idx+1)/(L+1)：那样混合区末端 alpha < 1（如 L=2 时仅 0.67），
        # 与后面的纯新动作之间会产生突变。
        if blend_region:
            sorted_blend_ts = sorted(blend_region)

        def _blend_alpha(ts: int) -> float:
            if ts not in blend_region:
                return 1.0  # no blending, use new action fully
            idx = sorted_blend_ts.index(ts)
            return (idx + 1) / len(sorted_blend_ts)

        # Collect incoming timesteps for quick lookup
        incoming_ts_set = {a.get_timestep() for a in incoming_actions}

        # Web 控制台：队列组成快照（每个动作的来源，用于新旧 chunk 融合可视化）
        viz = []

        # First, add all new/updated actions from the incoming chunk
        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # Skip actions older than the latest executed action
            if new_action.get_timestep() <= latest_action:
                continue

            ts = new_action.get_timestep()

            # New timestep: add directly
            if ts not in current_action_queue:
                future_action_queue.put(new_action)
                viz.append({"ts": ts, "src": "new", "alpha": 1.0})
                continue

            # Existing timestep: aggregate (with optional progressive blending)
            old_action = current_action_queue[ts]
            new_raw_action = new_action.get_action()

            if use_blending and ts in blend_region:
                alpha = _blend_alpha(ts)
                final_action = (1 - alpha) * old_action + alpha * new_raw_action
                viz.append({"ts": ts, "src": "blend", "alpha": round(alpha, 3)})
            elif self.config.rtc_enabled:
                # RTC 启用时禁用 aggregate_fn：非混合重叠区直接采用新动作，
                # 与 blend 区末尾（α=1.0）无缝衔接，队列只有 旧/融合/新 三种来源
                final_action = new_raw_action
                viz.append({"ts": ts, "src": "new", "alpha": 1.0})
            else:
                final_action = aggregate_fn(old_action, new_raw_action)
                viz.append({"ts": ts, "src": "agg", "alpha": 1.0})

            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=ts,
                    action=final_action,
                )
            )

        # RTC safety: preserve old actions that are not covered by the incoming chunk
        # and are still in the future. This prevents gaps if server returns a shorter chunk.
        for old_action in internal_queue:
            ts = old_action.get_timestep()
            with self.latest_action_lock:
                if ts <= self.latest_action:
                    continue
            if ts not in incoming_ts_set:
                future_action_queue.put(old_action)
                viz.append({"ts": ts, "src": "old", "alpha": 0.0})

        viz.sort(key=lambda e: e["ts"])
        with self.action_queue_lock:
            self.action_queue = future_action_queue
            self._queue_viz = viz

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the transport (background thread, transport-agnostic).

        统一循环：transport.receive_chunk() 阻塞拿块 → 延迟入滑动窗口 →
        task_active 丢弃 → 合并入队 → must_go.set()。
        """
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            result = self.transport.receive_chunk()
            if result is None:
                continue
            timed_actions, latency_ms = result

            self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
            self.latency_tracker.update(latency_ms)

            # 任务停止期间到达的动作块直接丢弃（观测仍在上传，恢复后重新规划）
            if not self.task_active:
                self.logger.debug("Discarding action chunk: task is stopped")
                continue

            incoming_timesteps = [a.get_timestep() for a in timed_actions]

            if verbose:
                with self.latest_action_lock:
                    latest_action = self.latest_action

                self.logger.info(
                    f"Received action chunk for step #{incoming_timesteps[0]} | "
                    f"Latest action: #{latest_action} | "
                    f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                    f"Latency: {latency_ms:.2f}ms | "
                    f"Avg latency: {self.latency_tracker.avg_latency_ms:.2f}ms"
                )

            # Chunk 使用率：必须在合并【之前】采样队列长度——合并会覆盖旧队列，
            # 之后就读不到"旧 chunk 还剩多少"了。旧 chunk 的未执行长度 = 该采样值。
            with self.action_queue_lock:
                queue_size_before_merge = self.action_queue.qsize()
            self.chunk_usage.settle_previous(queue_size_before_merge, len(timed_actions))

            # Update action queue
            start_time = time.perf_counter()
            self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
            queue_update_time = time.perf_counter() - start_time

            self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

            if verbose:
                self.logger.debug(
                    f"Queue update complete ({queue_update_time:.6f}s) | "
                    f"Updated action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]}"
                )

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """执行一步动作并发送给机器人。

        - 插值细分（action_interp_steps > 1）时，优先执行当前插值段的子步（不消耗队列），
          子步走完后才从队列取下一个目标动作，并从上一目标线性过渡过去；
          队列因此以 FPS/N 的速率消耗，而执行频率仍为 FPS，动作更平滑。
        - RTC: 可选对执行动作做 EMA 低通滤波，抑制高频抖动（夹爪关节透传）。
        """
        n_interp = self.config.action_interp_steps
        timestep = None
        get_end = get_start = 0.0

        if n_interp > 1 and self._interp_remaining > 0:
            # 插值子步：在起点与目标之间线性插值，不消耗队列
            k = n_interp - self._interp_remaining + 1  # 1-indexed
            alpha = k / n_interp
            raw_action_tensor = self._interp_start + alpha * (self._interp_target - self._interp_start)
            self._interp_remaining -= 1
        else:
            # 从队列取下一个目标动作
            get_start = time.perf_counter()
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                try:
                    timed_action = self.action_queue.get_nowait()
                except Empty:
                    # 网页「停止任务」在 actions_available() 之后清空了队列：这一轮不动
                    return None
            get_end = time.perf_counter() - get_start

            new_target = timed_action.get_action()
            timestep = timed_action.get_timestep()

            if n_interp > 1 and self._interp_last_target is not None:
                # 从上一目标到新目标分 N 步过渡，本步先走 1/N
                self._interp_start = self._interp_last_target
                self._interp_target = new_target
                self._interp_remaining = n_interp - 1
                raw_action_tensor = self._interp_start + (1.0 / n_interp) * (new_target - self._interp_start)
            else:
                # 首个动作或未启用插值：直接执行
                raw_action_tensor = new_target

            self._interp_last_target = new_target

            with self.latest_action_lock:
                self.latest_action = timestep

        # RTC: EMA smoothing at execution layer（夹爪关节透传不平滑）
        if self.config.rtc_enabled and self.config.rtc_action_ema_alpha < 1.0:
            with self._ema_action_lock:
                if self._ema_prev_action is None:
                    smoothed_action_tensor = raw_action_tensor
                else:
                    alpha = self.config.rtc_action_ema_alpha
                    blended = alpha * raw_action_tensor + (1 - alpha) * self._ema_prev_action
                    if self._ema_mask is not None:
                        blended = torch.where(self._ema_mask, blended, raw_action_tensor)
                    smoothed_action_tensor = blended
                self._ema_prev_action = smoothed_action_tensor.clone()
        else:
            smoothed_action_tensor = raw_action_tensor

        action = self._action_tensor_to_action_dict(smoothed_action_tensor)
        _performed_action = self.robot.send_action(action)
        self.actions_sent_count += 1
        # Chunk 使用率：记录本 chunk 计量窗口内执行掉的动作数（用于进行中的近似读数）
        if timestep is not None:
            self.chunk_usage.on_action_executed(timestep)

        if verbose and timestep is not None:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()
            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timestep} performed | "
                f"Queue size: {current_queue_size}"
            )
            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Check if the client should send a new observation.

        RTC 自适应触发逻辑：
        - 当 rtc_enabled=True 时，根据历史推理延迟动态计算触发点。
          目标是在旧 chunk 刚好用完前，新 chunk 能及时到达。
        - 当 rtc_enabled=False 时，回退到固定 threshold 逻辑。
        """
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()

        if self.action_chunk_size <= 0:
            return True

        if self.config.rtc_enabled:
            # 插值细分时，每个队列动作可执行 N 个控制步；统一换算成控制步比较
            n_interp = self.config.action_interp_steps
            queue_steps = queue_size * n_interp
            chunk_steps = self.action_chunk_size * n_interp
            # 自适应：估算「触发检查 → 新 chunk 入队」会消耗多少控制步。
            # = 观测采集+发送开销（时间戳之前，单独追踪）
            #   + 推理往返延迟（时间戳→收到动作，取窗口最大值抗抖动）
            #   + 触发检查粒度（每轮最多晚发现 1 步）
            control_dt_ms = self.config.environment_dt * 1000.0
            est_steps = (
                self._obs_overhead_tracker.avg_latency_ms + self.latency_tracker.max_latency_ms
            ) / max(control_dt_ms, 1e-6)
            # 触发点 = 预估消耗步数 + 安全余量（margin 以队列动作计，×N 换算为控制步；
            # 它同时决定新旧 chunk 的重叠长度 = 混合过渡带宽度）
            trigger_point = max(
                1, math.ceil(est_steps + 1.0 + self.config.rtc_safety_margin_steps * n_interp)
            )
            # 下界（floor）：剩余步数掉到 chunk × lookahead_ratio 时就触发，保证观测新鲜度。
            # 方向：ratio 调大 → 下界抬高 → 触发更早 → 每个 chunk 被执行的步数更少
            #       （老块很快被替换，动作更新鲜、抗延迟抖动余量更大，但推理更频繁）；
            #       ratio 调小 → 触发更晚 → 老块被用掉更多步（开环预测尾部更长，误差更大）。
            # 该下界只在超过自适应触发点(est+margin)时生效；否则触发点由延迟决定，
            # 此时调 ratio 不产生任何效果（这解释了"调了没反应"的现象）。
            # 必须用 max：若触发点低于 est+margin，新 chunk 到达前队列就会耗尽，
            # 导致机械臂停顿、chunk 到达后动作突变。
            min_trigger = max(1, int(chunk_steps * self.config.rtc_lookahead_ratio))
            trigger_point = max(trigger_point, min_trigger)
            # 上界不超过 chunk 可执行步数（推理比整个 chunk 还慢时，持续触发推理是最优策略）
            trigger_point = min(trigger_point, chunk_steps)
            return queue_steps <= trigger_point
        else:
            return queue_size / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation | None:
        """采集机器人观测（原始 dict + 当前任务文本），非阻塞提交给传输层。

        本函数不做任何网络 I/O：传输内部只保留最新观测（替换旧值），
        序列化与发送由传输层自己的线程/连接模型完成。
        """
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # 采集耗时（发生在观测时间戳之前，latency_tracker 测不到），
            # 计入触发点估计，避免新 chunk 到达前队列耗尽导致机械臂停顿
            self._obs_overhead_tracker.update(obs_capture_time * 1000)

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            # 非阻塞提交给传输层：内部只保留最新观测（始终发送最新）
            self.transport.submit_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")
            return None

    def control_loop(self, task: str, verbose: bool = False) -> tuple[RawObservation | None, Any]:
        """Combined function for executing actions and streaming observations.

        Main thread loop:
        - (1) perform actions from the queue (paused when the task is stopped);
        - (2) stream observations with the current task text to the remote policy server.
        """
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        # 固定位置刷新的进度条：动态显示 action 队列中的动作余量（队列大小 / chunk 大小）
        queue_bar = _FixedLineQueueBar() if self.config.display_queue_bar else None

        try:
            while self.running:
                control_loop_start = time.perf_counter()

                # (1) Performing actions, when available (and task is active)
                action_ms = None
                if self.task_active and (self.actions_available() or self._interp_remaining > 0):
                    _t = time.perf_counter()
                    _performed_action = self.control_loop_action(verbose)
                    action_ms = (time.perf_counter() - _t) * 1000

                # (2) Streaming observations to the remote policy server.
                #     任务文本每步读取最新值（支持运行时动态修改）；
                #     任务停止时观测采集/上传保持运行（服务器持续收到实时画面流），
                #     仅动作执行暂停、返回的动作块被丢弃。
                obs_ms = None
                if self._ready_to_send_observation():
                    _t = time.perf_counter()
                    _captured_observation = self.control_loop_observation(self.current_task, verbose)
                    obs_ms = (time.perf_counter() - _t) * 1000

                # 注意：相机帧采集在独立后台线程（_webui_capture_loop），不占用控制循环时间

                # (3) 更新动作余量进度条
                if queue_bar is not None:
                    self._update_queue_bar(queue_bar)

                elapsed = time.perf_counter() - control_loop_start
                self.logger.debug(f"Control loop (ms): {elapsed * 1000:.2f}")

                # Dynamically adjust sleep time to maintain the desired control frequency
                time.sleep(max(0, self.config.environment_dt - elapsed))

                # 分段耗时统计（帧率瓶颈定位）
                self._loop_stats.update(action_ms, obs_ms, (time.perf_counter() - control_loop_start) * 1000)
        finally:
            if queue_bar is not None:
                queue_bar.close()

        return _captured_observation, _performed_action

    def _update_queue_bar(self, queue_bar: "_FixedLineQueueBar") -> None:
        """刷新固定位置进度条，展示动作队列余量与推理状态。"""
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        with self.latest_action_lock:
            latest_action = self.latest_action

        snap = self.chunk_usage.snapshot()
        # 已结算值优先；首个 chunk 结算前用进行中的近似值（由 snapshot 统一提供）
        usage_ratio = snap.get("chunk_usage_ratio_display")

        queue_bar.update(
            queue_size=queue_size,
            chunk_size=self.action_chunk_size,
            step=latest_action,
            infer_ms=self.latency_tracker.avg_latency_ms,
            usage_ratio=usage_ratio,
        )


@draccus.wrap()
def run_client(cfg: PolicyRobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = PolicyRobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        action_receiver_thread = threading.Thread(
            target=client.receive_actions, daemon=True, kwargs={"verbose": True}
        )
        action_receiver_thread.start()

        try:
            # 主线程跑控制循环
            client.control_loop(task=cfg.task, verbose=True)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


def main():
    register_third_party_plugins()
    run_client()


if __name__ == "__main__":
    main()
