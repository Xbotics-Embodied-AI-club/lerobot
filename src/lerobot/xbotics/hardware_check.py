"""网页版硬件检测（Xbotics）：浏览器里看所有摄像头的实时画面，并用「插拔对比」找出机械臂的串口。

给没有桌面的板子用：板子上起这个服务，学员在自己电脑的浏览器里打开 ``http://<板子>:8005``，

1. 每个摄像头一张卡片，MJPEG 实时画面，标出协商到的分辨率、像素格式与帧率；
2. 列出当前串口（描述、VID:PID、硬件 ID）；
3. 「插拔对比」：插着时记一次、拔掉后再点一次，消失的那个就是机械臂的串口，替代 ``lerobot-find-port`` 的命令行问答；
4. 「读电机」：扫一条串口上的飞特舵机，读一次位置、电压、温度、电流、负载、力矩开关与状态码（只读）。

板子装了 Xbotics 的 udev 规则时（``/dev/xbotics/*``，见 XBOTICS.md），串口与摄像头旁边一并标出它的固定名字。

移植自 gitee ``zhuzhuangtian/lerobot-v0.5.1-custom`` 的 ``lerobot_tools/01_查看硬件端口.py``；
改动：默认端口 8005（接板子 8001–8004 的连号）；摄像头不再按名字写死过滤，Linux 上改为只列 USB 相机的画面节点
（RK3566 这类芯片自带二十多个 rkisp / rkcif 视频节点，打不开、也不是相机），``--all-cameras`` 列全部；标出固定名字。

用法::

    lerobot-hardware-check                     # 监听全部网卡的 8005
    lerobot-hardware-check --port 8100 --camera-filter "USB HD Camera"
    lerobot-hardware-check --all-cameras       # 连芯片自带的视频节点一起列
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"

DEFAULT_PORT = 8005
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_QUALITY = 80  # 推给网页的 JPEG 质量
# Xbotics udev 规则给板子上的串口与摄像头起的固定名字（按插在哪个口）
FIXED_NAME_DIR = Path("/dev/xbotics")

# 只显示设备名包含这个子串的摄像头（不区分大小写）；空串 = 全部显示。由 --camera-filter 设置。
camera_name_filter = ""
# Linux 上默认只列 USB 相机的画面节点；--all-cameras 时列出全部 /dev/videoN
all_cameras = False
V4L_SYSFS = Path("/sys/class/video4linux")


# ---------------------------------------------------------------------------
# 固定名字
# ---------------------------------------------------------------------------


def fixed_names() -> dict[str, str]:
    """真实设备路径 → 固定名字（``/dev/xbotics/follower`` 等）；没装规则或不是 Linux 时为空。"""
    if not IS_LINUX or not FIXED_NAME_DIR.is_dir():
        return {}
    return {str(link.resolve()): str(link) for link in sorted(FIXED_NAME_DIR.iterdir()) if link.is_symlink()}


# ---------------------------------------------------------------------------
# 摄像头
# ---------------------------------------------------------------------------


def _open_capture(source):
    """按平台选后端打开摄像头。"""
    if IS_LINUX:
        return cv2.VideoCapture(source, cv2.CAP_V4L2)
    if IS_WINDOWS:
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def _configure_capture(cap) -> str:
    """尽量协商成 MJPG + 640×480，把单路 USB 带宽压到 20–40 Mbps，几路挂在同一个集线器下也能同时出画面。

    返回实际协商到的像素格式。Windows DSHOW 下顺序敏感：必须先设分辨率再设 FOURCC，否则 FOURCC 被静默忽略、
    退回未压缩的 YUY2（单路约 150 Mbps，两路同一个集线器就互相挤，表现为一路 30 fps、一路 1 fps 甚至黑屏）。
    所以设完回读一次，不对再补设。
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    if int(cap.get(cv2.CAP_PROP_FOURCC)) != mjpg:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    actual = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((actual >> (8 * i)) & 0xFF) for i in range(4))


def _device_name(source) -> str:
    """摄像头的设备名（如 "USB HD Camera"）。

    Linux 读 sysfs；Windows 用 pygrabber 枚举 DirectShow 设备，索引与 ``cv2.CAP_DSHOW`` 一一对应。
    """
    if IS_LINUX:
        m = re.search(r"(\d+)$", str(source))
        if m:
            p = Path(f"/sys/class/video4linux/video{m.group(1)}/name")
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace").strip()
        return ""
    if IS_WINDOWS:
        try:
            idx = int(source)
        except (TypeError, ValueError):
            return ""
        return _windows_video_names().get(idx, "")
    return ""


_win_video_names: dict[int, str] | None = None  # None = 还没枚举，{} = 枚举失败


def _windows_video_names() -> dict[int, str]:
    """DSHOW 索引 → 设备名，枚举一次后缓存。COM 枚举在有设备被占用时偶发失败，重试几次再放弃。"""
    global _win_video_names
    if _win_video_names is None:
        try:
            from pygrabber.dshow_graph import FilterGraph

            for attempt in range(3):
                try:
                    _win_video_names = dict(enumerate(FilterGraph().get_input_devices()))
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(0.5)
        except Exception as exc:  # 没装 pygrabber 或枚举失败
            print(f"[警告] 无法枚举 DirectShow 摄像头设备名（{exc}），Windows 下请安装 pygrabber")
            _win_video_names = {}
    return _win_video_names or {}


def is_usb_capture_node(node: str) -> bool:
    """``videoN`` 是不是 USB 相机出画面的那个节点：挂在 USB 上，且 index 为 0（UVC 相机的 index 1 是元数据节点）。"""
    sysfs = V4L_SYSFS / node
    try:
        on_usb = any(re.fullmatch(r"usb\d+", part) for part in (sysfs / "device").resolve().parts)
        return on_usb and (sysfs / "index").read_text().strip() == "0"
    except OSError:
        return False


def _camera_candidates(max_index: int = 10) -> list:
    """Linux 枚举 /dev/videoN（默认只留 USB 相机的画面节点），其他平台探测数字索引。"""
    if IS_LINUX:
        numbered = [
            (int(m.group(1)), p)
            for p in Path("/dev").glob("video*")
            if (m := re.fullmatch(r"video(\d+)", p.name))
        ]
        return [str(p) for _, p in sorted(numbered) if all_cameras or is_usb_capture_node(p.name)]
    return list(range(max_index))


def detect_cameras() -> list[dict]:
    """探测所有能用的摄像头，返回 ``[{source, name, fixed_name, width, height, format?, error?}, ...]``。"""
    aliases = fixed_names()
    cameras = []
    for source in _camera_candidates():
        name = _device_name(source)
        # 有的摄像头会多出一个只带元数据、没有画面的节点
        if "metadata" in name.lower():
            continue
        if camera_name_filter and camera_name_filter.lower() not in name.lower():
            if name:
                continue
            if IS_WINDOWS and _windows_video_names():
                continue  # 索引不在 DirectShow 设备列表里，探测越界
            # 读不到设备名就判断不了，保留并提示
            print(f"[警告] 无法读取 {source} 的设备名，名称过滤未生效，已保留显示")
        cap = _open_capture(source)
        if not cap.isOpened():  # DSHOW 偶发第一次打不开，重试一次
            cap.release()
            time.sleep(0.3)
            cap = _open_capture(source)
        entry = {
            "source": str(source),
            "name": name,
            "fixed_name": aliases.get(str(source), ""),
            "width": 0,
            "height": 0,
        }
        if cap.isOpened():
            entry["format"] = _configure_capture(cap)
            ok, _ = cap.read()
            if ok:
                entry["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                entry["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                entry["error"] = "读取失败（可能被其他程序占用或 USB 带宽不足）"
            cap.release()
            cameras.append(entry)
        elif name:
            # 枚举看得见、打不开：照样列出卡片，网页那边持续自动重试
            entry["error"] = "打开失败（可能被其他程序占用或 USB 带宽不足）"
            cameras.append(entry)
        else:
            cap.release()
    return cameras


class CameraStreamer:
    """一个摄像头的后台取流线程，几个网页同时看时共享最新一帧；没人看了就停。"""

    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.cond = threading.Condition()
        self.latest_jpeg: bytes | None = None
        self.running = False
        self.thread: threading.Thread | None = None
        self.viewers = 0
        self.fps = 0.0
        self.status = "未启动"
        self.format = ""  # 实际协商到的像素格式（MJPG / YUY2 / ...）

    def _sleep_breakable(self, seconds: float) -> None:
        """睡眠，但看的人走了（running=False）就及时醒来。"""
        slept = 0.0
        while self.running and slept < seconds:
            time.sleep(0.1)
            slept += 0.1

    def _loop(self) -> None:
        retry_interval = 3.0  # 打不开或断流后隔多久重试（秒）
        cap = None
        frames = 0
        fps_window_start = time.monotonic()
        fail_streak = 0
        try:
            while self.running:
                if cap is None:
                    cap = _open_capture(self.source)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        # 设备在、打不开：多半被别的程序占着，或与占用中的设备同挂一个集线器、带宽不够
                        self.status = (
                            f"打开失败（可能被其他程序占用或 USB 带宽不足），{retry_interval:.0f}s 后重试"
                        )
                        with self.cond:
                            self.cond.notify_all()
                        self._sleep_breakable(retry_interval)
                        continue
                    self.format = _configure_capture(cap)
                    frames = 0
                    fps_window_start = time.monotonic()
                    fail_streak = 0
                    self.status = "推流中"
                ok, frame = cap.read()
                if not ok:
                    fail_streak += 1
                    if fail_streak >= 100:  # 一直读不到帧：关掉重开（可能拔了或被抢占）
                        cap.release()
                        cap = None
                        continue
                    self.status = "读取失败（疑似 USB 带宽不足或被占用）"
                    time.sleep(0.05)
                    continue
                fail_streak = 0
                frames += 1
                now = time.monotonic()
                if now - fps_window_start >= 1.0:
                    self.fps = frames / (now - fps_window_start)
                    frames = 0
                    fps_window_start = now
                self.status = "推流中"
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
                if not ok:
                    continue
                with self.cond:
                    self.latest_jpeg = buf.tobytes()
                    self.cond.notify_all()
        finally:
            if cap is not None:
                cap.release()
            self.status = "已停止"
            with self.cond:
                self.latest_jpeg = None
                self.cond.notify_all()

    def add_viewer(self) -> None:
        with self.cond:
            self.viewers += 1
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()

    def remove_viewer(self) -> None:
        with self.cond:
            self.viewers = max(0, self.viewers - 1)
            if self.viewers == 0:
                self.running = False
                self.cond.notify_all()


_streamers: dict[str, CameraStreamer] = {}
_streamers_lock = threading.Lock()


def get_streamer(source: str) -> CameraStreamer:
    with _streamers_lock:
        if source not in _streamers:
            _streamers[source] = CameraStreamer(source)
        return _streamers[source]


# ---------------------------------------------------------------------------
# 串口
# ---------------------------------------------------------------------------


def list_serial_ports() -> list[dict]:
    """列出串口，返回 ``[{device, fixed_name, description, hwid, vid_pid}, ...]``。"""
    aliases = fixed_names()
    try:
        from serial.tools import list_ports
    except ImportError:
        # 没装 pyserial 时退回 lerobot-find-port 的枚举办法
        if IS_WINDOWS:
            return []
        return [
            {
                "device": str(p),
                "fixed_name": aliases.get(str(p), ""),
                "description": "",
                "hwid": "",
                "vid_pid": None,
            }
            for p in sorted(Path("/dev").glob("tty*"))
        ]
    ports = []
    for p in list_ports.comports():
        vid_pid = f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None and p.pid is not None else None
        ports.append(
            {
                "device": p.device,
                "fixed_name": aliases.get(p.device, ""),
                "description": p.description,
                "hwid": p.hwid,
                "vid_pid": vid_pid,
            }
        )
    # pyserial 在 Linux 上会滤掉挂在 platform 总线上的 ttyS*，而 RK3566 板子的两条臂正接在板载串口上：
    # 有固定名字的串口不在列表里就补上
    listed = {port["device"] for port in ports}
    for real, alias in aliases.items():
        if Path(real).name.startswith("tty") and real not in listed:
            ports.append(
                {"device": real, "fixed_name": alias, "description": "板载串口", "hwid": "", "vid_pid": None}
            )
    return ports


# ---------------------------------------------------------------------------
# 电机
# ---------------------------------------------------------------------------

MOTOR_BAUDRATE = 1_000_000  # SO-101 的飞特舵机出厂波特率
# 读哪些寄存器、怎么换算成页面上的读数（STS3215：电压 0.1 V/单位、电流 6.5 mA/单位、负载 0.1 %/单位）
MOTOR_REGISTERS = {
    "Present_Position": ("position", 1),
    "Present_Voltage": ("voltage_v", 0.1),
    "Present_Temperature": ("temperature_c", 1),
    "Present_Current": ("current_ma", 6.5),
    "Present_Load": ("load_pct", 0.1),
    "Torque_Enable": ("torque_on", 1),
    "Status": ("status", 1),
}


_port_locks: dict[str, threading.Lock] = {}
_port_locks_guard = threading.Lock()


def _port_lock(port: str) -> threading.Lock:
    """同一条串口同一时刻只读一次：连点两下「读电机」不该让本进程自己把串口打开两次。"""
    with _port_locks_guard:
        return _port_locks.setdefault(port, threading.Lock())


def read_motors(port: str) -> dict:
    """扫一条飞特总线上的舵机，逐个读当前状态。只读不写：不改力矩、不改任何参数，断开时也不卸力。

    用底层读法（报错不抛）：舵机在报警（过载、过热、电压异常）时回包里带报警位，上层的
    ``broadcast_ping`` / ``read`` 会把它整个跳过或抛错 —— 而这正是最想看到的那一个。
    报警位单独放在 ``alarm``（及 ``alarm_text``）里。

    和正在用这条串口的程序（lerobot-record、推理客户端、网页标定）同时读会互相打乱数据，由用的人避开。
    """
    import scservo_sdk as scs

    from lerobot.motors.encoding_utils import decode_sign_magnitude
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.feetech.tables import MODEL_NUMBER
    from lerobot.motors.motors_bus import get_address

    with _port_lock(port):
        bus = None
        try:
            bus = FeetechMotorsBus(port, motors={})
            model_names = {number: name for name, number in bus.model_number_table.items()}
            bus.connect(handshake=False)
            bus.set_baudrate(MOTOR_BAUDRATE)
            ids_status, comm = {}, None
            for _ in range(3):
                ids_status, comm = bus._broadcast_ping()
                if bus._is_comm_success(comm):
                    break
            if not ids_status:
                if comm is not None and not bus._is_comm_success(comm) and comm != scs.COMM_RX_TIMEOUT:
                    return {
                        "port": port,
                        "motors": [],
                        "error": f"总线通信失败：{bus.packet_handler.getTxRxResult(comm)}",
                    }
                return {
                    "port": port,
                    "motors": [],
                    "error": "这条总线上没有扫到舵机：检查舵机供电、接线，以及串口是否选对",
                }

            motors = []
            for id_, ping_status in sorted(ids_status.items()):
                alarm = ping_status
                number, comm, error = bus._read(*MODEL_NUMBER, id_, num_retry=2, raise_on_error=False)
                alarm |= error
                model = model_names.get(number) if bus._is_comm_success(comm) else None
                entry = {
                    "id": id_,
                    "model": model or (f"未知型号 {number}" if bus._is_comm_success(comm) else "读不到"),
                }
                if model:
                    encodings = bus.model_encoding_table.get(model, {})
                    for register, (key, scale) in MOTOR_REGISTERS.items():
                        addr, length = get_address(bus.model_ctrl_table, model, register)
                        value, comm, error = bus._read(addr, length, id_, num_retry=2, raise_on_error=False)
                        alarm |= error
                        if not bus._is_comm_success(comm):
                            entry[key] = None
                            continue
                        if register in encodings:
                            value = decode_sign_magnitude(value, encodings[register])
                        entry[key] = round(value * scale, 1) if scale != 1 else int(value)
                entry["alarm"] = alarm
                entry["alarm_text"] = bus.packet_handler.getRxPacketError(alarm) if alarm else ""
                motors.append(entry)
            return {"port": port, "motors": motors, "error": None}
        except Exception as error:
            return {"port": port, "motors": [], "error": f"打不开串口或读取中断：{error}"}
        finally:
            if bus is not None and bus.is_connected:
                bus.disconnect(disable_torque=False)


# ---------------------------------------------------------------------------
# 网页
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>摄像头 / 串口检测</title>
<style>
  body { font-family: sans-serif; margin: 0; padding: 16px; background: #1e1e1e; color: #ddd; }
  h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
  button { background: #0d6efd; color: #fff; border: none; border-radius: 4px;
           padding: 6px 14px; margin: 4px 4px 4px 0; cursor: pointer; font-size: 14px; }
  button.secondary { background: #495057; }
  button:hover { opacity: .85; }
  .grid { display: flex; flex-wrap: wrap; gap: 12px; }
  .card { background: #2b2b2b; border-radius: 6px; padding: 10px; }
  .card img { display: block; max-width: 480px; width: 100%; background: #000; border-radius: 4px; }
  .card .title { font-size: 13px; margin-bottom: 6px; color: #9ecbff; word-break: break-all; }
  .fixed { color: #4ade80; }
  table { border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th, td { border: 1px solid #444; padding: 5px 10px; text-align: left; }
  th { background: #333; }
  .mono { font-family: monospace; }
  .ok { color: #4ade80; } .warn { color: #facc15; } .err { color: #f87171; }
  #diff-result { margin-top: 10px; font-size: 14px; }
  .hint { color: #888; font-size: 12px; }
</style>
</head>
<body>
<h1>摄像头 / 串口 网页检测工具</h1>

<h2>摄像头实时画面</h2>
<button onclick="loadCameras()">重新扫描摄像头</button>
<span id="cam-status" class="hint"></span>
<div id="cameras" class="grid"></div>

<h2>串口设备</h2>
<button onclick="loadPorts()">刷新串口列表</button>
<span class="hint">找机械臂端口号：</span>
<button class="secondary" onclick="snapshotPorts()">① 记录基准（插着 USB 时点击）</button>
<button class="secondary" onclick="diffPorts()">② 拔掉 USB 后点击，对比差异</button>
<div id="diff-result"></div>
<table id="ports-table">
  <thead><tr><th>端口</th><th>固定名字</th><th>描述</th><th>VID:PID</th><th>硬件 ID</th><th>电机</th></tr></thead>
  <tbody></tbody>
</table>

<h2>电机状态</h2>
<span class="hint">点串口那一行的「读电机」，扫这条总线上的舵机并读一次当前状态；只读不写，不改力矩。报警的舵机也会列出来，报警原因在最后一列。
录数据、推理、网页标定正在用这条臂时别点，同时读会互相打乱数据。</span>
<div id="motors-result"></div>

<script>
async function loadCameras() {
  const status = document.getElementById('cam-status');
  status.textContent = ' 扫描中…';
  const res = await fetch('/api/cameras?refresh=1');
  const cams = await res.json();
  status.textContent = ` 共 ${cams.length} 个`;
  const box = document.getElementById('cameras');
  box.innerHTML = '';
  for (const c of cams) {
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.src = c.source;
    const src = encodeURIComponent(c.source);
    const size = c.width > 0 ? `　${c.width}×${c.height}${c.format ? ' ' + c.format : ''}` : '';
    const fixed = c.fixed_name ? `　<span class="fixed">${c.fixed_name}</span>` : '';
    const initStatus = c.error
      ? `<div class="stream-status err">${c.error}，持续重试中…</div>`
      : `<div class="stream-status hint">连接中…</div>`;
    card.innerHTML =
      `<div class="title mono">${c.name || '(未知设备)'}　${c.source}${fixed}${size}</div>` +
      initStatus +
      // 打不开的设备也挂上流地址：服务端每 3 秒自动重试，恢复后画面自动出现
      `<img src="/stream?src=${src}" alt="${c.source}" style="opacity:0"` +
      ` onload="this.style.opacity=1" onerror="this.style.opacity=0">`;
    box.appendChild(card);
  }
  if (cams.length === 0) {
    box.innerHTML = '<span class="warn">未检测到摄像头</span>';
  }
}

async function updateStreamStats() {
  for (const card of document.querySelectorAll('#cameras .card')) {
    const src = card.dataset.src;
    if (!src) continue;
    try {
      const res = await fetch('/api/stream_stats?src=' + encodeURIComponent(src));
      const st = await res.json();
      const el = card.querySelector('.stream-status');
      if (el) {
        const fmt = st.format ? `　${st.format}` : '';
        el.textContent = st.fps > 0.05 ? `${st.status}　${st.fps.toFixed(1)} fps${fmt}` : st.status;
        el.className = 'stream-status hint' +
          (st.status === '推流中' ? ' ok' : (st.status === '未启动' ? '' : ' err'));
      }
    } catch (e) { /* 轮询偶尔失败不管 */ }
  }
}
setInterval(updateStreamStats, 1000);

async function fetchPorts() {
  const res = await fetch('/api/ports');
  return res.json();
}

function renderPorts(ports) {
  const tbody = document.querySelector('#ports-table tbody');
  tbody.innerHTML = '';
  for (const p of ports) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class="mono"><b>${p.device}</b></td>` +
      `<td class="mono fixed">${p.fixed_name || ''}</td>` +
      `<td>${p.description || ''}</td>` +
      `<td class="mono">${p.vid_pid || ''}</td>` +
      `<td class="mono">${p.hwid || ''}</td>` +
      `<td><button class="secondary" onclick="readMotors('${p.device}', '${p.fixed_name || ''}')">读电机</button></td>`;
    tbody.appendChild(tr);
  }
}

async function loadPorts() { renderPorts(await fetchPorts()); }

let baseline = null;
async function snapshotPorts() {
  baseline = await fetchPorts();
  renderPorts(baseline);
  document.getElementById('diff-result').innerHTML =
    `<span class="ok">已记录基准（${baseline.length} 个端口）。现在请拔掉机械臂的 USB 线，然后点击 ②。</span>`;
}

async function diffPorts() {
  if (!baseline) {
    document.getElementById('diff-result').innerHTML =
      '<span class="err">请先点击 ① 记录基准。</span>';
    return;
  }
  const now = await fetchPorts();
  renderPorts(now);
  const nowSet = new Set(now.map(p => p.device));
  const gone = baseline.filter(p => !nowSet.has(p.device));
  const el = document.getElementById('diff-result');
  if (gone.length === 1) {
    el.innerHTML = `<span class="ok">✔ 检测成功！机械臂端口是 <b class="mono">${gone[0].device}</b>` +
      `（${gone[0].description || '无描述'}）。请重新插回 USB 线。</span>`;
  } else if (gone.length === 0) {
    el.innerHTML = '<span class="warn">未检测到端口变化，请确认 USB 已拔掉后再试。</span>';
  } else {
    el.innerHTML = `<span class="warn">检测到 ${gone.length} 个端口消失：` +
      gone.map(p => `<b class="mono">${p.device}</b>`).join('、') +
      '，请只拔掉机械臂的 USB 后重试。</span>';
  }
}

let readingMotors = false;
async function readMotors(port, fixedName) {
  if (readingMotors) return;
  readingMotors = true;
  const buttons = document.querySelectorAll('#ports-table button');
  buttons.forEach(b => b.disabled = true);
  const box = document.getElementById('motors-result');
  const label = fixedName ? `${port}（${fixedName}）` : port;
  box.innerHTML = `<span class="hint">正在读 ${label} …</span>`;
  try {
    const res = await fetch('/api/motors?port=' + encodeURIComponent(port));
    if (!res.ok) throw new Error(`HTTP ${res.status}（串口可能已拔掉，先点「刷新串口列表」）`);
    const r = await res.json();
    const fmt = v => (v === null || v === undefined) ? '—' : v;
    let html = `<div class="title mono" style="margin-top:8px">${label}：扫到 ${r.motors.length} 个舵机</div>`;
    if (r.error) html += `<div class="err">${r.error}</div>`;
    if (r.motors.length) {
      html += '<table><thead><tr><th>ID</th><th>型号</th><th>位置（0–4095）</th><th>电压 V</th>' +
        '<th>温度 ℃</th><th>电流 mA</th><th>负载 %</th><th>力矩</th><th>状态码</th><th>报警</th></tr></thead><tbody>';
      for (const m of r.motors) {
        const hot = m.temperature_c !== null && m.temperature_c !== undefined && m.temperature_c >= 60;
        html += `<tr><td class="mono">${m.id}</td><td>${m.model}</td><td class="mono">${fmt(m.position)}</td>` +
          `<td class="mono">${fmt(m.voltage_v)}</td><td class="mono ${hot ? 'warn' : ''}">${fmt(m.temperature_c)}</td>` +
          `<td class="mono">${fmt(m.current_ma)}</td><td class="mono">${fmt(m.load_pct)}</td>` +
          `<td>${m.torque_on === null || m.torque_on === undefined ? '—' : (m.torque_on ? '开' : '关')}</td>` +
          `<td class="mono ${m.status ? 'err' : ''}">${fmt(m.status)}</td>` +
          `<td class="${m.alarm ? 'err' : ''}">${m.alarm ? m.alarm_text || m.alarm : '无'}</td></tr>`;
      }
      html += '</tbody></table>';
    }
    box.innerHTML = html;
  } catch (e) {
    box.innerHTML = `<span class="err">读取失败：${e.message}</span>`;
  } finally {
    readingMotors = false;
    buttons.forEach(b => b.disabled = false);
  }
}

loadCameras();
loadPorts();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/":
            self._send_html(INDEX_HTML)
        elif parsed.path == "/api/cameras":
            self._send_json(detect_cameras())
        elif parsed.path == "/api/stream_stats":
            src = query.get("src", [""])[0]
            streamer = _streamers.get(src)
            if streamer is None:
                self._send_json({"fps": 0.0, "status": "未启动", "format": ""})
            else:
                self._send_json(
                    {"fps": round(streamer.fps, 1), "status": streamer.status, "format": streamer.format}
                )
        elif parsed.path == "/api/ports":
            self._send_json(list_serial_ports())
        elif parsed.path == "/api/motors":
            port = query.get("port", [""])[0]
            if port not in {p["device"] for p in list_serial_ports()}:
                self.send_error(400, "unknown port")
                return
            self._send_json(read_motors(port))
        elif parsed.path == "/stream":
            src = query.get("src", [None])[0]
            if src is None:
                self.send_error(400, "missing src")
                return
            self._handle_stream(src)
        else:
            self.send_error(404)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_stream(self, source: str) -> None:
        streamer = get_streamer(source)
        streamer.add_viewer()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            while True:
                with streamer.cond:
                    if not streamer.cond.wait(timeout=10) and not streamer.running:
                        break  # 摄像头打不开
                    data = streamer.latest_jpeg
                if data is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        except OSError:
            pass  # 浏览器断开（BrokenPipe / ConnectionReset 等）
        finally:
            streamer.remove_viewer()

    def log_message(self, format, *args):  # noqa: A002
        pass


def port_available(host: str, port: int) -> bool:
    """端口能不能绑上。占着时先报错，免得浏览器连到别的程序。

    Windows 允许不带 SO_EXCLUSIVEADDRUSE 的通配地址（0.0.0.0）与已绑定的具体地址重叠 ——
    表现为端口明明被别的程序占着、预检却通过、浏览器实际连到了别的程序，所以 Windows 下要设它。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform == "win32":
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        # 与 ThreadingHTTPServer 一样设 SO_REUSEADDR：刚关掉的服务留下的 TIME_WAIT 连接不该挡住重启
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main() -> None:
    global camera_name_filter, all_cameras
    parser = argparse.ArgumentParser(description="网页版摄像头与串口检测")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0（局域网可访问）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口，默认 {DEFAULT_PORT}")
    parser.add_argument(
        "--camera-filter",
        default="",
        help='只显示设备名包含这个子串的摄像头（不区分大小写），如 "USB HD Camera"；默认全部显示',
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Linux 上列出全部 /dev/videoN（默认只列 USB 相机出画面的节点）",
    )
    args = parser.parse_args()
    camera_name_filter = args.camera_filter
    all_cameras = args.all_cameras

    if not port_available(args.host, args.port):
        sys.exit(f"[错误] 端口 {args.port}（{args.host}）已被占用，请用 --port 换一个端口，或先释放它")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    # 舵机驱动在 RK3566 上首次导入要十几秒：后台先导入，页面上第一次点「读电机」就不用等
    threading.Thread(target=lambda: __import__("lerobot.motors.feetech"), daemon=True).start()

    print("=" * 50)
    print("服务已启动，请在浏览器打开:")
    print(f"  本机:   http://127.0.0.1:{args.port}")
    print(f"  局域网: http://<这台机器的 IP>:{args.port}")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
