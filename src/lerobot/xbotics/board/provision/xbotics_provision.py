"""星禾套件端侧板的网络设置页：热点配网，以及连服务器的 SSH 隧道。

端侧板开启一个热点，电脑或手机连接后在浏览器中设置 Wi-Fi 与网线，结果实时显示。热点建在 wlan0 上另建的虚拟接口 ap0
上（AIC8800 支持热点与 Wi-Fi 客户端同时工作、可在不同信道），因此端侧板连接 Wi-Fi 时热点不中断，页面可持续显示连接状态与获得的地址。
DHCP / DNS 用 NetworkManager 共享模式自带的 dnsmasq；dnsmasq-shared.d 中的 address=/#/ 让所有域名都解析到端侧板，
再关闭 ap0 的转发使热点设备无法访问外网，连接热点的电脑或手机检测联网时即弹出本页面。
局域网中也可打开同一页面（http://<端侧板地址>:8001/，或 80 端口）。只依赖系统 Python 标准库与 nmcli，以 root 运行。

页面另有「连接服务器」一栏：填写服务器地址、账号与密码（或私钥），端侧板通过 SSH 隧道将服务器上的推理服务、
TensorBoard 等端口映射到自身的 0.0.0.0，隧道实现见同目录的 xbotics_tunnel.py。

已在星禾套件端侧板（RK3566 + AIC8800，Ubuntu 22.04、NetworkManager 1.36）上实测；
随维护版 LeRobot 安装与升级（python -m lerobot.xbotics.board install），见 XBOTICS.md。
"""

import ipaddress
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 端侧板上由系统 Python 直接运行本文件（不进入 LeRobot 环境），同目录的隧道模块按文件路径导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xbotics_tunnel as tunnel  # noqa: E402

AP_IFACE, STA_IFACE, ETH_IFACE = "ap0", "wlan0", "eth0"
AP_CON = "xbotics-ap"
TRIAL_CON = "xbotics-try"  # 试连新 Wi-Fi 时的临时配置名，连接成功后改为网络名
AP_ADDR = "10.42.0.1"
# 端侧板上的网页服务端口连号（8001–8006），本页在 8001。
# 80 端口提供同一页面：连接热点的设备检测联网只访问 80，弹出本页依赖它；`http://10.42.0.1` 不带端口也可打开。
PORTS = (80, 8001)
PAGE = (Path(__file__).resolve().parent / "index.html").read_bytes()
# 判断能否访问外网：连接两个公共 DNS 的 53 端口，任一可连即视为可访问
ONLINE_PROBES = (("223.5.5.5", 53), ("1.1.1.1", 53))


def nmcli(*args: str, timeout: float = 60) -> tuple[int, str, str]:
    proc = subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr.strip()


def split_terse(line: str) -> list[str]:
    """按未转义的冒号切分 nmcli -t 的一行输出（SSID 中的冒号写作 \\:）。"""
    fields, cur, escaped = [], [], False
    for ch in line:
        if escaped:
            cur.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ":":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur))
    return fields


def ap_ssid() -> str:
    mac = Path(f"/sys/class/net/{STA_IFACE}/address").read_text().strip().replace(":", "")
    return f"Xbotics-{mac[-4:]}"


def ensure_hotspot() -> None:
    """建立虚拟接口 ap0，建立（或更新）并开启热点连接；热点设备不经端侧板访问外网。"""
    if not Path(f"/sys/class/net/{AP_IFACE}").exists():
        subprocess.run(
            ["iw", "dev", STA_IFACE, "interface", "add", AP_IFACE, "type", "__ap"],
            check=True,
        )
        time.sleep(1)
    settings = [
        "connection.interface-name", AP_IFACE,
        "connection.autoconnect", "no",
        "802-11-wireless.ssid", ap_ssid(),
        "802-11-wireless.mode", "ap",
        "802-11-wireless.band", "bg",
        "802-11-wireless.channel", "1",
        "ipv4.method", "shared",
        "ipv4.addresses", f"{AP_ADDR}/24",
        "ipv6.method", "disabled",
    ]  # fmt: skip
    if nmcli("connection", "show", AP_CON)[0] != 0:
        nmcli("connection", "add", "type", "wifi", "con-name", AP_CON, *settings)
    else:
        nmcli("connection", "modify", AP_CON, *settings)
    code, _, err = nmcli("connection", "up", AP_CON)
    if code != 0:
        raise RuntimeError(f"热点未能开启：{err}")
    # 热点仅供网络设置页使用：关闭 ap0 的转发，热点设备无法访问外网，检测联网失败后才会弹出本页。
    # 共享模式激活时会开启全局转发，因此须在 connection up 之后关闭。端侧板上未安装 iptables / nft，
    # 此处只写内核开关，不依赖它们
    Path(f"/proc/sys/net/ipv4/conf/{AP_IFACE}/forwarding").write_text("0\n")


def device_info(iface: str) -> dict:
    info: dict = {"connected": False, "connection": "", "addresses": [], "gateway": "", "dns": []}
    code, out, _ = nmcli(
        "-t", "-f", "GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS,IP4.GATEWAY,IP4.DNS",
        "device", "show", iface,
    )  # fmt: skip
    if code != 0:
        return info
    for line in out.splitlines():
        key, _, value = line.partition(":")
        if key == "GENERAL.STATE":
            info["connected"] = value.startswith("100")
        elif key == "GENERAL.CONNECTION":
            info["connection"] = value
        elif key.startswith("IP4.ADDRESS"):
            info["addresses"].append(value)
        elif key == "IP4.GATEWAY":
            info["gateway"] = value
        elif key.startswith("IP4.DNS"):
            info["dns"].append(value)
    return info


def online_via(iface: str) -> bool:
    for host, port in ONLINE_PROBES:
        try:
            with socket.socket() as s:
                s.settimeout(2)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode())
                s.connect((host, port))
                return True
        except OSError:
            continue
    return False


def eth_connection() -> str | None:
    """网线使用的连接配置：优先取 eth0 上正在使用的，否则取第一条有线配置。"""
    _, out, _ = nmcli("-t", "-f", "NAME,TYPE,DEVICE", "connection", "show")
    rows = [split_terse(line) for line in out.splitlines() if line]
    wired = [r for r in rows if len(r) == 3 and r[1] == "802-3-ethernet"]
    active = [r for r in wired if r[2] == ETH_IFACE]
    return (active or wired or [[None]])[0][0]


def eth_method() -> str:
    name = eth_connection()
    if not name:
        return ""
    _, out, _ = nmcli("-g", "ipv4.method", "connection", "show", name)
    return out.strip()


def saved_wifi() -> list[str]:
    _, out, _ = nmcli("-t", "-f", "NAME,TYPE", "connection", "show")
    rows = [split_terse(line) for line in out.splitlines() if line]
    hidden = {AP_CON, TRIAL_CON}
    return [r[0] for r in rows if len(r) == 2 and r[1] == "802-11-wireless" and r[0] not in hidden]


class Status:
    """后台每 3 秒刷新一次状态（检测外网需要数秒），页面轮询时直接读取缓存。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict = {}
        # at：这条消息的时间戳，页面据此识别用户已关闭的那一条
        self.job: dict = {"kind": "", "state": "idle", "message": "", "at": 0.0}

    def refresh(self) -> None:
        eth, wifi, ap = device_info(ETH_IFACE), device_info(STA_IFACE), device_info(AP_IFACE)
        eth["online"] = eth["connected"] and online_via(ETH_IFACE)
        eth["method"] = eth_method()
        wifi["online"] = wifi["connected"] and online_via(STA_IFACE)
        hostname = socket.gethostname()
        with self.lock:
            self.data = {
                "hostname": hostname,
                "mdns": f"{hostname}.local",
                "ethernet": eth,
                "wifi": wifi,
                "hotspot": {"ssid": ap_ssid(), "address": AP_ADDR, "up": ap["connected"]},
                "saved_wifi": saved_wifi(),
                "time": time.time(),
            }

    def loop(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception as e:  # noqa: BLE001 —— 刷新失败不应使整个服务退出，记录后下一轮重试
                print("状态刷新失败：", e, flush=True)
            time.sleep(3)

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.data, "job": dict(self.job)}

    def set_job(self, kind: str, state: str, message: str) -> None:
        with self.lock:
            self.job = {"kind": kind, "state": state, "message": message, "at": time.time()}


STATUS = Status()


def scan_networks() -> list[dict]:
    _, out, _ = nmcli(
        "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list",
        "ifname", STA_IFACE, "--rescan", "yes",
    )  # fmt: skip
    best: dict[str, dict] = {}
    for line in out.splitlines():
        fields = split_terse(line)
        if len(fields) != 3 or not fields[0]:
            continue
        ssid, signal, security = fields[0], int(fields[1] or 0), fields[2]
        if ssid not in best or signal > best[ssid]["signal"]:
            best[ssid] = {
                "ssid": ssid,
                "signal": signal,
                "security": security,
                "enterprise": "802.1X" in security,
            }
    return sorted(best.values(), key=lambda n: -n["signal"])


def explain(err: str) -> str:
    if "Secrets were required" in err or "802-1x" in err.lower():
        return "密码错误，或该网络需要账号登录"
    if "No network with SSID" in err:
        return "未找到该网络，请靠近路由器后重新扫描"
    return err or "连接失败"


def connect_wifi(ssid: str, password: str) -> None:
    """先以临时名称试连，成功后替换同名的旧配置；失败则删除临时配置并恢复旧配置。

    这样做针对 NetworkManager 1.36 的两处行为（实测）：连接失败时 nmcli 仍返回 0，成败只能看 wlan0
    实际连接的是哪条配置；对已保存的网络直接 `device wifi connect … password` 会改写旧配置的密码，
    输错一次就会让原本可用的配置失效。
    """
    existed = ssid in saved_wifi()
    nmcli("connection", "delete", "id", TRIAL_CON)  # 清理上次残留的临时配置
    args = [
        "--wait",
        "45",
        "device",
        "wifi",
        "connect",
        ssid,
        "ifname",
        STA_IFACE,
        "name",
        TRIAL_CON,
    ]
    if password:
        args += ["password", password]
    STATUS.set_job("wifi", "running", f"正在连接 {ssid}…")
    _, out, err = nmcli(*args, timeout=70)
    info = device_info(STA_IFACE)
    if not (info["connected"] and info["connection"] == TRIAL_CON):
        nmcli("connection", "delete", "id", TRIAL_CON)
        if existed:
            nmcli("--wait", "30", "connection", "up", "id", ssid)
        STATUS.set_job("wifi", "failed", f"连接 {ssid} 失败：{explain(out + err)}")
        return
    if existed:
        nmcli("connection", "delete", "id", ssid)
    nmcli("connection", "modify", TRIAL_CON, "connection.id", ssid)
    time.sleep(2)
    addrs = device_info(STA_IFACE)["addresses"]
    STATUS.set_job("wifi", "done", f"已连接 {ssid}，地址 {', '.join(addrs) or '正在获取'}")


def parse_wired(cfg: dict) -> tuple[str, str, list[str]]:
    """把页面上填写的固定 IP 校验成 nmcli 的写法；哪一项有误，报错就指明哪一项。"""

    def ipv4(label: str, text: object) -> str:
        try:
            return str(ipaddress.IPv4Address(str(text).strip()))
        except ValueError:
            raise ValueError(f"{label}「{text}」不是有效的 IPv4 地址，格式如 192.168.1.50") from None

    address = ipv4("地址", cfg.get("address", ""))
    prefix = str(cfg.get("prefix", "")).strip()
    if not prefix.isdigit() or not 1 <= int(prefix) <= 32:
        raise ValueError(f"前缀长度「{prefix}」应为 1 到 32 之间的整数，家用与办公网络通常为 24")
    gateway = ipv4("网关", cfg.get("gateway", ""))
    dns = [ipv4("DNS", d) for d in cfg.get("dns", []) if str(d).strip()]
    network = ipaddress.IPv4Interface(f"{address}/{prefix}").network
    if ipaddress.IPv4Address(gateway) not in network:
        raise ValueError(f"网关 {gateway} 不在 {network} 网段内，地址与网关须在同一网段")
    return f"{address}/{prefix}", gateway, dns


def apply_wired(cfg: dict) -> None:
    name = eth_connection()
    if not name:
        STATUS.set_job("wired", "failed", "端侧板上没有有线网络的连接配置")
        return
    if cfg.get("mode") == "auto":
        args = ["ipv4.method", "auto", "ipv4.addresses", "", "ipv4.gateway", "", "ipv4.dns", ""]
    else:
        try:
            iface, gateway, dns = parse_wired(cfg)
        except ValueError as e:
            STATUS.set_job("wired", "failed", str(e))
            return
        args = [
            "ipv4.method", "manual",
            "ipv4.addresses", iface,
            "ipv4.gateway", gateway,
            "ipv4.dns", ",".join(dns),
        ]  # fmt: skip
    STATUS.set_job("wired", "running", "正在应用网线设置…")
    code, _, err = nmcli("connection", "modify", name, *args)
    if code != 0:
        STATUS.set_job("wired", "failed", f"网线设置保存失败：{err}")
        return
    # 与连接 Wi-Fi 相同，nmcli 失败时也可能返回 0，成败以 eth0 的实际状态为准
    _, out, err = nmcli("--wait", "30", "connection", "up", name)
    time.sleep(2)
    info = device_info(ETH_IFACE)
    if not (info["connected"] and info["connection"] == name):
        STATUS.set_job("wired", "failed", f"网线设置已保存，但未能连接：{(out + err).strip()}")
        return
    STATUS.set_job("wired", "done", f"网线设置已生效，地址 {', '.join(info['addresses']) or '正在获取'}")


def systemctl(*args: str) -> tuple[int, str]:
    proc = subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=30, check=False)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


# ssh 的英文报错与对应的中文说明。同一次失败会输出多行（口令错误之后还会有一行 Permission denied），
# 因此按此处的先后顺序匹配，越具体的越靠前
TUNNEL_ERRORS = (
    ("incorrect passphrase", "私钥口令不对，请修改后点击「保存并连接」"),
    ("invalid format", "私钥格式不对：请粘贴完整私钥，从 -----BEGIN 行到 -----END 行"),
    ("Permission denied", "登录被拒：账号、密码或私钥不正确，请修改后点击「保存并连接」"),
    ("Could not resolve hostname", "找不到该服务器地址，请检查拼写"),
    ("Connection refused", "服务器拒绝连接：SSH 端口不正确，或服务器未开启 SSH 服务"),
    ("Connection timed out", "连不上服务器：地址不正确，或端侧板所在网络无法访问该服务器"),
    ("No route to host", "连不上服务器：端侧板所在网络无法访问该服务器"),
    ("Network is unreachable", "端侧板尚未联网：请先在「Wi-Fi」或「网线」标签中连接网络"),
    ("cannot listen to port", "端侧板上的端口已被其他程序占用，请更换端侧板端口"),
    ("Address already in use", "端侧板上的端口已被其他程序占用，请更换端侧板端口"),
    (
        "REMOTE HOST IDENTIFICATION HAS CHANGED",
        "服务器指纹已变化（可能重装过系统）：再次点击「保存并连接」即可重新识别",
    ),
    (
        "Host key verification failed",
        "服务器指纹已变化（可能重装过系统）：再次点击「保存并连接」即可重新识别",
    ),
    (
        "administratively prohibited",
        "服务器不允许端口转发：请在服务器的 sshd_config 中设置 AllowTcpForwarding yes",
    ),
)


def tunnel_error(tid: str) -> str:
    """这台服务器的隧道最近一两次启动中 ssh 报告的错误，转成中文说明；没有错误时返回空串。"""
    proc = subprocess.run(
        ["journalctl", "-u", tunnel.unit(tid), "-n", "40", "-o", "cat", "--no-pager"],
        capture_output=True, text=True, timeout=10, check=False,
    )  # fmt: skip
    lines = proc.stdout.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("隧道：")]
    # 只看最近两次启动的输出：上一次失败的原因，以及本次（重连中）的输出
    recent = lines[starts[-2] if len(starts) > 1 else 0 :]
    # 「channel N: open failed」表示某一路转发在服务器端无服务，不属于隧道本身的错误，页面在映射列表中单独标示
    recent = [line for line in recent if not line.startswith("channel ")]
    for needle, text in TUNNEL_ERRORS:
        if any(needle in line for line in recent):
            return text
    return ""


def probe_forward(port: int) -> str:
    """经隧道连接一次端侧板上的该端口，不发送任何数据。

    ok = 服务器端有服务（连接保持打开、等待客户端先发送，或服务先发来数据）；closed = 连接后立即被断开
    （ssh 在服务器端连不上该端口，随即关闭端侧板这一侧的连接）；down = 端侧板上没有监听（隧道未建立）。
    不发送数据，是因为 gRPC 这类协议收到无法识别的数据会直接断开，发送后就无法与「无服务」区分。
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
            s.settimeout(2)
            try:
                return "ok" if s.recv(1) else "closed"
            except TimeoutError:
                return "ok"
            except OSError:
                return "closed"
    except OSError:
        return "down"


class Tunnels:
    """各台服务器的隧道状态。端口探测每台每 15 秒一次（每次探测都会在服务器上留下一条连接记录）。"""

    def __init__(self) -> None:
        self.probes: dict[str, dict[int, str]] = {}
        self.probed_at: dict[str, float] = {}

    def status(self, tid: str) -> dict:
        cfg = tunnel.load_config(tid)
        _, enabled = systemctl("is-enabled", tunnel.unit(tid))
        _, active = systemctl("is-active", tunnel.unit(tid))
        if active == "active" and time.time() - self.probed_at.get(tid, 0.0) > 15:
            self.probes[tid] = {int(f["local"]): probe_forward(int(f["local"])) for f in cfg["forwards"]}
            self.probed_at[tid] = time.time()
        elif active != "active":
            self.probes.pop(tid, None)
        return {
            "id": tid,
            "name": cfg["name"],
            "host": cfg["host"],
            "port": cfg["port"],
            "user": cfg["user"],
            "auth": cfg["auth"],
            "has_password": bool(cfg.get("password")),
            "has_key": tunnel.key_path(tid).exists(),
            "forwards": cfg["forwards"],
            "enabled": enabled == "enabled",
            "state": active,
            "probes": {str(k): v for k, v in self.probes.get(tid, {}).items()},
            "error": tunnel_error(tid) if enabled == "enabled" else "",
        }

    def all(self) -> dict:
        return {
            "servers": [self.status(tid) for tid in tunnel.tunnel_ids()],
            "presets": tunnel.PRESET_FORWARDS,
        }


TUNNELS = Tunnels()


def parse_tunnel(body: dict, old: dict, ports_in_use: set[int]) -> dict:
    """把页面上填写的一台服务器的设置校验成配置；哪一项有误，报错就指明哪一项。

    未重新填写的密码与私钥沿用原值。ports_in_use 是其他服务器已占用的端侧板端口，同一端口不能映射两次。
    """
    name = str(body.get("name", "")).strip()
    host = str(body.get("host", "")).strip()
    user = str(body.get("user", "")).strip()
    if not host:
        raise ValueError("请填写服务器地址")
    if not user:
        raise ValueError("请填写账号")
    port = str(body.get("port") or 22).strip()
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f"SSH 端口「{port}」应为 1 到 65535 之间的整数，通常为 22")
    auth = "key" if body.get("auth") == "key" else "password"
    same_login = old.get("host") == host and old.get("user") == user
    cfg = {"name": name or host, "host": host, "port": int(port), "user": user, "auth": auth}
    if auth == "password":
        password = body.get("password") or (old.get("password") if same_login else "")
        if not password:
            raise ValueError("请填写密码")
        cfg["password"] = password
    else:
        key = str(body.get("key") or "").strip()
        if key and "PRIVATE KEY" not in key:
            raise ValueError(
                "这不是私钥：请粘贴完整私钥，从 -----BEGIN … PRIVATE KEY----- 到 -----END … PRIVATE KEY-----"
            )
        if not key and not (same_login and old.get("has_key_file")):
            raise ValueError("请填写私钥")
        if key:
            cfg["_key"] = key + "\n"
        cfg["passphrase"] = body.get("passphrase") or (
            old.get("passphrase", "") if same_login and not key else ""
        )
    forwards, seen = [], set()
    for i, f in enumerate(body.get("forwards") or [], 1):
        try:
            local, remote = int(f.get("local")), int(f.get("remote") or f.get("local"))
        except (TypeError, ValueError):
            raise ValueError(f"第 {i} 条映射的端口须为数字") from None
        if not (1024 <= local <= 65535 and 1 <= remote <= 65535):
            raise ValueError(f"第 {i} 条映射：端侧板端口须在 1024–65535 之间，服务器端口须在 1–65535 之间")
        if local in seen or local in PORTS or 8001 <= local <= 8007 or local in (9090, 9091):
            raise ValueError(
                f"第 {i} 条映射：端侧板端口 {local} 已被占用（8001–8007 为端侧板自带的网页，9090、9091 为录数据预览）"
            )
        if local in ports_in_use:
            raise ValueError(
                f"第 {i} 条映射：端侧板端口 {local} 已被另一台服务器的映射占用，请更换端侧板端口"
            )
        seen.add(local)
        remote_host = str(f.get("remote_host") or "127.0.0.1").strip()
        forwards.append({"name": str(f.get("name") or f"端口 {local}").strip(), "local": local,
                         "remote_host": remote_host, "remote": remote, "web": f.get("web") is not False})  # fmt: skip
    if not forwards:
        raise ValueError("至少需要映射一个端口")
    cfg["forwards"] = forwards
    return cfg


def write_private(path: Path, text: str) -> None:
    """写入仅 root 可读的文件：创建时即为 0600，不存在其他用户可读的时间窗口。"""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)


def new_tunnel_id() -> str:
    used = {int(t) for t in tunnel.tunnel_ids() if t.isdigit()}
    return str(next(n for n in range(1, len(used) + 2) if n not in used))


def save_tunnel(body: dict) -> None:
    """保存一台服务器的设置并（重新）连接；body 带 id 时修改已有的服务器，不带时新增一台。"""
    tid = str(body.get("id") or "")
    if tid and tid not in tunnel.tunnel_ids():
        STATUS.set_job("tunnel", "failed", "这台服务器的设置已被删除，请刷新页面")
        return
    old = {**tunnel.load_config(tid), "has_key_file": tunnel.key_path(tid).exists()} if tid else {}
    ports_in_use = {
        int(f["local"])
        for other in tunnel.tunnel_ids()
        if other != tid
        for f in tunnel.load_config(other)["forwards"]
    }
    try:
        cfg = parse_tunnel(body, old, ports_in_use)
    except ValueError as e:
        STATUS.set_job("tunnel", "failed", str(e))
        return
    tid = tid or new_tunnel_id()
    STATUS.set_job("tunnel", "running", f"正在连接 {cfg['name']}（{cfg['user']}@{cfg['host']}）…")
    tunnel.CONF_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = cfg.pop("_key", None)
    if key:
        write_private(tunnel.key_path(tid), key)
    # 每次保存都重新识别服务器指纹：更换服务器或服务器重装系统后，点击一次「保存并连接」即可连接
    tunnel.known_hosts_path(tid).unlink(missing_ok=True)
    write_private(tunnel.config_path(tid), json.dumps(cfg, ensure_ascii=False, indent=2))
    systemctl("enable", tunnel.unit(tid))
    code, out = systemctl("restart", tunnel.unit(tid))
    if code != 0:
        STATUS.set_job("tunnel", "failed", f"隧道服务未能启动：{out}")
        return
    TUNNELS.probed_at.pop(tid, None)
    # 登录与建立转发通常需要两三秒；等待端侧板上的端口开始监听，最多 15 秒，超时则读取报错
    for _ in range(15):
        time.sleep(1)
        if all(probe_forward(f["local"]) != "down" for f in cfg["forwards"]):
            names = "、".join(f"{f['name']}（{f['local']}）" for f in cfg["forwards"])
            STATUS.set_job("tunnel", "done", f"已连接 {cfg['name']}：{names} 已映射到端侧板")
            return
    STATUS.set_job("tunnel", "failed", tunnel_error(tid) or "隧道尚未连接，请稍后查看状态")


def stop_tunnel(tid: str) -> None:
    systemctl("disable", "--now", tunnel.unit(tid))
    name = tunnel.load_config(tid).get("name", tid)
    STATUS.set_job("tunnel", "done", f"已断开与 {name} 的连接，开机不再自动连接")


def delete_tunnel(tid: str) -> None:
    name = tunnel.load_config(tid).get("name", tid)
    systemctl("disable", "--now", tunnel.unit(tid))
    for path in (
        tunnel.config_path(tid),
        tunnel.key_path(tid),
        tunnel.known_hosts_path(tid),
        tunnel.CONF_DIR / f"{tid}.askpass",
    ):
        path.unlink(missing_ok=True)
    TUNNELS.probes.pop(tid, None)
    STATUS.set_job("tunnel", "done", f"已删除服务器 {name}")


def in_background(fn, *args) -> None:
    """在后台线程执行一次页面操作；出现未预料的异常时写入日志，并把状态置为失败。

    否则状态会停在「进行中」，之后所有操作都被「上一个操作尚未完成」挡住。
    """

    def task() -> None:
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001 —— 任何异常都要收尾成「失败」，页面才能继续操作
            print(f"{fn.__name__} 出错：{e!r}", file=sys.stderr, flush=True)
            STATUS.set_job(STATUS.snapshot()["job"]["kind"] or "error", "failed", f"操作失败：{e}")

    threading.Thread(target=task, daemon=True).start()


def board_hosts() -> set[str]:
    hostname = socket.gethostname()
    hosts = {AP_ADDR, hostname, f"{hostname}.local", "localhost", "127.0.0.1"}
    for iface in (ETH_IFACE, STA_IFACE):
        hosts |= {a.split("/")[0] for a in device_info(iface)["addresses"]}
    return hosts


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        """不逐条记录请求：页面每 2 秒轮询一次状态，逐条记录会使日志过多。"""
        del format, args

    def send_json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        host = (self.headers.get("Host") or "").split(":")[0]
        if path == "/api/status":
            return self.send_json(STATUS.snapshot())
        if path == "/api/networks":
            return self.send_json(scan_networks())
        if path == "/api/tunnels":
            return self.send_json(TUNNELS.all())
        if path in ("/", "/index.html") and host in board_hosts():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return None
        # 其他请求（系统的联网检测、访问其他网站）一律重定向到本页，系统据此弹出网络设置页
        self.send_response(302)
        self.send_header("Location", f"http://{AP_ADDR}/")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return None

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self.send_json({"error": "请求格式错误"}, 400)
        if STATUS.snapshot()["job"]["state"] == "running":
            return self.send_json({"error": "上一个操作尚未完成"}, 409)
        # 面向用户的结果一律经 STATUS.set_job 发布：页面只从 /api/status 读取消息，只有这一个出口
        if self.path == "/api/wifi":
            ssid = (body.get("ssid") or "").strip()
            if not ssid:
                STATUS.set_job("wifi", "failed", "请先在列表中选择一个 Wi-Fi，或手动填写名称")
            else:
                in_background(connect_wifi, ssid, body.get("password") or "")
        elif self.path == "/api/wifi/forget":
            name = body.get("name") or ""
            code, _, err = nmcli("connection", "delete", "id", name)
            STATUS.set_job("wifi", "done" if code == 0 else "failed", f"已删除 {name}" if code == 0 else err)
        elif self.path == "/api/wired":
            in_background(apply_wired, body)
        elif self.path == "/api/tunnels/save":
            in_background(save_tunnel, body)
        elif self.path in ("/api/tunnels/stop", "/api/tunnels/delete"):
            tid = str(body.get("id") or "")
            if tid not in tunnel.tunnel_ids():
                return self.send_json({"error": "没有这台服务器的设置"}, 404)
            in_background(stop_tunnel if self.path.endswith("/stop") else delete_tunnel, tid)
        else:
            return self.send_json({"error": "接口不存在"}, 404)
        return self.send_json({"ok": True})


def main() -> None:
    ensure_hotspot()
    STATUS.refresh()
    threading.Thread(target=STATUS.loop, daemon=True).start()
    servers = [ThreadingHTTPServer(("0.0.0.0", port), Handler) for port in PORTS]
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        f"配网服务：热点 {ap_ssid()}，页面 http://{AP_ADDR}/（端口 {'、'.join(map(str, PORTS))}）", flush=True
    )
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
