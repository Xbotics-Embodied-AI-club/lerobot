"""星禾套件端侧板配网服务：板子开一个热点，电脑 / 手机连上后在浏览器里配 Wi-Fi 和网线，结果实时显示。

热点开在 wlan0 上另建的虚拟接口 ap0 上（AIC8800 支持热点与连 Wi-Fi 同时开、可在不同信道），
所以板子去连 Wi-Fi 时热点不断，页面能一直显示连没连上、拿到什么地址。
DHCP / DNS 用 NetworkManager 共享模式自带的 dnsmasq；dnsmasq-shared.d 里一条 address=/#/ 让所有域名都指向板子，
再关掉 ap0 的转发让热点设备上不了外网，连上热点的电脑 / 手机检测联网时就会弹出这个页面。
同一个页面在局域网里也能打开（http://<板子地址>/）。只用系统 Python 标准库与 nmcli，以 root 运行。

只在 ZTL A566（RK3566 + AIC8800，厂家 Ubuntu 22.04、NetworkManager 1.36）上实测过；装法见 XBOTICS.md。
"""

import ipaddress
import json
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AP_IFACE, STA_IFACE, ETH_IFACE = "ap0", "wlan0", "eth0"
AP_CON = "xbotics-ap"
TRIAL_CON = "xbotics-try"  # 试连新 Wi-Fi 时的临时配置名，成了再改成网络名
AP_ADDR = "10.42.0.1"
PORT = 80
PAGE = (Path(__file__).resolve().parent / "index.html").read_bytes()
# 用来判断「能不能上外网」：两个公共 DNS 的 53 端口，任一连得上就算通
ONLINE_PROBES = (("223.5.5.5", 53), ("1.1.1.1", 53))


def nmcli(*args: str, timeout: float = 60) -> tuple[int, str, str]:
    proc = subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr.strip()


def split_terse(line: str) -> list[str]:
    """nmcli -t 的一行按没转义的冒号切开（SSID 里的冒号会写成 \\:）。"""
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
    """建虚拟接口 ap0、建（或更新）热点连接并打开；不让热点设备经板子上外网。"""
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
        raise RuntimeError(f"热点没开起来：{err}")
    # 热点设备只给配网页用：关掉 ap0 的转发，它们上不了外网、检测联网失败才会弹出配网页。
    # 共享模式激活时会把全局转发打开，所以要在 connection up 之后关。板子上没装 iptables / nft，
    # 这里只写内核开关，不依赖它们
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
    """网线用的那条连接配置：正在 eth0 上用的优先，否则第一条有线配置。"""
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
    """后台每 3 秒刷一次状态（查外网要几秒），页面轮询时直接读缓存。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict = {}
        # at：这条消息的时刻，页面凭它认出「用户已经点掉的那一条」
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
            except Exception as e:  # noqa: BLE001 —— 刷新失败不能让整个服务退出，记下来下一轮再试
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
        return "密码不对，或这个网络要账号登录"
    if "No network with SSID" in err:
        return "没找到这个网络，靠近一点再扫一次"
    return err or "连接失败"


def connect_wifi(ssid: str, password: str) -> None:
    """先用临时名字试连，成了再顶替同名的旧配置；没成就删掉临时的、换回旧的。

    两个坑（NetworkManager 1.36 实测）：连接失败时 nmcli 照样退出码 0，成败只能看 wlan0 实际连上了哪条；
    对已保存的网络直接 `device wifi connect … password` 会把旧配置的密码改掉，输错一次原来能用的也坏了。
    """
    existed = ssid in saved_wifi()
    nmcli("connection", "delete", "id", TRIAL_CON)  # 上次没收拾干净的
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
        STATUS.set_job("wifi", "failed", f"连 {ssid} 失败：{explain(out + err)}")
        return
    if existed:
        nmcli("connection", "delete", "id", ssid)
    nmcli("connection", "modify", TRIAL_CON, "connection.id", ssid)
    time.sleep(2)
    addrs = device_info(STA_IFACE)["addresses"]
    STATUS.set_job("wifi", "done", f"已连上 {ssid}，地址 {', '.join(addrs) or '还在获取'}")


def parse_wired(cfg: dict) -> tuple[str, str, list[str]]:
    """把页面上填的固定 IP 校验成 nmcli 要的写法；哪一格不对，报错里就点名哪一格。"""

    def ipv4(label: str, text: object) -> str:
        try:
            return str(ipaddress.IPv4Address(str(text).strip()))
        except ValueError:
            raise ValueError(f"{label}「{text}」不是 IPv4 地址，应当像 192.168.1.50 这样") from None

    address = ipv4("地址", cfg.get("address", ""))
    prefix = str(cfg.get("prefix", "")).strip()
    if not prefix.isdigit() or not 1 <= int(prefix) <= 32:
        raise ValueError(f"前缀长度「{prefix}」应当是 1 到 32 的数字，一般的家用 / 办公网是 24")
    gateway = ipv4("网关", cfg.get("gateway", ""))
    dns = [ipv4("DNS", d) for d in cfg.get("dns", []) if str(d).strip()]
    network = ipaddress.IPv4Interface(f"{address}/{prefix}").network
    if ipaddress.IPv4Address(gateway) not in network:
        raise ValueError(f"网关 {gateway} 不在 {network} 这个网段里，地址和网关要在同一个网段")
    return f"{address}/{prefix}", gateway, dns


def apply_wired(cfg: dict) -> None:
    name = eth_connection()
    if not name:
        STATUS.set_job("wired", "failed", "板子上没有网线的连接配置")
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
        STATUS.set_job("wired", "failed", f"网线设置没存上：{err}")
        return
    # 和连 Wi-Fi 一样，nmcli 失败也可能退出码 0：成败看 eth0 实际状态
    _, out, err = nmcli("--wait", "30", "connection", "up", name)
    time.sleep(2)
    info = device_info(ETH_IFACE)
    if not (info["connected"] and info["connection"] == name):
        STATUS.set_job("wired", "failed", f"网线设置已存，但没连上：{(out + err).strip()}")
        return
    STATUS.set_job("wired", "done", f"网线已应用，地址 {', '.join(info['addresses']) or '还在获取'}")


def in_background(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True).start()


def board_hosts() -> set[str]:
    hostname = socket.gethostname()
    hosts = {AP_ADDR, hostname, f"{hostname}.local", "localhost", "127.0.0.1"}
    for iface in (ETH_IFACE, STA_IFACE):
        hosts |= {a.split("/")[0] for a in device_info(iface)["addresses"]}
    return hosts


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        """不逐条记请求：页面每 2 秒轮询一次状态，记下来会把日志刷满。"""
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
        if path in ("/", "/index.html") and host in board_hosts():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return None
        # 其他请求（系统检测联网、打开别的网站）一律跳到配网页，系统就会弹出它
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
            return self.send_json({"error": "请求格式不对"}, 400)
        if STATUS.snapshot()["job"]["state"] == "running":
            return self.send_json({"error": "上一个操作还没完成"}, 409)
        # 给人看的结果一律走 STATUS.set_job：页面只从 /api/status 读消息，只有这一个出口
        if self.path == "/api/wifi":
            ssid = (body.get("ssid") or "").strip()
            if not ssid:
                STATUS.set_job("wifi", "failed", "先在上面的列表里点一个 Wi-Fi，或者手动填名字")
            else:
                in_background(connect_wifi, ssid, body.get("password") or "")
        elif self.path == "/api/wifi/forget":
            name = body.get("name") or ""
            code, _, err = nmcli("connection", "delete", "id", name)
            STATUS.set_job("wifi", "done" if code == 0 else "failed", f"已删掉 {name}" if code == 0 else err)
        elif self.path == "/api/wired":
            in_background(apply_wired, body)
        else:
            return self.send_json({"error": "没有这个接口"}, 404)
        return self.send_json({"ok": True})


def main() -> None:
    ensure_hotspot()
    STATUS.refresh()
    threading.Thread(target=STATUS.loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"配网服务：热点 {ap_ssid()}，页面 http://{AP_ADDR}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
