"""把星禾套件端侧板安装（或升级）为本版维护版 LeRobot 对应的状态。

端侧板上需要维护的内容全部在本包中，升级 LeRobot 即升级全部。一次安装下列各项，重复运行只修改有变化的部分
（以 root 在端侧板的 LeRobot 环境中运行）：

- 系统：开机不进入桌面、CPU 与内存锁定最高频率、相机不自动降帧、提高 inotify 监视数上限（VS Code 需要）、
  机械臂与相机的固定设备名（udev）
- 网络设置页（8001，热点配网 + 连服务器的 SSH 隧道）
- 网页终端 ttyd（8002）、JupyterLab（8003）、浏览器版 VS Code（8004，版本钉在 ``CODE_SERVER_VERSION``）
- 网页硬件检测（8005）、网页标定（8006）与开关命令 ``xbotics-config``
- 端侧板 LeRobot 环境 ``/root/xbotics`` 的 pyproject：只修改 ``board_env/pyproject.toml`` 所管理的几项，其余保持不变

首次安装开启 8002–8006 五个网页服务；之后再次安装（升级）保持 ``xbotics-config`` 中的开关不变。
正在运行的服务有改动时重启；运行 LeRobot 代码的服务（JupyterLab、硬件检测、网页标定）在 LeRobot 版本变化时重启。
重启延后 3 秒，并放在单独的 systemd 单元中执行：在网页终端或 VS Code 终端中升级时，重启这些服务不会中断升级本身。
已经设置过的密码（JupyterLab、VS Code）不会被覆盖。

    python -m lerobot.xbotics.board install
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

BOARD = Path(__file__).resolve().parent
ROOT_HOME = Path("/root")
ENV_DIR = ROOT_HOME / "xbotics"
UV = ROOT_HOME / ".local/bin/uv"
MARKER = Path("/etc/xbotics/installed")  # 首次安装后写入（内容为当时安装的 LeRobot 版本）；存在即视为升级
DEFAULT_PASSWORD = "123456"  # 与端侧板 root 密码相同：网页终端、JupyterLab、VS Code 共用一个密码
CODE_SERVER_VERSION = "4.141.0"
CODE_SERVER_EXTENSIONS = ("ms-python.python", "detachhead.basedpyright", "tamasfe.even-better-toml")
APT_PACKAGES = ("ttyd", "iw", "whiptail")
SYSTEMD = Path("/etc/systemd/system")

# (包内文件, 安装位置, 权限, 使用它的 systemd 单元)
FILES = [
    ("udev/99-xbotics-rk3566.rules", "/etc/udev/rules.d/99-xbotics-rk3566.rules", 0o644, "udev"),
    ("udev/99-xbotics-camera-fps.rules", "/etc/udev/rules.d/99-xbotics-camera-fps.rules", 0o644, "udev"),
    ("system/xbotics-maxfreq.service", SYSTEMD / "xbotics-maxfreq.service", 0o644, "xbotics-maxfreq"),
    ("system/99-xbotics-inotify.conf", "/etc/sysctl.d/99-xbotics-inotify.conf", 0o644, "sysctl"),
    ("provision/xbotics_provision.py", "/opt/xbotics/provision/xbotics_provision.py", 0o644, "xbotics-provision"),
    ("provision/xbotics_tunnel.py", "/opt/xbotics/provision/xbotics_tunnel.py", 0o644, "xbotics-provision xbotics-tunnel"),
    ("provision/index.html", "/opt/xbotics/provision/index.html", 0o644, "xbotics-provision"),
    ("provision/xbotics-captive.conf", "/etc/NetworkManager/dnsmasq-shared.d/xbotics-captive.conf", 0o644, "xbotics-provision"),
    ("provision/xbotics-provision.service", SYSTEMD / "xbotics-provision.service", 0o644, "xbotics-provision"),
    ("provision/xbotics-tunnel@.service", SYSTEMD / "xbotics-tunnel@.service", 0o644, "xbotics-tunnel"),
    ("config/xbotics-config", "/usr/local/bin/xbotics-config", 0o755, ""),
    ("config/xbotics-hardware-check.service", SYSTEMD / "xbotics-hardware-check.service", 0o644, "xbotics-hardware-check"),
    ("config/xbotics-calibrate-web.service", SYSTEMD / "xbotics-calibrate-web.service", 0o644, "xbotics-calibrate-web"),
    ("jupyter/xbotics-jupyter.service", SYSTEMD / "xbotics-jupyter.service", 0o644, "xbotics-jupyter"),
    ("jupyter/notification.jupyterlab-settings",
     ROOT_HOME / ".jupyter/lab/user-settings/@jupyterlab/apputils-extension/notification.jupyterlab-settings", 0o644,
     "xbotics-jupyter"),
    ("vscode/settings.json", ROOT_HOME / ".local/share/code-server/User/settings.json", 0o644, "code-server@root"),
]  # fmt: skip

# xbotics-config 管理的五个网页服务：首次安装全部开启，之后保持用户的开关设置
WEB_SERVICES = (
    "ttyd",
    "xbotics-jupyter",
    "code-server@root",
    "xbotics-hardware-check",
    "xbotics-calibrate-web",
)
# 运行 LeRobot 环境中的代码：LeRobot 升级后须重启才能使用新版本
LEROBOT_SERVICES = ("xbotics-jupyter", "xbotics-hardware-check", "xbotics-calibrate-web")


def say(msg: str) -> None:
    print(f"· {msg}", flush=True)


def run(*cmd: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, **kw)


def write_if_changed(dest: Path, content: bytes, mode: int) -> bool:
    if dest.exists() and dest.read_bytes() == content:
        if dest.stat().st_mode & 0o777 != mode:
            dest.chmod(mode)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".xbotics-new")
    tmp.write_bytes(content)
    tmp.chmod(mode)
    os.replace(tmp, dest)
    return True


# ---------- 端侧板的 LeRobot 环境 ----------


def req_name(req: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else req


REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?(.*)$", re.S)


def merge_extras(template_req: str, old_req: str) -> str:
    """以模板的依赖写法为准，并入学员为它添加的可选组（如 lerobot[...] 中额外写的）。"""
    t, o = REQ.match(template_req), REQ.match(old_req)
    if not (t and o and o.group(2)):
        return template_req
    extras = [e.strip() for e in (t.group(2) or "").split(",") if e.strip()]
    extras += [e.strip() for e in o.group(2).split(",") if e.strip() and e.strip() not in extras]
    return f"{t.group(1)}[{','.join(extras)}]{t.group(3)}"


def ensure_tomlkit(template: str) -> None:
    """环境中没有 tomlkit（还没装 xbotics_tools 组的老环境）时，用 uv add 给 lerobot 补上可选组，同步后重新执行安装。

    uv add 只改这一条依赖、保留文件其余内容；补上的 xbotics_tools 组带来 tomlkit，重新执行后按逐键修改的方式升级。
    """
    try:
        import tomlkit  # noqa: F401
    except ImportError:
        old = tomllib.loads((ENV_DIR / "pyproject.toml").read_text())
        template_lerobot = next(
            d for d in tomllib.loads(template)["project"]["dependencies"] if req_name(d) == "lerobot"
        )
        old_lerobot = next(
            (d for d in old["project"]["dependencies"] if req_name(d) == "lerobot"), template_lerobot
        )
        say("LeRobot 环境：补上 xbotics_tools 组…")
        env = {**os.environ, "HOME": str(ROOT_HOME)}
        run(str(UV), "add", "--no-sync", merge_extras(template_lerobot, old_lerobot), cwd=ENV_DIR, env=env)
        run(str(UV), "sync", cwd=ENV_DIR, env=env)
        os.execv(sys.executable, [sys.executable, "-m", "lerobot.xbotics.board", "install"])


def render_env_pyproject(template: str, old: str | None) -> str:
    """升级端侧板环境的 pyproject：只修改维护版管理的几项，其余保持不变。

    保持不变的有学员用 uv add 添加的包、源、索引、依赖组和注释。管理的几项：Python 版本、模板中的依赖（保留学员为它们
    添加的可选组）、override-dependencies、LeRobot 的源。LeRobot 的源指向分支或提交（用 bootstrap.sh 的 XBOTICS_REF
    安装的测试板）时不修改；指向标签时替换为模板中的标签，发布新标签后升级即会切换。上一版模板管理、本版删除的依赖
    记录在 [tool.xbotics] 中，升级时一并删除。
    """
    if old is None:
        return template
    import tomlkit

    t = tomllib.loads(template)
    doc = tomlkit.parse(old)
    template_deps = t["project"]["dependencies"]
    managed = {req_name(d) for d in template_deps}
    previously_managed = set(doc.get("tool", {}).get("xbotics", {}).get("managed-dependencies", []))

    def put(table, key: str, value) -> None:
        """值不同时才写入：相同时保留文件原有的排版（多行列表、空格）。"""
        current = table.get(key)
        if getattr(current, "unwrap", lambda: current)() != value:
            table[key] = value

    project = doc.setdefault("project", tomlkit.table())
    old_deps = [str(d) for d in project.get("dependencies", [])]
    by_name = {req_name(d): d for d in old_deps}
    deps = [merge_extras(d, by_name[req_name(d)]) if req_name(d) in by_name else d for d in template_deps]
    deps += [d for d in old_deps if req_name(d) not in managed | previously_managed]
    put(project, "dependencies", deps)
    put(project, "requires-python", t["project"]["requires-python"])

    tool = doc.setdefault("tool", tomlkit.table())
    uv = tool.setdefault("uv", tomlkit.table())
    put(uv, "package", t["tool"]["uv"]["package"])
    put(uv, "override-dependencies", t["tool"]["uv"]["override-dependencies"])
    sources = uv.setdefault("sources", tomlkit.table())
    old_src = sources.get("lerobot") or {}
    template_src = t["tool"]["uv"]["sources"]["lerobot"]
    if not ("branch" in old_src or "rev" in old_src) and dict(old_src) != template_src:
        src = tomlkit.inline_table()
        src.update(template_src)
        sources["lerobot"] = src
    put(tool.setdefault("xbotics", tomlkit.table()), "managed-dependencies", sorted(managed))
    new = tomlkit.dumps(doc)
    return old if tomllib.loads(new) == tomllib.loads(old) else new


def sync_env() -> None:
    """按模板升级端侧板环境的 pyproject，有变化时重新同步依赖。

    同步失败时把 pyproject 恢复原样再抛出：否则文件已是新版，重试时判断为「已是最新」而跳过同步，缺的依赖补不上。
    """
    template = (BOARD / "board_env/pyproject.toml").read_text()
    dest = ENV_DIR / "pyproject.toml"
    old = dest.read_text() if dest.exists() else None
    if old is not None:
        ensure_tomlkit(template)
    new = render_env_pyproject(template, old)
    if old == new:
        say("LeRobot 环境：pyproject 已是最新")
        return
    write_if_changed(dest, new.encode(), 0o644)
    say("LeRobot 环境：pyproject 已按新版更新，正在重新同步依赖…")
    try:
        run(str(UV), "sync", cwd=ENV_DIR, env={**os.environ, "HOME": str(ROOT_HOME)})
    except subprocess.CalledProcessError:
        if old is not None:
            write_if_changed(dest, old.encode(), 0o644)
        raise


# ---------- 生成的配置 ----------


def ttyd_options() -> bool:
    path = Path("/etc/default/ttyd")
    text = path.read_text() if path.exists() else "# /etc/default/ttyd\n"
    line = 'TTYD_OPTIONS="-p 8002 -O login"'
    new = (
        re.sub(r"^TTYD_OPTIONS=.*$", line, text, flags=re.M)
        if "TTYD_OPTIONS=" in text
        else text + line + "\n"
    )
    return write_if_changed(path, new.encode(), 0o644)


def jupyter_config() -> bool:
    """写入 JupyterLab 的服务配置（8003、bash 终端、只认密码）；已设置的密码哈希保留，没有时用默认密码。"""
    path = ROOT_HOME / ".jupyter/jupyter_server_config.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    hashed = old.get("PasswordIdentityProvider", {}).get("hashed_password")
    if not hashed:
        from jupyter_server.auth import passwd

        hashed = passwd(DEFAULT_PASSWORD)
    cfg = {
        "ServerApp": {
            "ip": "0.0.0.0",
            "port": 8003,
            "open_browser": False,
            "allow_root": True,
            "root_dir": "/root",
            "terminado_settings": {"shell_command": ["/bin/bash"]},
        },
        "PasswordIdentityProvider": {"hashed_password": hashed, "allow_password_change": False},
    }
    return write_if_changed(path, (json.dumps(cfg, indent=2) + "\n").encode(), 0o600)


def code_server_config() -> bool:
    path = ROOT_HOME / ".config/code-server/config.yaml"
    old = path.read_text() if path.exists() else ""
    m = re.search(r"^password:\s*(.+)$", old, flags=re.M)
    password = m.group(1).strip() if m else DEFAULT_PASSWORD
    text = (
        "bind-addr: 0.0.0.0:8004\n"
        "auth: password\n"
        f"password: {password}\n"
        "cert: false\n"
        "disable-telemetry: true\n"
        "disable-update-check: true\n"
    )
    return write_if_changed(path, text.encode(), 0o600)


def code_server() -> bool:
    """安装（或切换到）固定版本的 code-server，并补齐插件；有改动时返回 True。"""
    changed = False
    have = (
        run("code-server", "--version", check=False, capture_output=True).stdout
        if shutil.which("code-server")
        else ""
    )
    if not have.startswith(CODE_SERVER_VERSION):
        say(f"浏览器版 VS Code：安装 code-server {CODE_SERVER_VERSION}…")
        script = run("curl", "-fsSL", "https://code-server.dev/install.sh", capture_output=True).stdout
        run(
            "sh",
            "-s",
            "--",
            "--version",
            CODE_SERVER_VERSION,
            input=script,
            env={**os.environ, "HOME": str(ROOT_HOME)},
        )
        changed = True
    installed = run("code-server", "--list-extensions", capture_output=True, check=False).stdout.split()
    for ext in CODE_SERVER_EXTENSIONS:
        if ext not in installed:
            say(f"浏览器版 VS Code：安装插件 {ext}")
            run("code-server", "--install-extension", ext, env={**os.environ, "HOME": str(ROOT_HOME)})
            changed = True
    return changed


def bashrc() -> None:
    path = ROOT_HOME / ".bashrc"
    line = "source ~/xbotics/.venv/bin/activate"
    text = path.read_text() if path.exists() else ""
    if line not in text.splitlines():
        path.write_text(text + ("" if text.endswith("\n") or not text else "\n") + line + "\n")
        say("已设置登录后自动进入 LeRobot 环境")


def board_hostname() -> bool:
    """主机名统一为 xbotics-XXXX（后四位同热点名，取自 Wi-Fi 网卡地址），有改动时返回 True。

    厂家镜像的默认主机名带厂家名，会出现在终端提示符与局域网的 .local 名字里；后四位每块板不同，
    同一局域网里有几块端侧板时 .local 名字也不冲突。已经是 xbotics- 开头的不再修改。
    """
    current = socket.gethostname()
    if current.startswith("xbotics-"):
        return False
    mac = Path("/sys/class/net/wlan0/address").read_text().strip().replace(":", "")
    name = f"xbotics-{mac[-4:]}"
    run("hostnamectl", "set-hostname", name)
    hosts = Path("/etc/hosts")
    lines = [line for line in hosts.read_text().splitlines() if not line.startswith("127.0.1.1")]
    hosts.write_text("\n".join([*lines, f"127.0.1.1\t{name}"]) + "\n")
    if is_active("avahi-daemon"):
        run("systemctl", "restart", "avahi-daemon")
    say(f"主机名：{current} → {name}（局域网中可用 {name}.local 访问）")
    return True


# ---------- 主流程 ----------


def apt_install() -> None:
    missing = [
        p for p in APT_PACKAGES if run("dpkg", "-s", p, check=False, capture_output=True).returncode != 0
    ]
    if missing:
        say(f"安装系统包：{' '.join(missing)}")
        run("apt-get", "update")
        run("apt-get", "install", "-y", *missing, env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})


def lerobot_version() -> str:
    """已安装的 LeRobot 版本：从 git 安装的取提交号，否则取版本号。"""
    from importlib import metadata

    dist = metadata.distribution("lerobot")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return direct.get("vcs_info", {}).get("commit_id") or dist.version


def systemctl(*args: str) -> subprocess.CompletedProcess:
    return run("systemctl", *args, check=False, capture_output=True)


def is_active(unit: str) -> bool:
    return systemctl("is-active", unit).stdout.strip() == "active"


def running_tunnels() -> list[str]:
    """正在运行的隧道实例（每台服务器一个 xbotics-tunnel@<编号>）。"""
    out = systemctl("list-units", "--state=active", "--plain", "--no-legend", "xbotics-tunnel@*").stdout
    return [line.split()[0].removesuffix(".service") for line in out.splitlines() if line.strip()]


def remove_obsolete() -> None:
    """删除旧版留下、现已不用的文件：单台服务器的隧道服务（已改为每台服务器一个实例）。"""
    old_unit = SYSTEMD / "xbotics-tunnel.service"
    if old_unit.exists():
        systemctl("disable", "--now", "xbotics-tunnel")
        old_unit.unlink()
        for name in ("tunnel.json", "tunnel_key", "tunnel_known_hosts", "tunnel_askpass"):
            (Path("/etc/xbotics") / name).unlink(missing_ok=True)
        run("systemctl", "daemon-reload")
        say("已移除旧版的单台服务器隧道（如需要，请在网络设置页重新添加服务器）")


def plan_restarts(
    running: set[str], changed_units: set[str], lerobot_changed: bool, tunnels: list[str]
) -> list[str]:
    """正在运行的服务中需要重启的：文件有改动的，LeRobot 版本变化时运行 LeRobot 代码的服务，隧道程序改动时的各隧道实例。"""
    units = ("xbotics-maxfreq", "xbotics-provision", *WEB_SERVICES)
    plan = [
        u
        for u in units
        if u in running and (u in changed_units or (lerobot_changed and u in LEROBOT_SERVICES))
    ]
    return plan + (tunnels if "xbotics-tunnel" in changed_units else [])


def install() -> None:
    """安装或升级端侧板上的全部内容；首次安装开启五个网页服务，升级保持现有开关，只重启有改动的服务。"""
    if os.geteuid() != 0:
        raise SystemExit(
            "须以 root 运行：sudo /root/xbotics/.venv/bin/python -m lerobot.xbotics.board install"
        )
    # 按旧版手册手工安装的端侧板没有 MARKER，但已有这些服务文件：按升级处理，不改变学员关闭的服务
    legacy = any((SYSTEMD / f"{u}.service").exists() for u in ("xbotics-provision", "xbotics-hardware-check"))
    first = not MARKER.exists() and not legacy
    version = lerobot_version()
    lerobot_changed = not MARKER.exists() or MARKER.read_text().strip() != version
    say("首次安装" if first else "升级：保持现有的服务开关，仅重启有改动的服务")
    apt_install()
    sync_env()
    renamed = board_hostname()

    changed_units: set[str] = set()
    for src, dest, mode, units in FILES:
        if write_if_changed(Path(dest), (BOARD / src).read_bytes(), mode):
            say(f"已更新 {dest}")
            changed_units.update(units.split())
    if ttyd_options():
        changed_units.add("ttyd")
    if jupyter_config():
        changed_units.add("xbotics-jupyter")
    if code_server_config() | code_server():
        changed_units.add("code-server@root")
    bashrc()

    if "sysctl" in changed_units:
        run("sysctl", "--load=/etc/sysctl.d/99-xbotics-inotify.conf", capture_output=True)
    if "udev" in changed_units:
        run("udevadm", "control", "--reload")
        run("udevadm", "trigger", "--action=add", "--subsystem-match=tty", "--subsystem-match=video4linux")
    run("systemctl", "daemon-reload")
    if systemctl("get-default").stdout.strip() != "multi-user.target":
        run("systemctl", "set-default", "multi-user.target")
        say("已设置开机不进入桌面（重启后生效）")

    remove_obsolete()
    always_on = ("xbotics-maxfreq", "xbotics-provision")
    running = {u for u in (*always_on, *WEB_SERVICES) if is_active(u)}
    if renamed:
        changed_units.add("xbotics-provision")  # 页面上显示的主机名与 .local 地址要更新
    to_restart = plan_restarts(running, changed_units, lerobot_changed, running_tunnels())
    for unit in always_on:
        run("systemctl", "enable", "--now", unit, capture_output=True)
    if first:
        for unit in WEB_SERVICES:
            run("systemctl", "enable", "--now", unit, capture_output=True)
        say("已开启五个网页服务（8002–8006），之后可用 xbotics-config 开关")

    if to_restart:
        # 放在单独的单元中、延后 3 秒执行：在网页终端或 VS Code 终端中升级时，重启这些服务不会中断本进程
        run(
            "systemd-run",
            "--on-active=3",
            "--collect",
            "systemctl",
            "restart",
            *to_restart,
            capture_output=True,
        )
        say(f"3 秒后重启：{' '.join(to_restart)}（这些网页会短暂断开，刷新即可）")
    # 安排好重启之后才记下安装状态：前面任何一步失败，重试时仍按「LeRobot 版本变化」处理，该重启的服务不会漏掉
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(version + "\n")
    say("安装完成：用 xbotics-config status 查看各服务状态")
