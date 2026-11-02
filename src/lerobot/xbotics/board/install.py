"""把星禾套件板子装成（或升级成）当前维护版 LeRobot 对应的样子：板子上要维护的东西全在这个包里，升级 LeRobot 就升级全部。

一次装下面这些，重复运行只改有变化的部分（以 root 在板子的 LeRobot 环境里跑）：

- 系统：开机不进桌面、CPU 与内存锁最高频、相机不自己降帧、机械臂与相机的固定设备名（udev）
- 网络设置页（8001，热点配网 + 连服务器的 SSH 隧道）
- 网页终端 ttyd（8002）、JupyterLab（8003）、浏览器版 VS Code（8004，版本钉在 ``CODE_SERVER_VERSION``）
- 网页硬件检测（8005）、网页标定（8006）与开关命令 ``xbotics-config``
- 板子的 LeRobot 环境 ``/root/xbotics`` 的 pyproject：按 ``env/pyproject.toml`` 重写，自己用 ``uv add`` 加的包与源保留

第一次装把 8002–8006 五个网页服务都打开；之后再装（升级）保持 ``xbotics-config`` 里的开关不变，
在跑的服务有改动才重启。重启延后 3 秒、放进单独的 systemd 单元：从网页终端或 VS Code 终端里升级时，
重启它们不会把升级本身掐断。密码（JupyterLab、VS Code）已经设过的不覆盖。

    python -m lerobot.xbotics.board install
"""

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

BOARD = Path(__file__).resolve().parent
ROOT_HOME = Path("/root")
ENV_DIR = ROOT_HOME / "xbotics"
UV = ROOT_HOME / ".local/bin/uv"
MARKER = Path("/etc/xbotics/installed")  # 第一次装完写下；有它就是升级
DEFAULT_PASSWORD = "123456"  # 与板子 root 密码相同：网页终端、JupyterLab、VS Code 只记一个
CODE_SERVER_VERSION = "4.141.0"
CODE_SERVER_EXTENSIONS = ("ms-python.python", "detachhead.basedpyright", "tamasfe.even-better-toml")
APT_PACKAGES = ("ttyd", "iw", "whiptail")
SYSTEMD = Path("/etc/systemd/system")

# (包里的文件, 装到哪, 权限, 用到它的 systemd 单元)
FILES = [
    ("udev/99-xbotics-rk3566.rules", "/etc/udev/rules.d/99-xbotics-rk3566.rules", 0o644, "udev"),
    ("udev/99-xbotics-camera-fps.rules", "/etc/udev/rules.d/99-xbotics-camera-fps.rules", 0o644, "udev"),
    ("system/xbotics-maxfreq.service", SYSTEMD / "xbotics-maxfreq.service", 0o644, "xbotics-maxfreq"),
    ("provision/xbotics_provision.py", "/opt/xbotics/provision/xbotics_provision.py", 0o644, "xbotics-provision"),
    ("provision/xbotics_tunnel.py", "/opt/xbotics/provision/xbotics_tunnel.py", 0o644, "xbotics-provision xbotics-tunnel"),
    ("provision/index.html", "/opt/xbotics/provision/index.html", 0o644, "xbotics-provision"),
    ("provision/xbotics-captive.conf", "/etc/NetworkManager/dnsmasq-shared.d/xbotics-captive.conf", 0o644, "xbotics-provision"),
    ("provision/xbotics-provision.service", SYSTEMD / "xbotics-provision.service", 0o644, "xbotics-provision"),
    ("provision/xbotics-tunnel.service", SYSTEMD / "xbotics-tunnel.service", 0o644, "xbotics-tunnel"),
    ("config/xbotics-config", "/usr/local/bin/xbotics-config", 0o755, ""),
    ("config/xbotics-hardware-check.service", SYSTEMD / "xbotics-hardware-check.service", 0o644, "xbotics-hardware-check"),
    ("config/xbotics-calibrate-web.service", SYSTEMD / "xbotics-calibrate-web.service", 0o644, "xbotics-calibrate-web"),
    ("jupyter/xbotics-jupyter.service", SYSTEMD / "xbotics-jupyter.service", 0o644, "xbotics-jupyter"),
    ("jupyter/notification.jupyterlab-settings",
     ROOT_HOME / ".jupyter/lab/user-settings/@jupyterlab/apputils-extension/notification.jupyterlab-settings", 0o644,
     "xbotics-jupyter"),
    ("vscode/settings.json", ROOT_HOME / ".local/share/code-server/User/settings.json", 0o644, "code-server@root"),
]  # fmt: skip

# xbotics-config 管的五个网页服务：第一次装全开，之后保持用户的开关
WEB_SERVICES = (
    "ttyd",
    "xbotics-jupyter",
    "code-server@root",
    "xbotics-hardware-check",
    "xbotics-calibrate-web",
)
# 跑的是 LeRobot 环境里的代码：LeRobot 升级了它们就要重启才用上新版
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


# ---------- 板子的 LeRobot 环境 ----------


def req_name(req: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else req


def toml_inline(table: dict) -> str:
    return "{ " + ", ".join(f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in table.items()) + " }"


def render_env_pyproject(template: str, old: str | None) -> str:
    """按模板重写板子环境的 pyproject；旧文件里模板没有的依赖与 [tool.uv.sources] 条目原样保留，LeRobot 的源也沿用旧的
    （bootstrap.sh 用 XBOTICS_REF 把它指到某个分支时，升级不把它改回标签）。"""
    if old is None:
        return template
    t, o = tomllib.loads(template), tomllib.loads(old)
    deps = t["project"]["dependencies"]
    names = {req_name(d) for d in deps}
    extra = [d for d in o.get("project", {}).get("dependencies", []) if req_name(d) not in names]
    text = re.sub(
        r"^dependencies = \[.*\]$",
        lambda _: "dependencies = " + json.dumps(deps + extra, ensure_ascii=False),
        template,
        count=1,
        flags=re.M,
    )
    old_sources = o.get("tool", {}).get("uv", {}).get("sources", {})
    if "lerobot" in old_sources:
        text = re.sub(
            r"^lerobot = \{.*\}$",
            lambda _: "lerobot = " + toml_inline(old_sources["lerobot"]),
            text,
            count=1,
            flags=re.M,
        )
    for name, src in old_sources.items():
        if name != "lerobot":
            text += f"{name} = {toml_inline(src)}\n"
    return text


def sync_env() -> None:
    dest = ENV_DIR / "pyproject.toml"
    old = dest.read_text() if dest.exists() else None
    new = render_env_pyproject((BOARD / "env/pyproject.toml").read_text(), old)
    if old == new:
        say("LeRobot 环境：pyproject 已是最新")
        return
    write_if_changed(dest, new.encode(), 0o644)
    say("LeRobot 环境：pyproject 已按新版改写，重新同步依赖…")
    run(str(UV), "sync", cwd=ENV_DIR, env={**os.environ, "HOME": str(ROOT_HOME)})


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
    """装（或换成）钉住的 code-server 版本，补齐插件；有改动返回 True。"""
    changed = False
    have = (
        run("code-server", "--version", check=False, capture_output=True).stdout
        if shutil.which("code-server")
        else ""
    )
    if not have.startswith(CODE_SERVER_VERSION):
        say(f"浏览器版 VS Code：装 code-server {CODE_SERVER_VERSION}…")
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
            say(f"浏览器版 VS Code：装插件 {ext}")
            run("code-server", "--install-extension", ext, env={**os.environ, "HOME": str(ROOT_HOME)})
            changed = True
    return changed


def bashrc() -> None:
    path = ROOT_HOME / ".bashrc"
    line = "source ~/xbotics/.venv/bin/activate"
    text = path.read_text() if path.exists() else ""
    if line not in text.splitlines():
        path.write_text(text + ("" if text.endswith("\n") or not text else "\n") + line + "\n")
        say("登录后自动进 LeRobot 环境")


# ---------- 主流程 ----------


def apt_install() -> None:
    missing = [
        p for p in APT_PACKAGES if run("dpkg", "-s", p, check=False, capture_output=True).returncode != 0
    ]
    if missing:
        say(f"装系统包：{' '.join(missing)}")
        run("apt-get", "update")
        run("apt-get", "install", "-y", *missing, env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})


def systemctl(*args: str) -> subprocess.CompletedProcess:
    return run("systemctl", *args, check=False, capture_output=True)


def is_active(unit: str) -> bool:
    return systemctl("is-active", unit).stdout.strip() == "active"


def install() -> None:
    if os.geteuid() != 0:
        raise SystemExit(
            "要用 root 运行：sudo /root/xbotics/.venv/bin/python -m lerobot.xbotics.board install"
        )
    first = not MARKER.exists()
    say("第一次安装" if first else "升级：保持现有的服务开关，有改动的服务才重启")
    apt_install()
    sync_env()

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

    if "udev" in changed_units:
        run("udevadm", "control", "--reload")
        run("udevadm", "trigger", "--action=add", "--subsystem-match=tty", "--subsystem-match=video4linux")
    run("systemctl", "daemon-reload")
    if systemctl("get-default").stdout.strip() != "multi-user.target":
        run("systemctl", "set-default", "multi-user.target")
        say("开机不进桌面（重启后生效）")

    for unit in ("xbotics-maxfreq", "xbotics-provision"):  # 这两个一直开着
        run("systemctl", "enable", "--now", unit, capture_output=True)
    to_restart = [u for u in ("xbotics-maxfreq", "xbotics-provision") if u in changed_units and not first]
    if first:
        for unit in WEB_SERVICES:
            run("systemctl", "enable", "--now", unit, capture_output=True)
        say("已打开五个网页服务（8002–8006），之后用 xbotics-config 开关")
    else:
        to_restart += [
            u for u in WEB_SERVICES if is_active(u) and (u in changed_units or u in LEROBOT_SERVICES)
        ]
    if "xbotics-tunnel" in changed_units and is_active("xbotics-tunnel"):
        to_restart.append("xbotics-tunnel")

    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text("installed by lerobot.xbotics.board\n")
    if to_restart:
        # 放进单独的单元、延后 3 秒：从网页终端 / VS Code 终端里升级时，重启它们不会掐断本进程
        run(
            "systemd-run",
            "--on-active=3",
            "--collect",
            "systemctl",
            "restart",
            *to_restart,
            capture_output=True,
        )
        say(f"3 秒后重启：{' '.join(to_restart)}（从这些网页里看的话会断开一下，刷新即可）")
    say("装好了：xbotics-config status 看各个服务")
