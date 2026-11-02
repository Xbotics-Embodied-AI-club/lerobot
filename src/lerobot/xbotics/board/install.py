"""把星禾套件板子装成（或升级成）当前维护版 LeRobot 对应的样子：板子上要维护的东西全在这个包里，升级 LeRobot 就升级全部。

一次装下面这些，重复运行只改有变化的部分（以 root 在板子的 LeRobot 环境里跑）：

- 系统：开机不进桌面、CPU 与内存锁最高频、相机不自己降帧、机械臂与相机的固定设备名（udev）
- 网络设置页（8001，热点配网 + 连服务器的 SSH 隧道）
- 网页终端 ttyd（8002）、JupyterLab（8003）、浏览器版 VS Code（8004，版本钉在 ``CODE_SERVER_VERSION``）
- 网页硬件检测（8005）、网页标定（8006）与开关命令 ``xbotics-config``
- 板子的 LeRobot 环境 ``/root/xbotics`` 的 pyproject：只改 ``board_env/pyproject.toml`` 管的几项，其余原样保留

第一次装把 8002–8006 五个网页服务都打开；之后再装（升级）保持 ``xbotics-config`` 里的开关不变。
在跑的服务有改动就重启；跑 LeRobot 代码的那几个（JupyterLab、硬件检测、网页标定）在 LeRobot 换了版本时重启。重启延后 3 秒、放进单独的 systemd 单元：从网页终端或 VS Code 终端里升级时，
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
MARKER = Path("/etc/xbotics/installed")  # 第一次装完写下（内容是当时装的 LeRobot 版本）；有它就是升级
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


REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?(.*)$", re.S)


def merge_extras(template_req: str, old_req: str) -> str:
    """模板的依赖写法为准，学员自己给它加的可选组（如 lerobot[...] 里多写的）并进来。"""
    t, o = REQ.match(template_req), REQ.match(old_req)
    if not (t and o and o.group(2)):
        return template_req
    extras = [e.strip() for e in (t.group(2) or "").split(",") if e.strip()]
    extras += [e.strip() for e in o.group(2).split(",") if e.strip() and e.strip() not in extras]
    return f"{t.group(1)}[{','.join(extras)}]{t.group(3)}"


def render_env_pyproject_plain(template: str, old: str) -> str:
    """没有 tomlkit 时（还没装 xbotics_tools 组的老环境，只在第一次升级时走这里）：按模板整份重写，只带上旧的依赖与源。"""
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
    for name, src in o.get("tool", {}).get("uv", {}).get("sources", {}).items():
        if name != "lerobot":
            text += f"{json.dumps(name)} = {toml_inline(src)}\n"
    return text


def render_env_pyproject(template: str, old: str | None) -> str:
    """板子环境的 pyproject 升级：只改维护版管的几项，其余（学员 uv add 的包、源、索引、依赖组、注释）原样保留。

    管的几项：Python 版本、模板里的依赖（学员给它们加的可选组保留）、override-dependencies、LeRobot 的源。
    LeRobot 的源指到分支或提交（bootstrap.sh 用 XBOTICS_REF 装的测试板）时不动；是标签就换成模板的，
    以后发新标签，升级就会换过去。上一版模板管、这一版删掉的依赖记在 [tool.xbotics]，升级时一并删掉。
    """
    if old is None:
        return template
    try:
        import tomlkit
    except ImportError:
        return render_env_pyproject_plain(template, old)
    t = tomllib.loads(template)
    doc = tomlkit.parse(old)
    template_deps = t["project"]["dependencies"]
    managed = {req_name(d) for d in template_deps}
    previously_managed = set(doc.get("tool", {}).get("xbotics", {}).get("managed-dependencies", []))

    def put(table, key: str, value) -> None:
        """值不同才写：相同时保留文件里原来的排版（多行列表、空格），学员看着不乱。"""
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
    dest = ENV_DIR / "pyproject.toml"
    old = dest.read_text() if dest.exists() else None
    new = render_env_pyproject((BOARD / "board_env/pyproject.toml").read_text(), old)
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


def lerobot_version() -> str:
    """装着的 LeRobot：从 git 装的取提交号，否则取版本号。"""
    from importlib import metadata

    dist = metadata.distribution("lerobot")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return direct.get("vcs_info", {}).get("commit_id") or dist.version


def systemctl(*args: str) -> subprocess.CompletedProcess:
    return run("systemctl", *args, check=False, capture_output=True)


def is_active(unit: str) -> bool:
    return systemctl("is-active", unit).stdout.strip() == "active"


def plan_restarts(running: set[str], changed_units: set[str], lerobot_changed: bool) -> list[str]:
    """在跑的服务里，哪些要重启：文件改了的；跑 LeRobot 代码的那几个在 LeRobot 换了版本时。"""
    units = ("xbotics-maxfreq", "xbotics-provision", *WEB_SERVICES, "xbotics-tunnel")
    return [
        u
        for u in units
        if u in running and (u in changed_units or (lerobot_changed and u in LEROBOT_SERVICES))
    ]


def install() -> None:
    if os.geteuid() != 0:
        raise SystemExit(
            "要用 root 运行：sudo /root/xbotics/.venv/bin/python -m lerobot.xbotics.board install"
        )
    # 照旧手册手工装过的板子没有 MARKER，但已经有这些服务文件：按升级处理，不去动学员关掉的服务
    legacy = any((SYSTEMD / f"{u}.service").exists() for u in ("xbotics-provision", "xbotics-hardware-check"))
    first = not MARKER.exists() and not legacy
    version = lerobot_version()
    lerobot_changed = not MARKER.exists() or MARKER.read_text().strip() != version
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

    always_on = ("xbotics-maxfreq", "xbotics-provision")
    running = {u for u in (*always_on, *WEB_SERVICES, "xbotics-tunnel") if is_active(u)}
    to_restart = plan_restarts(running, changed_units, lerobot_changed)
    for unit in always_on:
        run("systemctl", "enable", "--now", unit, capture_output=True)
    if first:
        for unit in WEB_SERVICES:
            run("systemctl", "enable", "--now", unit, capture_output=True)
        say("已打开五个网页服务（8002–8006），之后用 xbotics-config 开关")

    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(version + "\n")
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
