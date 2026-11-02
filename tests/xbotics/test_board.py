"""板子安装器与连服务器的隧道（Xbotics）：不碰真系统，只测纯逻辑与本机端口。

- 板子环境的 pyproject 升级只改维护版管的几项：学员自己加的包、源、索引、依赖组、可选组都保留，指到分支的 LeRobot 源不动；
- 升级时重启哪些服务；一键安装脚本怎么改 LeRobot 的源；
- 网络设置页上填的隧道设置怎么校验、没重填的密码与私钥怎么沿用；
- 拼出来的 ssh 命令：两种登录方式、每条映射都监听 0.0.0.0；
- ssh 报错翻成白话；经隧道的端口探测分得清「通 / 服务器上没服务 / 没映射上」。
"""

import importlib.util
import socket
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import pytest

from lerobot.xbotics.board.install import BOARD, FILES, plan_restarts, render_env_pyproject, req_name

PROVISION = BOARD / "provision"


def load(name: str):
    sys.path.insert(0, str(PROVISION))
    spec = importlib.util.spec_from_file_location(name, PROVISION / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tunnel(tmp_path, monkeypatch):
    mod = load("xbotics_tunnel")
    monkeypatch.setattr(mod, "CONF_DIR", tmp_path)
    monkeypatch.setattr(mod, "CONFIG", tmp_path / "tunnel.json")
    monkeypatch.setattr(mod, "KEY_FILE", tmp_path / "tunnel_key")
    monkeypatch.setattr(mod, "KNOWN_HOSTS", tmp_path / "known_hosts")
    return mod


@pytest.fixture
def provision(tunnel):
    mod = load("xbotics_provision")
    mod.tunnel = tunnel
    return mod


TEMPLATE = (BOARD / "board_env/pyproject.toml").read_text()


def test_every_packaged_file_exists():
    for src, *_ in FILES:
        assert (BOARD / src).is_file(), src


def test_fresh_env_is_the_template():
    assert render_env_pyproject(TEMPLATE, None) == TEMPLATE


def test_old_manual_env_gets_new_extras_and_keeps_user_packages():
    old = TEMPLATE.replace(
        '["lerobot[feetech,async,dataset,viz,xbotics_tools]", "jupyterlab"]',
        '["lerobot[feetech,async,dataset,viz]", "jupyterlab", "Scikit_Learn>=1.5", "mylib"]',
    )
    old = old.replace(
        'tag = "xbotics-v0.6.1" }\n',
        'tag = "xbotics-v0.6.1" }\nmylib = { git = "https://example.com/mylib.git", branch = "dev" }\n',
    )
    new = tomllib.loads(render_env_pyproject(TEMPLATE, old))
    deps = new["project"]["dependencies"]
    assert deps[0] == "lerobot[feetech,async,dataset,viz,xbotics_tools]"
    assert "Scikit_Learn>=1.5" in deps and "mylib" in deps
    assert sum(req_name(d) == "lerobot" for d in deps) == 1
    assert new["tool"]["uv"]["sources"]["mylib"] == {"git": "https://example.com/mylib.git", "branch": "dev"}
    assert (
        new["tool"]["uv"]["override-dependencies"]
        == tomllib.loads(TEMPLATE)["tool"]["uv"]["override-dependencies"]
    )


def test_lerobot_source_pinned_to_a_branch_survives_upgrade():
    old = TEMPLATE.replace('tag = "xbotics-v0.6.1"', 'branch = "xbotics-0.6"')
    src = tomllib.loads(render_env_pyproject(TEMPLATE, old))["tool"]["uv"]["sources"]["lerobot"]
    assert src == {"git": "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", "branch": "xbotics-0.6"}


LEGACY_MANUAL_ENV = """[project]
name = "xbotics-board"
version = "0.0.0"
requires-python = "==3.12.*"
dependencies = ["lerobot[feetech,async,dataset,viz,mytools]", "jupyterlab", "scipy>=1.14"]

[dependency-groups]
dev = ["pytest"]

[tool.uv]
package = false
override-dependencies = ["torch==2.9.1"]

[[tool.uv.index]]
name = "mirror"
url = "https://mirrors.example.com/pypi/simple"

[tool.uv.sources]
lerobot = { git = "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", tag = "xbotics-v0.6.0" }
"my.pkg" = { index = "mirror" }
"""


def test_upgrade_keeps_everything_the_template_does_not_manage():
    pytest.importorskip("tomlkit")
    new = tomllib.loads(render_env_pyproject(TEMPLATE, LEGACY_MANUAL_ENV))
    t = tomllib.loads(TEMPLATE)
    deps = new["project"]["dependencies"]
    # 学员给 lerobot 加的可选组并进模板的写法，自己加的包留着
    assert deps[0] == "lerobot[feetech,async,dataset,viz,xbotics_tools,mytools]"
    assert "scipy>=1.14" in deps and "jupyterlab" in deps
    assert new["dependency-groups"] == {"dev": ["pytest"]}
    assert new["tool"]["uv"]["index"] == [
        {"name": "mirror", "url": "https://mirrors.example.com/pypi/simple"}
    ]
    assert new["tool"]["uv"]["sources"]["my.pkg"] == {"index": "mirror"}
    # 维护版管的几项按模板：override、以及标签源换成模板的新标签
    assert new["tool"]["uv"]["override-dependencies"] == t["tool"]["uv"]["override-dependencies"]
    assert new["tool"]["uv"]["sources"]["lerobot"] == t["tool"]["uv"]["sources"]["lerobot"]
    assert new["tool"]["xbotics"]["managed-dependencies"] == ["jupyterlab", "lerobot"]


def test_upgrade_is_idempotent_and_drops_deps_the_template_stopped_managing():
    pytest.importorskip("tomlkit")
    once = render_env_pyproject(TEMPLATE, LEGACY_MANUAL_ENV)
    assert render_env_pyproject(TEMPLATE, once) == once
    # 新模板不再要 jupyterlab：上一版记在 managed-dependencies 里的它被删掉，学员自己加的 scipy 留着
    newer = TEMPLATE.replace(', "jupyterlab"]', "]").replace('["jupyterlab", "lerobot"]', '["lerobot"]')
    deps = tomllib.loads(render_env_pyproject(newer, once))["project"]["dependencies"]
    assert not any(req_name(d) == "jupyterlab" for d in deps) and "scipy>=1.14" in deps


def test_restart_plan():
    running = {"xbotics-provision", "xbotics-jupyter", "ttyd", "xbotics-hardware-check"}
    # 什么都没变：一个都不重启
    assert plan_restarts(running, set(), lerobot_changed=False) == []
    # LeRobot 换了版本：在跑的、跑 LeRobot 代码的才重启；没在跑的网页标定不去拉起
    assert plan_restarts(running, set(), lerobot_changed=True) == [
        "xbotics-jupyter",
        "xbotics-hardware-check",
    ]
    # 文件改了：只重启在跑的
    assert plan_restarts(running, {"xbotics-provision", "xbotics-tunnel"}, False) == ["xbotics-provision"]


@pytest.mark.parametrize(
    "ref,source",
    [
        ("xbotics-v0.6.2", 'tag = "xbotics-v0.6.2"'),
        ("xbotics-0.6", 'branch = "xbotics-0.6"'),
        ("d3179a7c9", 'rev = "d3179a7c9"'),
    ],
)
def test_bootstrap_points_lerobot_at_the_ref(tmp_path, ref, source):
    (tmp_path / "pyproject.toml").write_text(TEMPLATE)
    script = BOARD / "bootstrap.sh"
    subprocess.run(
        ["bash", "-c", f'eval "$(sed -n "/^set_ref()/,/^}}/p" {script})"; set_ref {ref}'],
        cwd=tmp_path,
        check=True,
    )
    line = next(
        x for x in (tmp_path / "pyproject.toml").read_text().splitlines() if x.startswith("lerobot = ")
    )
    assert (
        line == f'lerobot = {{ git = "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", {source} }}'
    )


def test_ssh_command_password_and_key(tunnel):
    cfg = {"host": "srv", "port": 2222, "user": "u", "auth": "password", "forwards": tunnel.DEFAULT_FORWARDS}
    cmd = tunnel.ssh_command(cfg)
    assert cmd[-1] == "u@srv" and cmd[cmd.index("-p") + 1] == "2222"
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-L"] == [
        "0.0.0.0:8080:127.0.0.1:8080",
        "0.0.0.0:6006:127.0.0.1:6006",
    ]
    assert "PubkeyAuthentication=no" in cmd and "-i" not in cmd
    key = tunnel.ssh_command({**cfg, "auth": "key"})
    assert key[key.index("-i") + 1] == str(tunnel.KEY_FILE) and "PreferredAuthentications=publickey" in key


def test_askpass_answers_password_or_passphrase(tunnel):
    tunnel.CONFIG.write_text('{"password": "pw", "passphrase": "pp"}')
    assert tunnel.askpass("u@srv's password: ") == "pw"
    assert tunnel.askpass("Enter passphrase for key '/etc/xbotics/tunnel_key': ") == "pp"


FWD = [{"name": "TB", "local": "6006", "remote": "6006"}]


def test_parse_tunnel_validates_each_field(provision):
    ok = {"host": "srv", "user": "u", "password": "pw", "forwards": FWD}
    assert provision.parse_tunnel(ok, {})["forwards"][0] == {
        "name": "TB",
        "local": 6006,
        "remote_host": "127.0.0.1",
        "remote": 6006,
        "web": True,
    }
    for bad, word in [
        ({**ok, "host": " "}, "地址"),
        ({**ok, "user": ""}, "账号"),
        ({**ok, "port": "70000"}, "SSH 端口"),
        ({**ok, "password": ""}, "密码"),
        ({**ok, "forwards": []}, "至少"),
        ({**ok, "forwards": [{"local": "8003", "remote": "1"}]}, "已被占用"),
        ({**ok, "forwards": [{"local": "x", "remote": "1"}]}, "数字"),
        ({**ok, "auth": "key"}, "私钥没填"),
        ({**ok, "auth": "key", "key": "ssh-ed25519 AAAA… 公钥"}, "不像私钥"),
    ]:
        with pytest.raises(ValueError, match=word):
            provision.parse_tunnel(bad, {})


def test_parse_tunnel_keeps_saved_password_only_for_same_login(provision):
    old = {"host": "srv", "user": "u", "password": "saved"}
    body = {"host": "srv", "user": "u", "password": "", "forwards": FWD}
    assert provision.parse_tunnel(body, old)["password"] == "saved"
    with pytest.raises(ValueError, match="密码"):
        provision.parse_tunnel({**body, "host": "other"}, old)


def test_parse_tunnel_remote_host_and_saved_key(provision, tunnel):
    tunnel.KEY_FILE.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    body = {
        "host": "srv",
        "user": "u",
        "auth": "key",
        "forwards": [{"local": "9000", "remote_host": "10.0.0.5", "remote": "80"}],
    }
    cfg = provision.parse_tunnel(body, {"host": "srv", "user": "u", "auth": "key"})
    assert "_key" not in cfg and cfg["forwards"][0]["remote_host"] == "10.0.0.5"


def test_specific_key_errors_win_over_the_permission_denied_that_follows(provision, monkeypatch):
    out = (
        "隧道：0.0.0.0:6006 经 u@srv\n"
        'Load key "/etc/xbotics/tunnel_key": incorrect passphrase supplied to decrypt private key\n'
        "u@srv: Permission denied (publickey).\n"
    )
    monkeypatch.setattr(provision.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": out})())
    assert "口令不对" in provision.tunnel_error()


def test_forward_failures_are_not_tunnel_errors(provision, monkeypatch):
    out = "隧道：0.0.0.0:8080 经 u@srv\nchannel 2: open failed: connect failed: Connection refused\n"
    monkeypatch.setattr(provision.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": out})())
    assert provision.tunnel_error() == ""


@pytest.mark.parametrize(
    "line,word",
    [
        ("u@srv: Permission denied (publickey,password).", "登录被拒"),
        ("ssh: Could not resolve hostname nope: Name or service not known", "找不到"),
        ("ssh: connect to host 1.2.3.4 port 22: Connection timed out", "连不上"),
        ("ssh: connect to host 1.2.3.4 port 2200: Connection refused", "拒绝连接"),
        ("bind [0.0.0.0]:6006: Address already in use", "占用"),
        ('Load key "/etc/xbotics/tunnel_key": invalid format', "格式"),
    ],
)
def test_tunnel_errors_are_translated(provision, monkeypatch, line, word):
    out = f"隧道：0.0.0.0:6006 经 u@srv\n{line}\n"
    monkeypatch.setattr(provision.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": out})())
    assert word in provision.tunnel_error()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(handler) -> tuple[int, socket.socket]:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen()

    def loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            handler(conn)

    threading.Thread(target=loop, daemon=True).start()
    return srv.getsockname()[1], srv


def test_probe_tells_service_no_service_and_no_listener(provision):
    def answer(conn):
        conn.recv(64)
        conn.sendall(b"HTTP/1.0 200 OK\r\n\r\n")
        conn.close()

    port, srv = serve(answer)
    assert provision.probe_forward(port) == "ok"
    srv.close()

    def grpc_like(conn):  # gRPC 这类：不出声等客户端先说话，收到不认得的数据就断开
        conn.settimeout(3)
        try:
            if conn.recv(1):
                conn.close()
        except OSError:
            pass

    port, srv = serve(grpc_like)
    assert provision.probe_forward(port) == "ok"
    srv.close()
    port, srv = serve(lambda conn: conn.close())  # ssh 的本地转发在服务器那头连不上时就是这样：接了就断
    assert provision.probe_forward(port) == "closed"
    srv.close()
    assert provision.probe_forward(free_port()) == "down"


def test_bootstrap_and_config_scripts_are_executable():
    for rel in ("bootstrap.sh", "config/xbotics-config"):
        assert Path(BOARD / rel).stat().st_mode & 0o111, rel


def test_every_board_file_is_tracked_by_git():
    """板子靠从 GitHub 装的包拿这些文件：工作区里有、git 里没有（比如被 .gitignore 的 env/ 挡掉）的，板子上就缺。"""
    import subprocess

    repo = BOARD.parents[3]
    if not (repo / ".git").exists():
        pytest.skip("不在 git 检出里")
    tracked = set(
        subprocess.run(
            ["git", "ls-files", str(BOARD)], cwd=repo, capture_output=True, text=True
        ).stdout.split()
    )
    files = {
        str(p.relative_to(repo)) for p in BOARD.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    assert files - tracked == set()


def test_tunnel_stops_retrying_after_an_auth_failure():
    """认证失败的退出码要与服务里的 RestartPreventExitStatus 对上，否则 systemd 照样每 5 秒拿错密码去试。"""
    unit = (PROVISION / "xbotics-tunnel.service").read_text()
    tunnel_mod = load("xbotics_tunnel")
    assert f"RestartPreventExitStatus={tunnel_mod.AUTH_FAILED}" in unit
