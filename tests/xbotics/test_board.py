"""板子安装器与连服务器的隧道（Xbotics）：不碰真系统，只测纯逻辑与本机端口。

- 板子环境的 pyproject 按模板重写时，学员自己加的包与源、指到分支的 LeRobot 源都保留；
- 网络设置页上填的隧道设置怎么校验、没重填的密码与私钥怎么沿用；
- 拼出来的 ssh 命令：两种登录方式、每条映射都监听 0.0.0.0；
- ssh 报错翻成白话；经隧道的端口探测分得清「通 / 服务器上没服务 / 没映射上」。
"""

import importlib.util
import socket
import sys
import threading
import tomllib
from pathlib import Path

import pytest

from lerobot.xbotics.board.install import BOARD, FILES, render_env_pyproject, req_name

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


TEMPLATE = (BOARD / "env/pyproject.toml").read_text()


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
    old += 'mylib = { git = "https://example.com/mylib.git", branch = "dev" }\n'
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


@pytest.mark.parametrize(
    "line,word",
    [
        ("u@srv: Permission denied (publickey,password).", "登录被拒"),
        ("ssh: Could not resolve hostname nope: Name or service not known", "找不到"),
        ("ssh: connect to host 1.2.3.4 port 22: Connection timed out", "连不上"),
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
    port, srv = serve(lambda conn: conn.close())  # ssh 的本地转发在服务器那头连不上时就是这样：接了就断
    assert provision.probe_forward(port) == "closed"
    srv.close()
    assert provision.probe_forward(free_port()) == "down"


def test_bootstrap_and_config_scripts_are_executable():
    for rel in ("bootstrap.sh", "config/xbotics-config"):
        assert Path(BOARD / rel).stat().st_mode & 0o111, rel
