"""星禾套件板子连服务器的 SSH 隧道：把服务器上的端口（推理服务、TensorBoard……）映射到板子的 0.0.0.0。

隧道就是 OpenSSH 客户端的本地转发 ``ssh -N -L 0.0.0.0:<板子端口>:<服务器上的地址>:<端口>``，由 systemd 服务
``xbotics-tunnel`` 拉起，断了 5 秒后自己重连（``ServerAliveInterval`` 让断网在 30 秒内被发现）。
监听 0.0.0.0：板子上的程序连 ``127.0.0.1:<端口>``，同一网络里的电脑连 ``<板子IP>:<端口>``，走的是同一条隧道。

登录两种：账号密码，或私钥（可带口令）。密码与口令经 OpenSSH 自带的 ``SSH_ASKPASS`` 交给 ssh
（``SSH_ASKPASS_REQUIRE=force``，OpenSSH 8.4 起），每次重连都能自己登录，不用 sshpass。
设置由网络设置页写进 ``/etc/xbotics/tunnel.json``（只有 root 能读），私钥存成 ``/etc/xbotics/tunnel_key``。

    python3 xbotics_tunnel.py            # 按设置起隧道（systemd 用）
    python3 xbotics_tunnel.py --askpass  # ssh 要密码 / 口令时调用
"""

import json
import os
import sys
from pathlib import Path

CONF_DIR = Path("/etc/xbotics")
CONFIG = CONF_DIR / "tunnel.json"
KEY_FILE = CONF_DIR / "tunnel_key"
KNOWN_HOSTS = CONF_DIR / "tunnel_known_hosts"
UNIT = "xbotics-tunnel"

# 默认映射两路：服务器上的推理服务（policy_server 默认 8080）与 TensorBoard（默认 6006）。
# web：是不是网页，网络设置页只给网页那几路出「打开」链接（推理服务是 gRPC，浏览器打不开）
DEFAULT_FORWARDS = [
    {"name": "推理服务", "local": 8080, "remote_host": "127.0.0.1", "remote": 8080, "web": False},
    {"name": "TensorBoard", "local": 6006, "remote_host": "127.0.0.1", "remote": 6006, "web": True},
]


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except FileNotFoundError:
        return {}


def ssh_command(cfg: dict) -> list[str]:
    cmd = [
        "ssh", "-N",
        "-p", str(cfg.get("port") or 22),
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o", "NumberOfPasswordPrompts=1",
    ]  # fmt: skip
    if cfg.get("auth") == "key":
        cmd += ["-i", str(KEY_FILE), "-o", "IdentitiesOnly=yes", "-o", "PreferredAuthentications=publickey"]
    else:
        cmd += [
            "-o",
            "PubkeyAuthentication=no",
            "-o",
            "PreferredAuthentications=password,keyboard-interactive",
        ]
    for f in cfg.get("forwards", []):
        cmd += ["-L", f"0.0.0.0:{int(f['local'])}:{f.get('remote_host') or '127.0.0.1'}:{int(f['remote'])}"]
    cmd.append(f"{cfg['user']}@{cfg['host']}")
    return cmd


def askpass(prompt: str) -> str:
    cfg = load_config()
    if "passphrase" in prompt.lower():
        return cfg.get("passphrase", "")
    return cfg.get("password", "")


def main() -> None:
    if sys.argv[1:2] == ["--askpass"]:
        print(askpass(" ".join(sys.argv[2:])))
        return
    cfg = load_config()
    if not (cfg.get("host") and cfg.get("user") and cfg.get("forwards")):
        sys.exit("还没设置要连的服务器：在网络设置页「连服务器」一栏里填")
    # SSH_ASKPASS 只认一个可执行文件、不能带参数：写一个两行的小脚本转回本文件
    helper = CONF_DIR / "tunnel_askpass"
    helper.write_text(f'#!/bin/sh\nexec {sys.executable} {Path(__file__).resolve()} --askpass "$@"\n')
    helper.chmod(0o700)
    env = {**os.environ, "SSH_ASKPASS": str(helper), "SSH_ASKPASS_REQUIRE": "force"}
    cmd = ssh_command(cfg)
    print(
        "隧道：" + " ".join(a for a in cmd if a.startswith("0.0.0.0:")) + f" 经 {cfg['user']}@{cfg['host']}",
        flush=True,
    )
    os.execvpe("ssh", cmd, env)


if __name__ == "__main__":
    main()
