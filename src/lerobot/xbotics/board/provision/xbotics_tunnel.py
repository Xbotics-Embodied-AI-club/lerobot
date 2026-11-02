"""星禾套件端侧板连接服务器的 SSH 隧道。

把服务器上的端口（推理服务、TensorBoard 等）映射到端侧板的 0.0.0.0。隧道即 OpenSSH 客户端的本地转发
``ssh -N -L 0.0.0.0:<端侧板端口>:<服务器上的地址>:<端口>``，由 systemd 服务 ``xbotics-tunnel`` 启动，
中断 5 秒后自动重连（``ServerAliveInterval`` 使断网在 30 秒内被发现）。监听 0.0.0.0：端侧板上的程序访问
``127.0.0.1:<端口>``，同一网络中的电脑访问 ``<端侧板IP>:<端口>``，两者经过同一条隧道。

账号、密码或私钥不正确时不再重试（退出码 3，服务中设 RestartPreventExitStatus=3）：反复使用错误的密码登录，
服务器上的 fail2ban 等机制会锁定该账号。修改后在网络设置页再次点击「保存并连接」。

登录方式两种：账号密码，或私钥（可带口令）。密码与口令经 OpenSSH 自带的 ``SSH_ASKPASS`` 交给 ssh
（``SSH_ASKPASS_REQUIRE=force``，OpenSSH 8.4 起），每次重连都能自动登录，无需 sshpass。
设置由网络设置页写入 ``/etc/xbotics/tunnel.json``（仅 root 可读），私钥保存为 ``/etc/xbotics/tunnel_key``。

    python3 xbotics_tunnel.py            # 按设置启动隧道（由 systemd 调用）
    python3 xbotics_tunnel.py --askpass  # ssh 需要密码或口令时调用
"""

import json
import os
import subprocess
import sys
from pathlib import Path

CONF_DIR = Path("/etc/xbotics")
CONFIG = CONF_DIR / "tunnel.json"
KEY_FILE = CONF_DIR / "tunnel_key"
KNOWN_HOSTS = CONF_DIR / "tunnel_known_hosts"
UNIT = "xbotics-tunnel"
AUTH_FAILED = 3  # 与 xbotics-tunnel.service 的 RestartPreventExitStatus 对应
AUTH_ERRORS = (
    "Permission denied",
    "incorrect passphrase",
    "invalid format",
    "Too many authentication failures",
)

# 默认映射两路：服务器上的推理服务（policy_server 默认 8080）与 TensorBoard（默认 6006）。
# web：是否为网页；网络设置页只为网页类映射提供「打开」链接（推理服务是 gRPC，浏览器无法打开）
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
    """按网络设置页校验过的配置生成 ssh 命令；配置缺少键时直接抛出 KeyError，不以默认值代替。"""
    cmd = [
        "ssh", "-N",
        "-p", str(cfg["port"]),
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
    for f in cfg["forwards"]:
        cmd += ["-L", f"0.0.0.0:{int(f['local'])}:{f['remote_host']}:{int(f['remote'])}"]
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
        sys.exit("尚未设置要连接的服务器：请在网络设置页的「连接服务器」一栏填写")
    # SSH_ASKPASS 只接受一个可执行文件、不能带参数：写一个两行的脚本转回本文件
    helper = CONF_DIR / "tunnel_askpass"
    helper.write_text(f'#!/bin/sh\nexec {sys.executable} {Path(__file__).resolve()} --askpass "$@"\n')
    helper.chmod(0o700)
    env = {**os.environ, "SSH_ASKPASS": str(helper), "SSH_ASKPASS_REQUIRE": "force"}
    cmd = ssh_command(cfg)
    print(
        "隧道：" + " ".join(a for a in cmd if a.startswith("0.0.0.0:")) + f" 经 {cfg['user']}@{cfg['host']}",
        flush=True,
    )
    # 不用 exec：需要根据 ssh 的报错决定退出码。报错原样转发到标准错误，写入 journal，供网络设置页转成中文说明
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    auth_failed = False
    for line in proc.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
        auth_failed |= any(e in line for e in AUTH_ERRORS)
    code = proc.wait()
    sys.exit(AUTH_FAILED if auth_failed else code or 1)


if __name__ == "__main__":
    main()
