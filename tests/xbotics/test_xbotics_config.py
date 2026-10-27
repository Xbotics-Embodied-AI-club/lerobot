"""xbotics-config（板子网页服务开关）：用假的 systemctl / whiptail / hostname 跑真脚本。

假 systemctl 把每个服务的状态存成文件：installed/<unit>、enabled/<unit>、active/<unit>；
fail/<unit> 存在时 enable 失败；每次调用记进 calls.log。
"""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "xbotics" / "config" / "xbotics-config"
UNITS = {
    "terminal": "ttyd",
    "jupyter": "xbotics-jupyter",
    "vscode": "code-server@root",
    "hardware-check": "xbotics-hardware-check",
    "calibrate": "xbotics-calibrate-web",
}

FAKE_SYSTEMCTL = r"""#!/usr/bin/env bash
S="$FAKE_STATE"
echo "$*" >> "$S/calls.log"
cmd=$1; shift
[[ "${1:-}" == --now ]] && { now=1; shift; } || now=0
u=${1:-}
case "$cmd" in
  cat) [[ -e "$S/installed/$u" ]] ;;
  is-enabled) if [[ -e "$S/enabled/$u" ]]; then echo enabled; else echo disabled; exit 1; fi ;;
  is-active) if [[ -e "$S/active/$u" ]]; then echo active; else echo inactive; exit 1; fi ;;
  enable)
    if [[ -e "$S/fail/$u" ]]; then echo "Failed to enable unit: Unit $u is masked." >&2; exit 1; fi
    touch "$S/enabled/$u"; [[ $now == 1 ]] && touch "$S/active/$u"; exit 0 ;;
  disable) rm -f "$S/enabled/$u"; [[ $now == 1 ]] && rm -f "$S/active/$u"; exit 0 ;;
  stop) rm -f "$S/active/$u" ;;
esac
"""

FAKE_WHIPTAIL = r"""#!/usr/bin/env bash
for a in "$@"; do
  case "$a" in
    --checklist) echo "$WHIPTAIL_CHOICE" >&2; exit "${WHIPTAIL_RC:-0}" ;;
    --msgbox) printf '%s\n' "$@" >> "$FAKE_STATE/msgbox.log"; exit 0 ;;
    --yesno) exit "${WHIPTAIL_YESNO:-0}" ;;
  esac
done
"""


@pytest.fixture
def board(tmp_path):
    state = tmp_path / "state"
    for sub in ("installed", "enabled", "active", "fail"):
        (state / sub).mkdir(parents=True)
    (state / "calls.log").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in [
        ("systemctl", FAKE_SYSTEMCTL),
        ("whiptail", FAKE_WHIPTAIL),
        ("hostname", "#!/usr/bin/env bash\necho 10.0.0.2\n"),
    ]:
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)

    def set_units(installed=(), enabled=(), active=(), fail=()):
        for sub, names in [
            ("installed", installed),
            ("enabled", enabled),
            ("active", active),
            ("fail", fail),
        ]:
            for name in names:
                (state / sub / UNITS[name]).touch()

    def run(*args, **env):
        result = subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "FAKE_STATE": str(state),
                "XBOTICS_CONFIG_NO_ROOT_CHECK": "1",
                **env,
            },
            timeout=30,
        )
        return result

    def calls(verb):
        return [
            line.split()[-1]
            for line in (state / "calls.log").read_text().splitlines()
            if line.startswith(verb)
        ]

    def enabled():
        return {p.name for p in (state / "enabled").iterdir()}

    def active():
        return {p.name for p in (state / "active").iterdir()}

    return SimpleNamespace(
        set_units=set_units, run=run, calls=calls, enabled=enabled, active=active, state=state
    )


def test_status_marks_missing_units(board):
    board.set_units(installed=["terminal", "jupyter"], enabled=["terminal"], active=["terminal"])
    out = board.run("status").stdout
    assert "terminal         8002   enabled  active   http://10.0.0.2:8002" in out
    assert "calibrate        8006   没装" in out


def test_unknown_name_changes_nothing(board):
    board.set_units(installed=list(UNITS))
    r = board.run("enable", "calibrate", "nope")
    assert r.returncode == 1
    assert "不认识的服务名：nope" in r.stderr
    assert board.calls("enable") == []


def test_failed_enable_is_reported_not_claimed(board):
    """systemctl 失败时如实报错、退出码 1，不能照样说「已开启」。"""
    board.set_units(installed=list(UNITS), fail=["calibrate"])
    r = board.run("enable", "hardware-check", "calibrate")
    assert r.returncode == 1
    assert "已开启 hardware-check" in r.stdout
    assert "已开启 calibrate" not in r.stdout
    assert "calibrate：开启失败：Failed to enable unit" in r.stderr


def test_disable_all_handles_terminal_last(board):
    """关网页终端可能断开当前会话，所以它排在最后，前面的都已经关完。"""
    board.set_units(installed=list(UNITS), enabled=list(UNITS), active=list(UNITS))
    r = board.run("disable", "terminal", "all")
    assert r.returncode == 0
    assert board.calls("disable") == [
        UNITS[n] for n in ("jupyter", "vscode", "hardware-check", "calibrate", "terminal")
    ]
    assert board.enabled() == set()


def test_menu_only_changes_what_the_user_changed(board):
    """菜单初始勾选 = 开机自启的那些；没改的不动 —— 手动起着、没设自启的服务不会被顺手关掉。"""
    board.set_units(
        installed=list(UNITS),
        enabled=["terminal", "jupyter"],
        active=["terminal", "jupyter", "calibrate"],  # calibrate 是手动 start 的
    )
    r = board.run(WHIPTAIL_CHOICE='"terminal" "vscode"')
    assert r.returncode == 0, r.stderr
    assert board.calls("enable") == ["code-server@root"]
    assert board.calls("disable") == ["xbotics-jupyter"]
    assert "xbotics-calibrate-web" in board.active()
    assert "ttyd" in board.active()


def test_menu_survives_an_uninstalled_service(board):
    """勾上没装的服务：别的照常生效，结果框里写明没装，不能整个脚本悄悄退出。"""
    board.set_units(installed=["terminal", "jupyter", "vscode", "calibrate"])
    r = board.run(WHIPTAIL_CHOICE='"hardware-check" "calibrate"')
    assert r.returncode == 0, r.stderr
    assert "xbotics-calibrate-web" in board.enabled()
    log = (board.state / "msgbox.log").read_text()
    assert "hardware-check：板子上没装 xbotics-hardware-check" in log
    assert "已开启 calibrate" in log


def test_menu_cancel_changes_nothing(board):
    board.set_units(installed=list(UNITS), enabled=["terminal"], active=["terminal"])
    r = board.run(WHIPTAIL_CHOICE="", WHIPTAIL_RC="1")
    assert r.returncode == 0
    assert board.calls("enable") == [] and board.calls("disable") == []


def test_non_root_is_refused(board):
    r = subprocess.run(
        ["bash", str(SCRIPT), "enable", "calibrate"],
        capture_output=True,
        text=True,
        env={**os.environ, "XBOTICS_CONFIG_NO_ROOT_CHECK": ""},
        timeout=30,
    )
    if os.geteuid() == 0:
        pytest.skip("以 root 跑测试时看不到这条")
    assert r.returncode == 1 and "要用 root 运行" in r.stderr
