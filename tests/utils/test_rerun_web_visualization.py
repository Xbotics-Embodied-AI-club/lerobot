"""板子出网页的 rerun 预览（Xbotics）：跳转页、跨源放行、限频取最新、没有桌面时自动改走网页。

不起 rerun 服务（CI 上可以没有 rerun），只测纯逻辑与调度。真浏览器里的效果在板子上实测
（电脑上的 Firefox 只开 http://<板子>:9090，出两路画面与关节曲线）。
"""

import inspect

import numpy as np
import pytest

from lerobot.utils import keyboard_input as ki, rerun_web_visualization as web, visualization_utils as vu


def test_web_is_a_visualization_mode():
    assert "web" in vu.VISUALIZATION_MODES


@pytest.mark.parametrize(
    "host, expected_host",
    [("192.168.1.50:9090", "192.168.1.50"), ("board.local", "board.local"), ("[fe80::1]:9090", "[fe80::1]")],
)
def test_viewer_url_uses_the_host_the_browser_used(host, expected_host):
    url = web.viewer_url(host, grpc_port=9876, viewer_port=9091)
    assert url == (
        f"http://{expected_host}:9091/?url=rerun%2Bhttp%3A%2F%2F{expected_host.replace('[', '%5B').replace(']', '%5D').replace(':', '%3A')}"
        "%3A9876%2Fproxy"
    )


def test_cors_allows_the_viewer_port_when_rerun_supports_it():
    def serve_grpc_033(*, grpc_port=None, server_memory_limit="1GiB", cors_allow_origin=None):
        return ""

    def serve_grpc_026(*, grpc_port=None, server_memory_limit="25%"):
        return ""

    assert web.cors_kwargs(serve_grpc_033, viewer_port=9091) == {"cors_allow_origin": ["http://*:9091"]}
    # rerun 0.26 不按来源拦跨源，也没有这个参数
    assert web.cors_kwargs(serve_grpc_026, viewer_port=9091) == {}
    assert "cors_allow_origin" not in inspect.signature(serve_grpc_026).parameters


def test_latest_slot_keeps_only_the_newest_and_respects_the_rate():
    slot = web.LatestSlot(interval_s=0.1)
    assert slot.due(now=0.0)
    slot.put("a", now=0.0)
    assert not slot.due(now=0.05)
    assert slot.due(now=0.1)
    slot.put("b", now=0.1)
    slot.put("c", now=0.12)  # 没取走的被新的顶掉
    assert slot.take() == "c"
    assert slot.take() is None


def test_offer_copies_images_and_flattens_scalars():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    item = web.snapshot({"top": frame, "shoulder_pan.pos": 1.5}, {"shoulder_pan.pos": 2.0})
    frame[:] = 255  # 相机驱动复用缓冲区时不能影响已取的那份
    assert int(item.images["observation.top"].max()) == 0
    assert item.observation_scalars == {"observation.shoulder_pan.pos": 1.5}
    assert item.action_scalars == {"action.shoulder_pan.pos": 2.0}


@pytest.fixture
def backends(monkeypatch, tmp_path):
    # 「有桌面」按 X 服务的 socket 判：这里只有 :0 在跑
    (tmp_path / "X0").touch()
    monkeypatch.setattr(ki, "X11_SOCKET_DIR", tmp_path)
    ki.is_headless.cache_clear()
    calls = []
    for name in (
        "init_rerun",
        "log_rerun_data",
        "shutdown_rerun",
        "init_rerun_web",
        "log_rerun_web_data",
        "shutdown_rerun_web",
    ):
        monkeypatch.setattr(vu, name, lambda *a, _n=name, **k: calls.append(_n))
    yield calls
    vu._web_override = False
    ki.is_headless.cache_clear()


def test_rerun_without_a_desktop_becomes_web(backends, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    vu.init_visualization("rerun")
    vu.log_visualization_data("rerun", observation={})
    vu.shutdown_visualization("rerun")
    assert backends == ["init_rerun_web", "log_rerun_web_data", "shutdown_rerun_web"]


def test_rerun_with_a_desktop_stays_native(backends, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    vu.init_visualization("rerun")
    vu.log_visualization_data("rerun", observation={})
    assert backends == ["init_rerun", "log_rerun_data"]


def test_display_set_but_no_x_server_becomes_web(backends, monkeypatch):
    """板子镜像在登录 shell 里写死 DISPLAY=:0，桌面关了也照设；网页终端是登录 shell。

    当成有桌面会去拉原生查看器，拉不起来时 rerun 把每帧原图攒在内存里，几十秒就把 4 GB 的板子吃光。
    """
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    vu.init_visualization("rerun")
    assert backends == ["init_rerun_web"]


def test_remote_rerun_server_is_respected_without_a_desktop(backends, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    vu.init_visualization("rerun", ip="10.0.0.2", port=9876)
    assert backends == ["init_rerun"]


def test_web_mode_can_be_chosen_explicitly(backends, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    vu.init_visualization("web")
    vu.log_visualization_data("web", observation={})
    assert backends == ["init_rerun_web", "log_rerun_web_data"]
