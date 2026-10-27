"""带 Web 控制台的推理客户端（lerobot.xbotics.inference_console）：不接机器人、不连服务端，测各段逻辑。"""

import pickle  # nosec
import threading
import time
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("grpc")
pytest.importorskip("flask")
pytest.importorskip("msgpack")

from lerobot.async_inference.helpers import JpegImage, TimedAction, TimedObservation  # noqa: E402
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: E402
from lerobot.xbotics.inference_console.client import (  # noqa: E402
    ChunkUsageStats,
    InferenceLatencyTracker,
    PolicyRobotClient,
    PolicyRobotClientConfig,
)
from lerobot.xbotics.inference_console.msgpack_numpy import packb, unpackb  # noqa: E402
from lerobot.xbotics.inference_console.transport import (  # noqa: E402
    GrpcTransport,
    _ws_action_to_timed_actions,
)


def make_config(**overrides):
    return PolicyRobotClientConfig(
        robot=SO101FollowerConfig(port="/dev/null", id="test"),
        policy_type="act",
        pretrained_name_or_path="dummy",
        **overrides,
    )


# ---------------------------------------------------------------- 配置


def test_defaults_console_on_8007_and_jpeg_upload():
    cfg = make_config()
    assert cfg.webui_enabled
    assert (cfg.webui_host, cfg.webui_port) == ("0.0.0.0", 8007)
    assert cfg.transport == "grpc"
    assert cfg.observation_jpeg_quality == 90


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"observation_jpeg_quality": 101}, "observation_jpeg_quality"),
        ({"transport": "http"}, "transport"),
        ({"transport": "websocket"}, "ws_host"),
        ({"rtc_lookahead_ratio": 1.5}, "rtc_lookahead_ratio"),
        ({"action_interp_steps": 0}, "action_interp_steps"),
    ],
)
def test_invalid_config_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_config(**overrides)


# ---------------------------------------------------------------- gRPC 传输：默认压 JPEG


class _RecordingStub:
    def __init__(self):
        self.payload = b""

    def SendObservations(self, iterator):  # noqa: N802
        self.payload = b"".join(chunk.data for chunk in iterator)


def _transport(quality):
    transport = GrpcTransport.__new__(GrpcTransport)
    transport.config = SimpleNamespace(observation_jpeg_quality=quality)
    transport.logger = SimpleNamespace(debug=lambda *a, **k: None, error=lambda *a, **k: None)
    transport.shutdown_event = threading.Event()
    transport.stub = _RecordingStub()
    return transport


def _observation():
    frame = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    return TimedObservation(
        timestamp=time.time(), timestep=0, observation={"top": frame, "shoulder_pan.pos": 1.0, "task": "x"}
    )


def test_grpc_transport_sends_jpeg_by_default():
    transport = _transport(90)
    obs = _observation()
    assert transport._send_observation(obs)

    sent = pickle.loads(transport.stub.payload)  # nosec
    assert isinstance(sent.get_observation()["top"], JpegImage)
    assert sent.get_observation()["shoulder_pan.pos"] == 1.0
    assert isinstance(obs.get_observation()["top"], np.ndarray), "不该改动控制循环手里的那份观测"


def test_grpc_transport_sends_raw_when_quality_zero():
    transport = _transport(0)
    obs = _observation()
    transport._send_observation(obs)

    sent = pickle.loads(transport.stub.payload)  # nosec
    np.testing.assert_array_equal(sent.get_observation()["top"], obs.get_observation()["top"])


# ---------------------------------------------------------------- RTC：融合与触发


def _bare_client(**config_overrides):
    client = PolicyRobotClient.__new__(PolicyRobotClient)
    client.config = make_config(**config_overrides)
    client.action_queue = Queue()
    client.action_queue_lock = threading.Lock()
    client.latest_action_lock = threading.Lock()
    client.latest_action = -1
    client._queue_viz = []
    client.action_chunk_size = -1
    client._chunk_size_threshold = client.config.chunk_size_threshold
    client.latency_tracker = InferenceLatencyTracker(window_size=5, init_latency_ms=100.0)
    client._obs_overhead_tracker = InferenceLatencyTracker(window_size=5, init_latency_ms=0.0)
    return client


def _chunk(start, length, value):
    return [
        TimedAction(timestamp=0.0, timestep=start + i, action=torch.full((6,), float(value)))
        for i in range(length)
    ]


def test_blend_ramps_old_to_new_and_reaches_new_at_end_of_region():
    client = _bare_client(rtc_enabled=True, rtc_blend_steps=4)
    client._aggregate_action_queues(_chunk(0, 10, 0.0))
    client._aggregate_action_queues(_chunk(5, 10, 1.0))  # 与旧块在 5..9 重叠

    queue = {a.get_timestep(): a.get_action()[0].item() for a in client.action_queue.queue}
    assert [queue[t] for t in range(5, 9)] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert queue[9] == 1.0
    assert set(queue) == set(range(0, 15))
    assert {e["src"] for e in client._queue_viz if 5 <= e["ts"] <= 8} == {"blend"}


def test_actions_already_executed_are_dropped():
    client = _bare_client(rtc_enabled=True, rtc_blend_steps=0)
    client.latest_action = 4
    client._aggregate_action_queues(_chunk(0, 10, 1.0))
    assert min(a.get_timestep() for a in client.action_queue.queue) == 5


def test_rtc_disabled_uses_aggregate_function():
    client = _bare_client(rtc_enabled=False)
    client._aggregate_action_queues(_chunk(0, 4, 0.0))
    client._aggregate_action_queues(_chunk(0, 4, 1.0), lambda old, new: (old + new) / 2)
    assert {a.get_action()[0].item() for a in client.action_queue.queue} == {0.5}


def test_trigger_point_follows_latency_floor_and_chunk_cap():
    client = _bare_client(rtc_enabled=True, fps=10, rtc_lookahead_ratio=0.0, rtc_safety_margin_steps=0)
    client.action_chunk_size = 20
    # 延迟 100 ms、控制周期 100 ms ⇒ 预计消耗 1 步，触发点 = ceil(1 + 1) = 2
    for n, expected in [(3, False), (2, True)]:
        client.action_queue = Queue()
        for a in _chunk(0, n, 0.0):
            client.action_queue.put(a)
        assert client._ready_to_send_observation() is expected

    # lookahead 下界：20 × 0.5 = 10 步
    client.config.rtc_lookahead_ratio = 0.5
    client.action_queue = Queue()
    for a in _chunk(0, 10, 0.0):
        client.action_queue.put(a)
    assert client._ready_to_send_observation()

    # 推理比整个 chunk 还慢：触发点封顶为 chunk 长度，一直要
    client.config.rtc_lookahead_ratio = 0.0
    client.latency_tracker.update(10_000)
    client.action_queue = Queue()
    for a in _chunk(0, 20, 0.0):
        client.action_queue.put(a)
    assert client._ready_to_send_observation()


def test_chunk_usage_settles_on_next_arrival():
    stats = ChunkUsageStats()
    stats.settle_previous(queue_size_before_merge=0, nominal_size=20)
    stats.settle_previous(queue_size_before_merge=5, nominal_size=20)
    snap = stats.snapshot()
    assert snap["chunk_usage_ratio"] == pytest.approx(0.75)
    assert snap["chunk_refresh_depth"] == pytest.approx(0.25)


# ---------------------------------------------------------------- 运行时热调


def test_set_webui_rtc_validates_and_applies():
    client = _bare_client()
    client.fps_tracker = SimpleNamespace(target_fps=30)
    client.logger = SimpleNamespace(info=lambda *a, **k: None)

    result = client.set_webui_rtc({"rtc_blend_steps": "8", "rtc_enabled": "false", "fps": 15})
    assert result["rtc_blend_steps"] == 8
    assert result["rtc_enabled"] is False
    assert client.fps_tracker.target_fps == 15

    with pytest.raises(ValueError, match="unknown params"):
        client.set_webui_rtc({"server_port": 1})
    with pytest.raises(ValueError, match="rtc_action_ema_alpha"):
        client.set_webui_rtc({"rtc_action_ema_alpha": 2})

    # 一组里有一个非法：整组都不生效
    with pytest.raises(ValueError, match="rtc_action_ema_alpha"):
        client.set_webui_rtc({"rtc_blend_steps": 3, "rtc_action_ema_alpha": 2})
    assert client.config.rtc_blend_steps == 8


def test_websocket_config_does_not_need_a_model_path():
    cfg = PolicyRobotClientConfig(
        robot=SO101FollowerConfig(port="/dev/null", id="test"),
        transport="websocket",
        ws_host="ws://example",
        ws_robo_name="so101",
    )
    assert cfg.pretrained_name_or_path == ""
    with pytest.raises(ValueError, match="pretrained_name_or_path"):
        PolicyRobotClientConfig(robot=SO101FollowerConfig(port="/dev/null", id="test"), policy_type="act")


def test_websocket_transport_infers_each_observation_once():
    from lerobot.xbotics.inference_console.transport import WebsocketTransport

    calls = []

    class FakePolicy:
        uri = "ws://fake"
        last_timing = None

        def infer(self, obs):
            calls.append(obs["observation.state"].copy())
            return {"action": np.zeros((2, 1), dtype=np.float32)}

    transport = WebsocketTransport.__new__(WebsocketTransport)
    transport.config = SimpleNamespace(
        ws_obs_image_size=0,
        ws_image_codec="raw",
        ws_jpeg_quality=70,
        actions_per_chunk=2,
        action_output_keys=[],
    )
    transport.robot = SimpleNamespace(observation_features={"j.pos": float}, action_features={"j.pos": float})
    transport.logger = SimpleNamespace(
        warning=lambda *a, **k: None, info=lambda *a, **k: None, debug=lambda *a, **k: None, error=print
    )
    transport._latest_lock = threading.Lock()
    transport._latest_obs = None
    transport._closed = threading.Event()
    transport._policy = FakePolicy()

    transport.submit_observation(TimedObservation(timestamp=0.0, timestep=0, observation={"j.pos": 1.0}))
    assert transport.receive_chunk() is not None

    # 没有新观测时不拿旧的再推一次：receive_chunk 一直等，直到关闭
    threading.Timer(0.2, transport._closed.set).start()
    assert transport.receive_chunk() is None
    assert len(calls) == 1


def test_webui_routes():
    from lerobot.xbotics.inference_console.webui import _create_app

    calls = {}

    class FakeClient:
        config = SimpleNamespace(webui_video_enabled=False)
        running = True
        current_task = "pick"
        task_active = True

        def get_webui_status(self):
            return {"task": self.current_task}

        def get_webui_rtc(self):
            return {"fps": 30}

        def set_webui_rtc(self, params):
            raise ValueError("bad")

        def set_task(self, task):
            calls["task"] = task

        def set_task_active(self, active):
            calls["active"] = active
            return {"task_active": active}

    client = _create_app(FakeClient()).test_client()
    assert "推理控制台" in client.get("/").get_data(as_text=True)
    assert client.get("/api/status").get_json() == {"task": "pick"}
    assert client.post("/api/rtc", json={"fps": 1}).status_code == 400
    client.post("/api/task", json={"task": "place"})
    assert calls["task"] == "place"
    client.post("/api/control", json={"action": "stop"})
    assert calls["active"] is False
    assert client.get("/stream/top").status_code == 503  # 视频推流关着


# ---------------------------------------------------------------- WebSocket（LingBot-VLA）协议


def test_msgpack_numpy_roundtrip():
    data = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.float32(1.5), "c": "x"}
    out = unpackb(packb(data))
    np.testing.assert_array_equal(out["a"], data["a"])
    assert out["b"] == np.float32(1.5)
    assert out["c"] == "x"


def test_ws_actions_follow_configured_key_order():
    cfg = SimpleNamespace(actions_per_chunk=3, action_output_keys=["arm", "grip"])
    robot = SimpleNamespace(action_features={f"m{i}": float for i in range(4)})
    log = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    ws_action = {"grip": np.ones((5, 1)), "arm": np.zeros((5, 3)), "server_timing": {}}

    actions = _ws_action_to_timed_actions(ws_action, 10, cfg, robot, log)

    assert [a.get_timestep() for a in actions] == [10, 11, 12]
    assert actions[0].get_action().tolist() == [0.0, 0.0, 0.0, 1.0]
