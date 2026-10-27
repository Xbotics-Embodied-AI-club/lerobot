"""推理控制台端到端：真的 gRPC policy_server（策略打桩）↔ lerobot-inference-console 的客户端（假机器人带一路相机）。

验证：相机画面压成 JPEG 上传、服务端解回原尺寸数组；动作块回到客户端并被执行；Web 控制台的接口可用、热调立即生效。
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from concurrent import futures

import numpy as np
import pytest
import torch

pytest.importorskip("grpc")
pytest.importorskip("flask")
pytest.importorskip("msgpack")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

FRAME_SHAPE = (48, 64, 3)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:  # nosec
        return json.loads(response.read())


def test_inference_console_e2e(monkeypatch):
    import grpc

    from lerobot.async_inference import policy_server as policy_server_module
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import JpegImage, map_robot_keys_to_lerobot_features
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
    from lerobot.xbotics.inference_console.client import PolicyRobotClient, PolicyRobotClientConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    # ---------------------------------------------------------------- 假机器人：3 个关节 + 一路相机
    frame = np.random.default_rng(0).integers(0, 255, FRAME_SHAPE, dtype=np.uint8)
    monkeypatch.setattr(MockRobot, "_cameras_ft", property(lambda self: {"top": FRAME_SHAPE}))
    monkeypatch.setattr(
        MockRobot,
        "get_observation",
        lambda self: {**{f"{m}.pos": 0.0 for m in self.motors}, "top": frame.copy()},
    )

    # ---------------------------------------------------------------- 服务端：策略打桩，记下收到的画面
    seen = {"jpeg": 0, "decoded_shapes": set()}
    real_decode = policy_server_module.decode_observation_images

    def spying_decode(observation):
        seen["jpeg"] += isinstance(observation.get("top"), JpegImage)
        decoded = real_decode(observation)
        seen["decoded_shapes"].add(decoded["top"].shape)
        return decoded

    monkeypatch.setattr(policy_server_module, "decode_observation_images", spying_decode)

    server_port = _free_port()
    policy_server = PolicyServer(PolicyServerConfig(host="127.0.0.1", port=server_port, inference_latency=0))
    policy_server.actions_per_chunk = 20
    robot_config = MockRobotConfig()
    policy_server.lerobot_features = map_robot_keys_to_lerobot_features(make_robot_from_config(robot_config))

    def fake_predict(self, obs):
        return self._time_action_chunk(obs.get_timestamp(), [torch.zeros(3)] * 20, obs.get_timestep())

    monkeypatch.setattr(PolicyServer, "_predict_action_chunk", fake_predict)
    monkeypatch.setattr(PolicyServer, "SendPolicyInstructions", lambda self, r, c: services_pb2.Empty())

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"127.0.0.1:{server_port}")
    server.start()

    # ---------------------------------------------------------------- 客户端 + 控制台
    webui_port = _free_port()
    client = PolicyRobotClient(
        PolicyRobotClientConfig(
            robot=robot_config,
            policy_type="act",
            pretrained_name_or_path="test",
            task="pick",
            server_host="127.0.0.1",
            server_port=server_port,
            actions_per_chunk=20,
            fps=30,
            webui_host="127.0.0.1",
            webui_port=webui_port,
            display_queue_bar=False,
        )
    )
    try:
        assert client.start(), "client failed the handshake"
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        control = threading.Thread(target=client.control_loop, kwargs={"task": "pick"}, daemon=True)
        receiver.start()
        control.start()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and client.actions_sent_count < 30:
            time.sleep(0.1)

        assert client.actions_sent_count >= 30, "actions were not executed"
        assert seen["jpeg"] > 0, "camera frames were not sent as JPEG"
        assert seen["decoded_shapes"] == {FRAME_SHAPE}

        base = f"http://127.0.0.1:{webui_port}"
        status = _http(base + "/api/status")
        assert status["running"] and status["task"] == "pick"
        assert status["chunk_size"] == 20
        assert status["transport"] == {"type": "grpc", "address": f"127.0.0.1:{server_port}"}

        assert _http(base + "/api/rtc", {"rtc_blend_steps": 7})["rtc_blend_steps"] == 7
        assert client.config.rtc_blend_steps == 7
        assert _http(base + "/api/task", {"task": "place"})["task"] == "place"
        assert _http(base + "/api/control", {"action": "stop"}) == {"task_active": False}
        sent = client.actions_sent_count
        time.sleep(0.5)
        assert client.actions_sent_count == sent, "stopped task still executes actions"
    finally:
        client.stop()
        policy_server.shutdown_event.set()
        server.stop(grace=None)
