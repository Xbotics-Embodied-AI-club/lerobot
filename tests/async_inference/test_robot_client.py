# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import pickle
import time
from queue import Queue

import numpy as np
import pytest
import torch

# Skip entire module if required deps are not available
pytest.importorskip("grpc")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client():
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


# -----------------------------------------------------------------------------
# Regression test: robot type registry populated by robot_client imports
# -----------------------------------------------------------------------------


def test_robot_client_registers_builtin_robot_types():
    """Importing robot_client must populate RobotConfig's ChoiceRegistry.

    This is a regression test for a bug introduced in #2425, where removing
    robot module imports from robot_client.py caused RobotConfig's registry to
    be empty, breaking CLI argument parsing with:
      error: argument --robot.type: invalid choice: 'so101_follower' (choose from )

    Robot types are registered via @RobotConfig.register_subclass() decorators
    at import time, so all supported modules must be explicitly imported.
    """
    import lerobot.async_inference.robot_client  # noqa: F401
    from lerobot.robots.config import RobotConfig

    known_choices = RobotConfig.get_known_choices()

    expected_robot_types = [
        "so100_follower",
        "so101_follower",
        "koch_follower",
        "omx_follower",
        "bi_so_follower",
    ]
    for robot_type in expected_robot_types:
        assert robot_type in known_choices, (
            f"Robot type '{robot_type}' is not registered in RobotConfig's ChoiceRegistry. "
            f"Ensure the corresponding module is imported in robot_client.py. "
            f"Known choices: {sorted(known_choices)}"
        )


# -----------------------------------------------------------------------------
# 观测上传：压 JPEG + 后台线程发送（控制循环不等网络）
# -----------------------------------------------------------------------------


def _make_timed_obs(timestep: int, must_go: bool = False, frame: np.ndarray | None = None):
    from lerobot.async_inference.helpers import TimedObservation

    observation = {"shoulder_pan.pos": float(timestep)}
    if frame is not None:
        observation["top"] = frame
    return TimedObservation(
        timestamp=time.time(), timestep=timestep, observation=observation, must_go=must_go
    )


def _record_sends(robot_client, monkeypatch) -> list:
    sent = []

    def fake_send(obs):
        sent.append(obs)
        return True

    monkeypatch.setattr(robot_client, "send_observation", fake_send)
    return sent


def test_background_sender_sends_only_the_latest_observation(robot_client, monkeypatch):
    """发得慢时，没来得及发的旧观测被新的顶掉：服务端永远拿到最新的画面。"""
    sent = _record_sends(robot_client, monkeypatch)

    robot_client._submit_observation(_make_timed_obs(1))
    robot_client._submit_observation(_make_timed_obs(2))
    robot_client._send_pending_observation()
    robot_client._send_pending_observation()

    assert [obs.get_timestep() for obs in sent] == [2]


def test_background_sender_carries_must_go_to_the_replacing_observation(robot_client, monkeypatch):
    """被顶掉的那条若是 must_go（动作队列已空），顶替它的那条要接过标记，否则服务端可能不推理。"""
    sent = _record_sends(robot_client, monkeypatch)

    robot_client._submit_observation(_make_timed_obs(1, must_go=True))
    robot_client._submit_observation(_make_timed_obs(2, must_go=False))
    robot_client._send_pending_observation()

    assert sent[0].get_timestep() == 2
    assert sent[0].must_go


def test_control_loop_observation_hands_off_to_background_sender(robot_client, monkeypatch):
    sent = _record_sends(robot_client, monkeypatch)

    robot_client.control_loop_observation(task="pick")
    assert sent == [], "控制循环不应等发送"

    robot_client._send_pending_observation()
    assert len(sent) == 1
    assert sent[0].get_observation()["task"] == "pick"


def test_control_loop_observation_sends_inline_when_background_disabled(robot_client, monkeypatch):
    robot_client.config.send_observations_in_background = False
    sent = _record_sends(robot_client, monkeypatch)

    robot_client.control_loop_observation(task="pick")

    assert len(sent) == 1


def _capture_uploads(robot_client, monkeypatch) -> list:
    payloads = []

    def fake_send_observations(iterator):
        payloads.append(b"".join(message.data for message in iterator))

    monkeypatch.setattr(robot_client.stub, "SendObservations", fake_send_observations)
    return payloads


def test_send_observation_uploads_jpeg_frames(robot_client, monkeypatch):
    from lerobot.async_inference.helpers import JpegImage

    payloads = _capture_uploads(robot_client, monkeypatch)
    frame = np.full((48, 64, 3), 120, dtype=np.uint8)

    assert robot_client.send_observation(_make_timed_obs(3, frame=frame))

    uploaded = pickle.loads(payloads[0]).get_observation()  # nosec
    assert isinstance(uploaded["top"], JpegImage)
    assert uploaded["shoulder_pan.pos"] == 3.0


def test_send_observation_uploads_raw_frames_when_jpeg_disabled(robot_client, monkeypatch):
    robot_client.config.observation_jpeg_quality = 0
    payloads = _capture_uploads(robot_client, monkeypatch)
    frame = np.full((48, 64, 3), 120, dtype=np.uint8)

    assert robot_client.send_observation(_make_timed_obs(3, frame=frame))

    uploaded = pickle.loads(payloads[0]).get_observation()  # nosec
    assert isinstance(uploaded["top"], np.ndarray)
