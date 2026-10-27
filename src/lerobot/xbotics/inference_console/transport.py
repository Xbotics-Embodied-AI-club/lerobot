"""推理传输层：把「观测上传 + 动作块接收」抽成与协议无关的两个原语。

两种传输：
- GrpcTransport：LeRobot 的 gRPC 协议（Ready / SendPolicyInstructions / SendObservations / GetActions），
  连的就是 ``python -m lerobot.async_inference.policy_server``。后台线程发送（只留最新一条），
  相机画面默认压 JPEG 再发 —— 与维护版 ``robot_client`` 同一个实现（``encode_observation_images``），
  服务端解回数组；``--observation_jpeg_quality=0`` 发原图。
  ``receive_chunk()`` 轮询 GetActions（服务端推理完才返回），延迟 = 收包时刻 − 动作块首个时间戳。
- WebsocketTransport：LingBot-VLA 官方推理服务的 WebSocket 协议（msgpack，一问一答 infer + reset）。
  ``submit_observation`` 只留最新观测，``receive_chunk()`` 取它做一次往返，延迟即实测往返耗时。

上层客户端（client.PolicyRobotClient）只依赖 PolicyTransport 接口与 create_transport，RTC / EMA / 控制台与协议无关。
"""

import contextlib
import io
import pickle  # nosec
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from queue import Empty, Full, Queue
from typing import Any

import grpc
import numpy as np
import torch
from PIL import Image

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    encode_observation_images,
    map_robot_keys_to_lerobot_features,
)
from lerobot.transport import (  # type: ignore
    services_pb2,
    services_pb2_grpc,
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .image_tools import convert_to_uint8, resize_with_pad
from .websocket_client_policy import WebsocketClientPolicy


class PolicyTransport(ABC):
    """推理传输抽象接口（非线程安全约束见各实现）。"""

    @abstractmethod
    def setup(self) -> bool:
        """握手 + 策略下发/重置。返回 True 表示就绪。"""

    @abstractmethod
    def submit_observation(self, obs: TimedObservation) -> None:
        """非阻塞提交观测；实现内部只保留最新观测（替换旧值）。"""

    @abstractmethod
    def receive_chunk(self) -> tuple[list[TimedAction], float] | None:
        """阻塞直到拿到一个动作块，返回 (TimedAction 列表, 推理往返延迟 ms)。

        无新块/出错/关闭时返回 None，调用方继续轮询。"""

    @abstractmethod
    def close(self) -> None:
        """停止后台资源（线程/连接）。"""

    @property
    @abstractmethod
    def peer(self) -> str:
        """对端地址描述（用于日志/WebUI status）。"""


class GrpcTransport(PolicyTransport):
    """LeRobot 的 gRPC 异步推理传输。

    - 发送线程消费内部队列（maxsize=1，新的顶掉旧的），压 JPEG、pickle、SendObservations；
    - receive_chunk() 由调用方线程执行，轮询 GetActions。
    """

    def __init__(self, config, robot, logger):
        self.config = config
        self.robot = robot
        self.logger = logger

        lerobot_features = map_robot_keys_to_lerobot_features(robot)
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            config.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.shutdown_event = threading.Event()
        # 观测发送队列：maxsize=1，永远发送最新观测
        self.obs_send_queue: Queue = Queue(maxsize=1)
        self._sender_thread: threading.Thread | None = None

    @property
    def peer(self) -> str:
        return self.config.server_address

    def setup(self) -> bool:
        """Ready 握手 + SendPolicyInstructions 下发策略，并启动观测发送线程。"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            # 观测发送线程：pickle + RPC 上传在后台执行，控制循环零网络 I/O。
            self._sender_thread = threading.Thread(
                target=self._observation_sender, daemon=True, name="obs_sender"
            )
            self._sender_thread.start()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def submit_observation(self, obs: TimedObservation) -> None:
        """队列满则替换旧观测（始终发送最新）。"""
        try:
            self.obs_send_queue.put_nowait(obs)
        except Full:
            with contextlib.suppress(Empty):
                self.obs_send_queue.get_nowait()
            self.obs_send_queue.put_nowait(obs)

    def _send_observation(self, obs: TimedObservation) -> bool:
        """压 JPEG + pickle 序列化 + SendObservations 一元调用（发送线程内执行）。"""
        if self.shutdown_event.is_set():
            raise RuntimeError("Transport closed.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        if self.config.observation_jpeg_quality > 0:
            obs = replace(
                obs,
                observation=encode_observation_images(
                    obs.get_observation(), self.config.observation_jpeg_quality
                ),
            )
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _observation_sender(self):
        """观测发送后台线程：消费 obs_send_queue，压缩、序列化并上传。

        发送耗时（压缩 + 一次完整 RPC 往返，几十到几百 ms）不占控制循环；队列满时只留最新观测。
        """
        self.logger.info("Observation sender thread starting")
        while not self.shutdown_event.is_set():
            try:
                obs = self.obs_send_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._send_observation(obs)
            except RuntimeError:
                break  # transport closed
            except Exception as e:
                self.logger.error(f"Error in observation sender: {e}")

    def receive_chunk(self) -> tuple[list[TimedAction], float] | None:
        """轮询 GetActions 拿一个动作块（服务器阻塞至推理完成）。"""
        try:
            # Poll GetActions to get an action chunk from the server
            actions_chunk = self.stub.GetActions(services_pb2.Empty())
            if len(actions_chunk.data) == 0:
                return None  # received `Empty` from server, wait for next call

            receive_time = time.time()

            # Deserialize bytes back into list[TimedAction]
            deserialize_start = time.perf_counter()
            timed_actions = pickle.loads(actions_chunk.data)  # nosec
            deserialize_time = time.perf_counter() - deserialize_start

            if len(timed_actions) == 0:
                return None

            # Log device type of received actions
            received_device = timed_actions[0].get_action().device.type
            self.logger.debug(f"Received actions on device: {received_device}")

            # Move actions to client_device (e.g., for downstream planners that need GPU)
            client_device = self.config.client_device
            if client_device != "cpu":
                for timed_action in timed_actions:
                    if timed_action.get_action().device.type != client_device:
                        timed_action.action = timed_action.get_action().to(client_device)
                self.logger.debug(f"Converted actions to device: {client_device}")
            else:
                self.logger.debug(f"Actions kept on device: {client_device}")

            # 延迟估计 = 收包时刻 − 动作首元素时间戳（时间戳在推理完成后由服务器打，
            # 因此近似 推理+网络 耗时）
            server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

            self.logger.debug(f"Deserialization time: {deserialize_time * 1000:.2f}ms")
            return timed_actions, server_to_client_latency

        except grpc.RpcError as e:
            self.logger.error(f"Error receiving actions: {e}")
            return None

    def close(self) -> None:
        self.shutdown_event.set()
        if self._sender_thread is not None and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=2)
        self.channel.close()
        self.logger.debug("Transport closed, channel closed")


class WebsocketTransport(PolicyTransport):
    """LingBot-VLA WebSocket 传输（msgpack 协议）。

    - setup()：连接 + 收服务器 metadata + reset(robo_name)；
    - submit_observation()：写入 latest-slot（替换旧值）；
    - receive_chunk()：取最新观测 → policy.infer() 阻塞往返 → 动作格式转换。
      延迟为 infer() 实测往返（直接喂 RTC 触发估计）。must_go 对该传输无意义
      （服务器无去重），忽略。
    """

    def __init__(self, config, robot, logger):
        self.config = config
        self.robot = robot
        self.logger = logger

        self._latest_lock = threading.Lock()
        self._latest_obs: TimedObservation | None = None
        self._closed = threading.Event()
        self._policy: WebsocketClientPolicy | None = None

    @property
    def peer(self) -> str:
        return self._policy.uri if self._policy is not None else "(not connected)"

    def setup(self) -> bool:
        try:
            self._policy = WebsocketClientPolicy(
                host=self.config.ws_host,
                port=self.config.ws_port,
                path=self.config.ws_path,
                use_ssl=self.config.ws_use_ssl,
                api_key=self.config.ws_api_key or None,
            )
            metadata = self._policy.get_server_metadata()
            self.logger.info(f"Connected to LingBot-VLA server, metadata: {metadata}")
            self.logger.info(f"Sending reset (robo_name={self.config.ws_robo_name})")
            self._policy.reset(self.config.ws_robo_name)
            self._closed.clear()
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to LingBot-VLA server: {e}")
            return False

    def submit_observation(self, obs: TimedObservation) -> None:
        with self._latest_lock:
            self._latest_obs = obs

    def receive_chunk(self) -> tuple[list[TimedAction], float] | None:
        # 等待最新观测（控制循环提交）；关闭时返回 None
        while not self._closed.is_set():
            with self._latest_lock:
                # 取走即清空：每条观测只推一次；控制循环不交新观测（队列还够）时就不再拿旧画面重复推理
                obs, self._latest_obs = self._latest_obs, None
            if obs is not None:
                break
            time.sleep(0.005)
        else:
            return None
        if self._closed.is_set():
            return None

        raw_observation = obs.get_observation()
        task = raw_observation.get("task", "")
        ws_obs = _build_ws_observation(raw_observation, task, self.config, self.robot, self.logger)

        t0 = time.perf_counter()
        try:
            action_dict = self._policy.infer(ws_obs)
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        timing = getattr(self._policy, "last_timing", None)
        if timing:
            self.logger.debug(
                f"[Latency breakdown] pack={timing['pack_ms']:.0f}ms | "
                f"roundtrip={timing['roundtrip_ms']:.0f}ms | "
                f"unpack={timing['unpack_ms']:.0f}ms | "
                f"upload={timing['upload_mb']:.2f}MB | download={timing['download_mb']:.2f}MB"
            )

        try:
            timed_actions = _ws_action_to_timed_actions(
                action_dict, obs.get_timestep() + 1, self.config, self.robot, self.logger
            )
        except Exception as e:
            self.logger.error(f"Action conversion error: {e}")
            return None

        if len(timed_actions) == 0:
            return None
        return timed_actions, latency_ms

    def close(self) -> None:
        self._closed.set()
        if self._policy is not None:
            with contextlib.suppress(Exception):
                self._policy.close()
        self.logger.debug("Transport closed")


# ---------------- WebSocket 协议格式转换（移植自 feat/lingbot-vla-client 分支） ----------------


def _build_ws_observation(raw_observation: dict, task: str, config, robot, logger) -> dict:
    """Convert LeRobot RawObservation to LingBot-VLA observation format.

    LingBot-VLA expects:
    - images: dict of {camera_name: np.uint8 HWC image | JPEG bytes}
    - observation.state: np.float32 1D array (concatenated motor positions in feature order)
    - task: string
    """
    ws_obs = {}
    state_parts = []

    # Use robot's observation_features to distinguish images from state
    # and preserve the correct ordering for state concatenation.
    obs_features = robot.observation_features

    for key in obs_features:
        if key not in raw_observation:
            logger.warning(f"Missing observation key: {key}")
            continue

        value = raw_observation[key]

        # Camera features are tuples like (H, W, 3)
        if isinstance(obs_features[key], tuple):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
                img = convert_to_uint8(value)
            elif isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 3:
                img = value.cpu()
                if img.dtype == torch.float32:
                    img = (img * 255).clamp(0, 255)
                img = img.byte().permute(1, 2, 0).numpy()
            else:
                ws_obs[key] = value
                continue
            # 传输优化：resize（可选） + 编码（raw / jpeg）
            ws_obs[key] = _encode_image_for_transport(img, config)
        else:
            # State / motor features
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            elif not isinstance(value, np.ndarray):
                value = np.array([value], dtype=np.float32)

            if value.ndim == 0:
                value = value.reshape(1)
            state_parts.append(value.astype(np.float32).flatten())

    if state_parts:
        ws_obs["observation.state"] = np.concatenate(state_parts).astype(np.float32)

    ws_obs["task"] = task if task else ""

    return ws_obs


def _encode_image_for_transport(img: np.ndarray, config) -> np.ndarray | bytes:
    """将 HWC uint8 图像按配置进行 resize 和编码，降低网络上传量。

    - ws_obs_image_size > 0: resize_with_pad 到 (size, size)
    - ws_image_codec == 'jpeg': 编码为 JPEG bytes（服务器端需解码）
    """
    if config.ws_obs_image_size > 0:
        img = resize_with_pad(img[None], config.ws_obs_image_size, config.ws_obs_image_size)[0]
    if config.ws_image_codec == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=config.ws_jpeg_quality)
        return buf.getvalue()
    return img


def _ws_action_to_timed_actions(
    ws_action: dict[str, Any], start_timestep: int, config, robot, logger
) -> list[TimedAction]:
    """Convert LingBot-VLA action dict to list of LeRobot TimedAction.

    LingBot-VLA returns action chunks like:
    {
        "action.arm.position": np.ndarray (actions_per_chunk, 12),
        "action.effector.position": np.ndarray (actions_per_chunk, 2),
        "server_timing": {...},  # metadata, ignore
    }

    双臂机器人按 [左臂6, 左夹爪, 右臂6, 右夹爪] 重组；
    泛型机器人走 action_output_keys 拼接回退（缺省按 key 排序）。
    服务器不返回 timestep，客户端用 start_timestep + t 自行编号。
    """
    # Filter out metadata keys
    candidate_keys = [k for k in ws_action if isinstance(ws_action[k], np.ndarray)]
    action_keys = [k for k in candidate_keys if ws_action[k].ndim == 2]

    if not action_keys:
        logger.warning("No valid action chunk found in LingBot-VLA output")
        return []

    # Special handling for bi-manual robots: reassemble arm + effector into correct order
    if "action.arm.position" in action_keys and "action.effector.position" in action_keys:
        arm = ws_action["action.arm.position"]  # (chunk_len, 12)
        effector = ws_action["action.effector.position"]  # (chunk_len, 2)
        chunk_len = arm.shape[0]

        # Log reassembled action chunk stats
        reassembled = np.concatenate(
            [
                arm[:, 0:6],
                effector[:, 0:1],
                arm[:, 6:12],
                effector[:, 1:2],
            ],
            axis=1,
        )
        logger.info(
            f"[Reassemble] action.arm.position + action.effector.position -> "
            f"shape={reassembled.shape} | "
            f"mean={np.mean(reassembled, axis=0).round(4).tolist()} | "
            f"std={np.std(reassembled, axis=0).round(4).tolist()} | "
            f"min={np.min(reassembled, axis=0).round(4).tolist()} | "
            f"max={np.max(reassembled, axis=0).round(4).tolist()}"
        )

        timed_actions = []
        for t in range(min(chunk_len, config.actions_per_chunk)):
            # Reassemble: [left_arm_joints(6), left_gripper(1), right_arm_joints(6), right_gripper(1)]
            action_vector = torch.from_numpy(
                np.concatenate(
                    [
                        arm[t, 0:6],
                        effector[t, 0:1],
                        arm[t, 6:12],
                        effector[t, 1:2],
                    ]
                )
            ).float()
            timed_actions.append(
                TimedAction(
                    timestamp=time.time(),
                    timestep=start_timestep + t,
                    action=action_vector,
                )
            )
        return timed_actions

    # Generic fallback for other robot types (handles flat 'action' key from server)
    ordered_keys = config.action_output_keys if config.action_output_keys else sorted(action_keys)

    for key in ordered_keys:
        if key not in ws_action:
            raise ValueError(
                f"Configured action key '{key}' not found in LingBot-VLA output. "
                f"Available keys: {list(ws_action.keys())}. "
                f"Please set --action_output_keys correctly."
            )

    chunk_len = ws_action[ordered_keys[0]].shape[0]

    expected_dim = sum(ws_action[k].shape[1] for k in ordered_keys)
    robot_action_dim = len(robot.action_features)
    if expected_dim != robot_action_dim:
        logger.warning(
            f"Action dimension mismatch: LingBot-VLA output dim={expected_dim} "
            f"(keys={ordered_keys}), robot expects dim={robot_action_dim}. "
            f"You may need to adjust --action_output_keys."
        )

    timed_actions = []
    for t in range(min(chunk_len, config.actions_per_chunk)):
        action_parts = []
        for key in ordered_keys:
            action_t = ws_action[key][t]
            if isinstance(action_t, torch.Tensor):
                action_t = action_t.cpu().numpy()
            action_parts.append(np.atleast_1d(action_t))

        action_vector = torch.from_numpy(np.concatenate(action_parts)).float()
        timed_actions.append(
            TimedAction(
                timestamp=time.time(),
                timestep=start_timestep + t,
                action=action_vector,
            )
        )

    return timed_actions


def create_transport(config, robot, logger) -> PolicyTransport:
    """按 config.transport 分派创建推理传输。"""
    if config.transport == "grpc":
        logger.info(f"Initializing client to connect to policy server at {config.server_address}")
        return GrpcTransport(config, robot, logger)
    if config.transport == "websocket":
        return WebsocketTransport(config, robot, logger)
    raise ValueError(f"Unknown transport '{config.transport}'. Available: grpc, websocket")
