#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import register, registry as gym_registry
from gymnasium.utils.env_checker import check_env

import lerobot
from lerobot.configs.types import PolicyFeature
from lerobot.envs.configs import EnvConfig, So101SimEnv
from lerobot.envs.factory import make_env, make_env_config
from lerobot.envs.utils import (
    _normalize_hub_result,
    _parse_hub_url,
    preprocess_observation,
)
from tests.utils import require_env

OBS_TYPES = ["state", "pixels", "pixels_agent_pos"]


@pytest.mark.parametrize("obs_type", OBS_TYPES)
@pytest.mark.parametrize("env_name, env_task", lerobot.env_task_pairs)
@require_env
def test_env(env_name, env_task, obs_type):
    if env_name == "aloha" and obs_type == "state":
        pytest.skip("`state` observations not available for aloha")

    package_name = f"gym_{env_name}"
    importlib.import_module(package_name)
    env = gym.make(f"{package_name}/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_name", lerobot.available_envs)
@require_env
def test_factory(env_name):
    cfg = make_env_config(env_name)
    envs = make_env(cfg, n_envs=1)
    suite_name = next(iter(envs))
    task_id = next(iter(envs[suite_name]))
    env = envs[suite_name][task_id]
    obs, _ = env.reset()
    obs = preprocess_observation(obs)

    # test image keys are float32 in range [0,1]
    for key in obs:
        if "image" not in key:
            continue
        img = obs[key]
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0

    env.close()


def test_factory_custom_gym_id():
    gym_id = "dummy_gym_pkg/DummyTask-v0"
    if gym_id in gym_registry:
        pytest.skip(f"Environment ID {gym_id} is already registered")

    @EnvConfig.register_subclass("dummy")
    @dataclass
    class DummyEnv(EnvConfig):
        task: str = "DummyTask-v0"
        fps: int = 10
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def package_name(self) -> str:
            return "dummy_gym_pkg"

        @property
        def gym_id(self) -> str:
            return gym_id

        @property
        def gym_kwargs(self) -> dict:
            return {}

    try:
        register(id=gym_id, entry_point="gymnasium.envs.classic_control:CartPoleEnv")

        cfg = DummyEnv()
        envs_dict = make_env(cfg, n_envs=1)
        dummy_envs = envs_dict["dummy"]
        assert len(dummy_envs) == 1
        env = next(iter(dummy_envs.values()))
        assert env is not None and isinstance(env, gym.vector.VectorEnv)
        env.close()

    finally:
        if gym_id in gym_registry:
            del gym_registry[gym_id]


# Hub environment loading tests


def test_make_env_hub_url_parsing():
    """Test URL parsing for hub environment references."""
    # simple repo_id
    repo_id, revision, file_path = _parse_hub_url("user/repo")
    assert repo_id == "user/repo"
    assert revision is None
    assert file_path == "env.py"

    # repo with revision
    repo_id, revision, file_path = _parse_hub_url("user/repo@main")
    assert repo_id == "user/repo"
    assert revision == "main"
    assert file_path == "env.py"

    # repo with custom file path
    repo_id, revision, file_path = _parse_hub_url("user/repo:custom_env.py")
    assert repo_id == "user/repo"
    assert revision is None
    assert file_path == "custom_env.py"

    # repo with revision and custom file path
    repo_id, revision, file_path = _parse_hub_url("user/repo@v1.0:envs/my_env.py")
    assert repo_id == "user/repo"
    assert revision == "v1.0"
    assert file_path == "envs/my_env.py"

    # repo with commit hash
    repo_id, revision, file_path = _parse_hub_url("org/repo@abc123def456")
    assert repo_id == "org/repo"
    assert revision == "abc123def456"
    assert file_path == "env.py"


def test_normalize_hub_result():
    """Test normalization of different return types from hub make_env."""
    # test with VectorEnv (most common case)
    mock_vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    result = _normalize_hub_result(mock_vec_env)
    assert isinstance(result, dict)
    assert len(result) == 1
    suite_name = next(iter(result))
    assert 0 in result[suite_name]
    assert isinstance(result[suite_name][0], gym.vector.VectorEnv)
    mock_vec_env.close()

    # test with single Env
    mock_env = gym.make("CartPole-v1")
    result = _normalize_hub_result(mock_env)
    assert isinstance(result, dict)
    suite_name = next(iter(result))
    assert 0 in result[suite_name]
    assert isinstance(result[suite_name][0], gym.vector.VectorEnv)
    result[suite_name][0].close()

    # test with dict (already normalized)
    mock_vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    input_dict = {"my_suite": {0: mock_vec_env}}
    result = _normalize_hub_result(input_dict)
    assert result == input_dict
    assert "my_suite" in result
    assert 0 in result["my_suite"]
    mock_vec_env.close()

    # test with invalid type
    with pytest.raises(ValueError, match="Hub `make_env` must return"):
        _normalize_hub_result("invalid_type")


def test_make_env_from_hub_requires_trust_remote_code():
    """Test that loading from hub requires explicit trust_remote_code=True."""
    hub_id = "lerobot/cartpole-env"

    # Should raise RuntimeError when trust_remote_code=False (default)
    with pytest.raises(RuntimeError, match="Refusing to execute remote code"):
        make_env(hub_id, trust_remote_code=False)

    # Should also raise when not specified (defaults to False)
    with pytest.raises(RuntimeError, match="Refusing to execute remote code"):
        make_env(hub_id)


@pytest.mark.parametrize(
    "hub_id",
    [
        "lerobot/cartpole-env",
        "lerobot/cartpole-env@main",
        "lerobot/cartpole-env:env.py",
    ],
)
def test_make_env_from_hub_with_trust(hub_id):
    """Test loading environment from Hugging Face Hub with trust_remote_code=True."""
    # load environment from hub
    envs_dict = make_env(hub_id, n_envs=2, trust_remote_code=True)

    # verify structure
    assert isinstance(envs_dict, dict)
    assert len(envs_dict) >= 1

    # get the first suite and task
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]

    # verify it's a vector environment
    assert isinstance(env, gym.vector.VectorEnv)
    assert env.num_envs == 2

    # test basic environment interaction
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(obs, (dict, np.ndarray))

    # take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, np.ndarray)
    assert len(reward) == 2

    # clean up
    env.close()


def test_make_env_from_hub_async():
    """Test loading hub environment with async vector environments."""
    hub_id = "lerobot/cartpole-env"

    # load with async envs
    envs_dict = make_env(hub_id, n_envs=2, use_async_envs=True, trust_remote_code=True)

    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]

    # verify it's an async vector environment
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == 2

    # test basic interaction
    obs, info = env.reset()
    assert obs is not None

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert len(reward) == 2

    # clean up
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# so101_sim 评测口的默认值契约（我们自有的 EnvConfig 子类，不涉上游代码）
#
# 这几项配错都不报错、只表现为一个会被误读成「策略没学会」的低成功率，所以默认值必须
# 与「已交付的 SO-101 数据集是怎么产生的」一致，且要有测试钉住 —— 曾经默认是 20 fps /
# 128×128 / control_mode=None / 无口径换算，四项全与数据不符。
#
# 直接构造 So101SimEnv 而不经过 make_env_config：后者是**上游代码**，不动它。
# 本组只读配置、不建仿真，因此在没有 GPU 的机器上也能跑。
# ─────────────────────────────────────────────────────────────────────────────


def test_so101_sim_defaults_match_real_robot():
    """默认值对齐真机口径：臂关节度 + 夹爪百分比、绝对关节角、标定分辨率、装得下最长轨迹。"""
    cfg = So101SimEnv()

    assert cfg.unit_convention == "real", "默认必须是真机口径（臂关节度、夹爪行程百分比）"
    assert cfg.control_mode == "pd_joint_pos", "数据集录的是绝对关节角"
    assert (cfg.observation_width, cfg.observation_height) == (640, 480), (
        "相机的竖直视野角是在 640×480 下标定的，改宽高比会改水平视野"
    )
    assert cfg.fps == 30, "仿真 control_freq 与数据帧率都是 30"
    assert cfg.episode_length >= 444, "已交付数据最长 444 帧，短了会静默截断"


def test_so101_sim_gym_kwargs_carries_unit_convention():
    """口径必须下发给仿真侧 —— 不在 gym_kwargs 里就等于这个参数不存在。"""
    cfg = So101SimEnv(unit_convention="maniskill")

    assert cfg.gym_kwargs["unit_convention"] == "maniskill"
    for key in ("control_mode", "observation_width", "observation_height", "episode_length"):
        assert key in cfg.gym_kwargs, f"{key} 没有下发给仿真侧"


def test_so101_sim_rejects_unknown_unit_convention():
    """非法口径要当场报错，而不是留到评测出一个低成功率再让人反推。"""
    with pytest.raises(ValueError, match="unit_convention"):
        So101SimEnv(unit_convention="deg")


def test_so101_sim_visual_feature_shape_follows_resolution():
    """观测空间的形状要跟着分辨率走 —— 声明与实际分岔时没有任何一步会报错。"""
    cfg = So101SimEnv(observation_width=320, observation_height=240)

    for camera in ("top", "wrist"):
        assert cfg.features[f"pixels/{camera}"].shape == (240, 320, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 追加注册的契约：新类型能造出来，且上游已有行为**逐字未变**
#
# 规则是「加注册可以，改行为不行」。所以这里既要证明 so101_sim 能造，
# 也要证明上游那三个分支与 else 的报文没被动过 —— 后者是本组测试真正的价值：
# 一旦有人把 if/elif 改写成注册表查表，报文与可接受的类型集就都变了，
# 而那种改写在功能上「看起来更好」，正是最容易被放行的一类越界。
# ─────────────────────────────────────────────────────────────────────────────


def test_make_env_config_builds_so101_sim():
    """我们追加的注册能造出对应配置。"""
    cfg = make_env_config("so101_sim")

    assert cfg.type == "so101_sim"
    assert isinstance(cfg, So101SimEnv)


@pytest.mark.parametrize("env_type", ["aloha", "pusht", "libero"])
def test_make_env_config_upstream_branches_unchanged(env_type):
    """上游原有的三个分支照旧可用。"""
    assert make_env_config(env_type).type == env_type


def test_make_env_config_unknown_type_keeps_upstream_message():
    """未注册类型的报文必须与上游逐字相同 —— 改报文也是改行为。"""
    with pytest.raises(ValueError, match=r"^Policy type 'nope' is not available\.$"):
        make_env_config("nope")
