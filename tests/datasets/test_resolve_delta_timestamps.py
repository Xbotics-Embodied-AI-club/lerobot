"""resolve_delta_timestamps (Xbotics: per-modality indices backported from upstream main for lingbot_vla_v2).

LingBot-VLA 2.0's distillation needs the current and a future frame of every camera
(``image_observation_delta_indices``); without per-modality indices the dataset only returns the
current frame and the first training step fails. Policies that do not define the per-modality
properties keep the shared ``observation_delta_indices``, exactly as before.
"""

from types import SimpleNamespace

import pytest

from lerobot.datasets.factory import resolve_delta_timestamps

FPS = 10
FEATURES = {
    "observation.images.top": {},
    "observation.images.wrist": {},
    "observation.state": {},
    "observation.environment_state": {},
    "action": {},
    "timestamp": {},
}


def _cfg(**indices):
    base = {"observation_delta_indices": None, "action_delta_indices": None, "reward_delta_indices": None}
    return SimpleNamespace(**{**base, **indices})


def _meta(features=FEATURES):
    return SimpleNamespace(features=features, fps=FPS)


def test_shared_observation_indices_are_unchanged_for_policies_without_modality_indices():
    cfg = _cfg(observation_delta_indices=[-1, 0], action_delta_indices=[0, 1, 2])
    assert resolve_delta_timestamps(cfg, _meta()) == {
        "observation.images.top": [-0.1, 0.0],
        "observation.images.wrist": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "observation.environment_state": [-0.1, 0.0],
        "action": [0.0, 0.1, 0.2],
    }


def test_image_indices_apply_to_cameras_only():
    cfg = _cfg(action_delta_indices=[0, 1], image_observation_delta_indices=[0, 49])
    assert resolve_delta_timestamps(cfg, _meta()) == {
        "observation.images.top": [0.0, 4.9],
        "observation.images.wrist": [0.0, 4.9],
        "action": [0.0, 0.1],
    }


def test_image_history_without_any_image_key_fails_loudly():
    cfg = _cfg(image_observation_delta_indices=[0, 49])
    with pytest.raises(ValueError, match="history frames per camera"):
        resolve_delta_timestamps(cfg, _meta({"observation.state": {}, "action": {}}))
