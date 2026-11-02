"""lerobot-train 的 TensorBoard 记录（Xbotics）：写进去的量能被 TensorBoard 原样读回，续训时曲线从续训的那一步接上。"""

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from lerobot.common.tensorboard_utils import TensorBoardLogger, flatten_scalars, resolve_log_dir  # noqa: E402
from lerobot.configs.default import TensorBoardConfig  # noqa: E402


def make_cfg(output_dir, log_dir=None):
    return SimpleNamespace(
        output_dir=output_dir,
        tensorboard=TensorBoardConfig(log_dir=log_dir),
        to_dict=lambda: {"policy": {"type": "act"}, "steps": 10},
    )


def read_scalars(log_dir) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def test_default_log_dir_is_under_output_dir(tmp_path):
    assert resolve_log_dir(make_cfg(tmp_path)) == tmp_path / "tensorboard"
    assert resolve_log_dir(make_cfg(tmp_path, tmp_path / "tb")) == tmp_path / "tb"


def test_flatten_keeps_only_numeric_leaves():
    d = {"loss": 0.5, "steps": 3, "ok": True, "name": "x", "paths": [1, 2], "suite": {"pc_success": 40.0}}
    assert flatten_scalars(d) == {"loss": 0.5, "steps": 3.0, "suite/pc_success": 40.0}


def test_train_and_eval_scalars_read_back(tmp_path):
    logger = TensorBoardLogger(make_cfg(tmp_path))
    for step in (1, 2, 3):
        logger.log_dict({"loss": 1.0 / step, "lr": 1e-4, "name": "ignored"}, step)
    logger.log_dict({"eval_loss": 0.25, "so101": {"pc_success": 60.0}}, 3, mode="eval")
    logger.close()

    scalars = read_scalars(tmp_path / "tensorboard")
    assert [s for s, _ in scalars["train/loss"]] == [1, 2, 3]
    assert scalars["train/loss"][2][1] == pytest.approx(1 / 3)
    assert scalars["eval/eval_loss"] == [(3, 0.25)]
    assert scalars["eval/so101/pc_success"] == [(3, 60.0)]
    assert "train/name" not in scalars


def test_resume_drops_points_past_the_checkpoint(tmp_path):
    first = TensorBoardLogger(make_cfg(tmp_path))
    for step in range(1, 11):  # 跑到第 10 步中断，最后一个检查点在第 5 步
        first.log_dict({"loss": 1.0}, step)
    first.close()

    resumed = TensorBoardLogger(make_cfg(tmp_path), start_step=5)
    for step in range(5, 8):
        resumed.log_dict({"loss": 2.0}, step)
    resumed.close()

    loss = read_scalars(tmp_path / "tensorboard")["train/loss"]
    assert [s for s, _ in loss] == [1, 2, 3, 4, 5, 6, 7]
    assert [v for s, v in loss if s >= 5] == [2.0, 2.0, 2.0]


def test_mode_must_be_train_or_eval(tmp_path):
    logger = TensorBoardLogger(make_cfg(tmp_path))
    with pytest.raises(ValueError):
        logger.log_dict({"loss": 1.0}, 1, mode="test")
    logger.close()
