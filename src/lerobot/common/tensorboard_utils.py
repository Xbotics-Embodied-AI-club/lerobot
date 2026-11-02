"""lerobot-train 的 TensorBoard 记录（Xbotics）。

训练时把曲线写在本地，训练过程中即可查看，无需账号与网络。记录的量和名字与 W&B 那一路相同（``train/loss``、``train/grad_norm``、``train/lr``、``eval/eval_loss``、
``eval/pc_success`` ……），写入 ``<output_dir>/tensorboard`` 下的事件文件；训练开始时日志中给出查看命令
``tensorboard --logdir <output_dir>/tensorboard``。默认开启，``--tensorboard.enable=false`` 关闭。

做法参照上游 PR #154（未合并）：使用 torch 自带的 ``SummaryWriter``。另有两处处理：

- 每个记录窗口写完即 flush：``SummaryWriter`` 默认两分钟才写盘一次，正在查看的 TensorBoard 要等待同样长的时间才显示新数据点。
- 续训时设置 ``purge_step``：中断前写在检查点之后的数据点作废（检查点所在步保留），曲线从续训的下一步接续，不出现重叠。
"""

import json
import logging
import numbers
from pathlib import Path

from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig


def resolve_log_dir(cfg: TrainPipelineConfig) -> Path:
    return Path(cfg.tensorboard.log_dir) if cfg.tensorboard.log_dir else Path(cfg.output_dir) / "tensorboard"


def flatten_scalars(d: dict, prefix: str = "") -> dict[str, float]:
    """取出嵌套字典中的数值叶子，键拼接为 ``a/b/c``；字符串、列表与布尔值不记录。"""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_scalars(v, key))
        elif isinstance(v, numbers.Real) and not isinstance(v, bool):
            out[key] = float(v)
    return out


class TensorBoardLogger:
    """一次训练的 TensorBoard 事件文件写入器，只在主进程上创建。"""

    def __init__(self, cfg: TrainPipelineConfig, start_step: int = 0):
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = resolve_log_dir(cfg)
        self._writer = SummaryWriter(
            log_dir=str(self.log_dir), purge_step=start_step + 1 if start_step else None
        )
        if not start_step:
            config = json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=str)
            self._writer.add_text("config", f"```json\n{config}\n```", 0)
        logging.info(colored("Logs will be written to TensorBoard.", "blue", attrs=["bold"]))
        command = f"tensorboard --logdir {self.log_dir}"
        logging.info(f"Watch this run --> {colored(command, 'yellow', attrs=['bold'])}")

    def log_dict(self, d: dict, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        for key, value in flatten_scalars(d).items():
            self._writer.add_scalar(f"{mode}/{key}", value, step)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
