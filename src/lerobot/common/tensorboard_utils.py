"""lerobot-train 的 TensorBoard 记录（Xbotics）：训练时把曲线写在本地，边训边看，不要账号也不要联网。

记的量和名字与 W&B 那一路相同（``train/loss``、``train/grad_norm``、``train/lr``、``eval/eval_loss``、
``eval/pc_success`` ……），写进 ``<output_dir>/tensorboard`` 下的事件文件；训练开始时日志里打出查看命令
``tensorboard --logdir <output_dir>/tensorboard``。默认开，``--tensorboard.enable=false`` 关掉。

做法照上游 PR #154（没合并）：用 torch 自带的 ``SummaryWriter``。另两处：

- 每个记录窗口写完就 flush：``SummaryWriter`` 默认两分钟才落一次盘，开着的 TensorBoard 要等那么久才出新点。
- 续训时带 ``purge_step``：中断前写在检查点之后的那段点作废（检查点那一步本身保留），曲线从续训的下一步接上，不出现重叠的尾巴。
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
    """取嵌套字典里的数值叶子，键拼成 ``a/b/c``；字符串、列表、布尔值不记。"""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_scalars(v, key))
        elif isinstance(v, numbers.Real) and not isinstance(v, bool):
            out[key] = float(v)
    return out


class TensorBoardLogger:
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
