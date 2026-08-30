from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES


@PreTrainedConfig.register_subclass("openvla")
@dataclass
class OpenVLAConfig(PreTrainedConfig):
    """Configuration for the OpenVLA LeRobot policy wrapper."""

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    pretrained_path: Path | str | None = "openvla/openvla-7b"
    vlm_model_name: str | None = None
    image_feature_key: str = f"{OBS_IMAGES}.image"
    task_key: str = "task"
    default_task: str = ""
    prompt_template: str = "In: What action should the robot take to {task}?\nOut:"
    unnorm_key: str = "libero_10"
    center_crop: bool = True
    # LIBERO eval already rotates camera frames in LiberoProcessorStep.
    rotate_image_180: bool = False
    # LIBERO maps the gripper action to {-1,+1} (binarized) for env.step. True for LIBERO eval;
    # real-robot deploys with a continuous gripper set False.
    apply_libero_gripper: bool = True
    action_dim: int = 7
    action_slice: str | None = None
    # Relative actions: tokenize (action - state) deltas instead of absolute joint targets, so the
    # 256 action-token bins resolve the small per-step motion (fixes the absolute no-op). Mirrors pi0.
    use_relative_actions: bool = False
    # Joints kept absolute (not converted to relative). The gripper is a position, not a delta.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Action dim names (e.g. joint1.pos..gripper.pos); used to build the relative mask.
    action_feature_names: list[str] | None = None
    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    action_stats_key: str = "exp014_multiobject_balanced_abs"
    action_stats: dict[str, list[float]] | None = None
    action_token_ignore_index: int = -100
    tokenizer_max_length: int | None = None

    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_dim != 7:
            raise ValueError("OpenVLA Piper action-token training is configured for 7D actions.")
        if not self.pretrained_path and not self.vlm_model_name:
            raise ValueError("OpenVLAConfig requires `pretrained_path` or `vlm_model_name`.")
        if not self.output_features:
            from lerobot.configs.types import PolicyFeature

            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))}

    def validate_features(self) -> None:
        if self.image_features and self.image_feature_key not in self.input_features:
            available = ", ".join(sorted(self.image_features))
            raise ValueError(
                f"OpenVLA expects image feature '{self.image_feature_key}', available visual keys: {available}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None
