from __future__ import annotations

from typing import Any

from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)

from .configuration_openvla import OpenVLAConfig


def make_openvla_pre_post_processors(
    config: OpenVLAConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Build OpenVLA pre/post processors.

    OpenVLA does its own (un)normalization via action-token q01/q99 inside the policy, so the
    pipelines are otherwise identity. When ``config.use_relative_actions`` is set, the
    preprocessor converts ``action -> action - state`` (joints relative, gripper kept absolute)
    so the model tokenizes deltas, and the postprocessor adds the cached state back to the
    predicted delta for execution. Disabled => both steps are no-ops (state still cached).
    """
    del dataset_stats

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    return (
        PolicyProcessorPipeline(
            steps=[relative_step],
            name="OpenVLAInputRelativeProcessor",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline(
            steps=[AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step)],
            name="OpenVLAActionAbsoluteProcessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
