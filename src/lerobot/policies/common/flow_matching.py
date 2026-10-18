#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Flow-matching sampling primitives shared across policies.

Canonical versions of the beta-distributed timestep sampler and the forward-Euler
denoising loop (with its real-time-chunking hook) that the openpi-derived policies
(pi0, pi05, smolvla, eo1) historically each carried a copy of. All functions are
stateless; adopting them does not affect checkpoints.
"""

import enum
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor


# Xbotics：以下两项照搬上游 main（#4075、#4631），LingBot-VLA 2.0（lingbot_vla_v2）训练时用；
# 0.6 线的其他策略仍走各自原来的写法，没有迁过来。
class FlowConvention(str, enum.Enum):  # noqa: UP042 —— 照搬上游逐字；StrEnum 的 str() 与之不同
    """Which end of the ``[0, 1]`` schedule holds the noise.

    Both conventions describe the same family of probability paths, they only disagree on
    the direction of time (and therefore on the sign of the velocity field). A checkpoint is
    tied to the convention it was trained with, so this is a property of the policy, never a
    tunable.
    """

    NOISE_AT_ONE = "noise_at_one"
    """openpi (pi0, pi05, smolvla, eo1): ``x_1`` is noise, integrate ``t: 1 -> 0``.

    ``dt = -1/num_steps``, ``time = 1.0 + step * dt``, and the clean sample is
    ``x_0 = x_t - time * v_t``.
    """

    NOISE_AT_ZERO = "noise_at_zero"
    """groot / evo1 / wall_x: ``x_0`` is noise, integrate ``t: 0 -> 1``.

    ``dt = +1/num_steps``, ``time = step * dt``, and the clean sample is
    ``x_1 = x_t + (1 - time) * v_t``.
    """


def make_flow_matching_inputs(
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    *,
    convention: FlowConvention,
    prefix_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the noised actions, velocity target and model timesteps for a training step.

    The interpolation is linear between the two endpoints of the probability path, and
    ``convention`` (see :class:`FlowConvention`) decides which endpoint the noise sits at,
    and therefore both the interpolation and the sign of the velocity target:

    * ``NOISE_AT_ONE``: ``x_t = t * noise + (1 - t) * actions`` and ``v = noise - actions``.
    * ``NOISE_AT_ZERO``: ``x_t = (1 - t) * noise + t * actions`` and ``v = actions - noise``.

    This is the training-time half of the contract that :func:`euler_integrate` implements at
    inference time. Nothing here checks that the two agree: a policy that passes different
    conventions to each learns a velocity pointing the wrong way along its own schedule.

    The returned target is computed in the dtype of ``actions`` and ``noise``. A caller that
    regresses in a different precision than it interpolates in (evo1) has to build its own.

    Args:
        actions: ``(batch, horizon, action_dim)`` clean action chunk.
        noise: Noise sample broadcastable to ``actions``.
        time: ``(batch,)`` timesteps.
        convention: Which end of the schedule holds the noise.
        prefix_mask: Optional ``(batch, horizon)`` boolean mask marking clean action-prefix
            positions (real-time chunking). Masked positions are given the model time at the
            clean end of the schedule, which makes the interpolation return the clean action
            there, so they are never noised. The inference-time counterpart is
            ``euler_integrate``'s ``hard_prefix`` / ``hard_prefix_mask``.

    Returns:
        ``(x_t, velocity_target, model_time)``: the noised actions fed to the network, the
        regression target for the predicted velocity, and the timesteps fed to the network.
        ``model_time`` is ``(batch,)`` without a prefix and ``(batch, horizon)`` with one.
    """
    if not isinstance(convention, FlowConvention):
        raise TypeError(f"convention must be a FlowConvention, got {convention!r}")
    if time.shape != actions.shape[:1]:
        raise ValueError(f"time must have shape (batch,), got {tuple(time.shape)}")
    if prefix_mask is not None and prefix_mask.shape != actions.shape[:2]:
        raise ValueError(f"prefix_mask must have shape (batch, horizon), got {tuple(prefix_mask.shape)}")
    noise_at_one = convention is FlowConvention.NOISE_AT_ONE

    if prefix_mask is None:
        model_time = time
        expanded_time = time[:, None, None]
    else:
        # Clean tokens sit at the far end of the schedule from the noise.
        clean_time = 0.0 if noise_at_one else 1.0
        model_time = time[:, None].expand_as(prefix_mask)
        model_time = torch.where(prefix_mask, torch.full_like(model_time, clean_time), model_time)
        expanded_time = model_time.unsqueeze(-1)

    if noise_at_one:
        x_t = expanded_time * noise + (1 - expanded_time) * actions
        velocity_target = noise - actions
    else:
        x_t = (1 - expanded_time) * noise + expanded_time * actions
        velocity_target = actions - noise

    return x_t, velocity_target, model_time


def sample_beta(alpha: float, beta: float, bsize: int, device) -> Tensor:  # see openpi (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def sample_noise(shape, device) -> Tensor:
    """Standard-normal float32 noise, the flow-matching x_1 sample."""
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )


def sample_time_beta(bsize: int, device, *, alpha: float, beta: float, scale: float, offset: float) -> Tensor:
    """Beta-distributed flow-matching timesteps: ``Beta(alpha, beta) * scale + offset`` (openpi convention)."""
    time_beta = sample_beta(alpha, beta, bsize, device)
    time = time_beta * scale + offset
    return time.to(dtype=torch.float32, device=device)


def euler_integrate(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    noise: Tensor,
    num_steps: int,
    *,
    rtc_processor: "RTCProcessor | None" = None,
    rtc_enabled: bool = False,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
) -> Tensor:
    """Forward-Euler integration of a velocity field from t=1 (noise) to t=0 (actions).

    This is the openpi sampling loop: ``dt = -1/num_steps``, ``time = 1.0 + step*dt``,
    ``x_t <- x_t + dt * v_t``, with the optional real-time-chunking (RTC) guidance hook
    wrapping the velocity computation and debug tracking after each step.

    Args:
        denoise_fn: Computes the velocity ``v_t`` from ``(x_t, time_tensor)`` where
            ``time_tensor`` is a float32 tensor of shape ``(batch_size,)``. The returned
            velocity must have the same shape and dtype as ``x_t``.
        noise: Initial sample ``x_1`` of shape ``(batch_size, ...)``.
        num_steps: Number of Euler steps.
        rtc_processor: Optional RTC processor. Debug tracking fires whenever it is set and
            has debugging enabled, even if RTC guidance itself is disabled (this mirrors
            the historical per-policy loops).
        rtc_enabled: Whether to route the velocity computation through
            ``rtc_processor.denoise_step`` (requires ``rtc_processor``).
        inference_delay: RTC guidance parameter, forwarded verbatim.
        prev_chunk_left_over: RTC guidance parameter, forwarded verbatim.
        execution_horizon: RTC guidance parameter, forwarded verbatim.
    """
    bsize = noise.shape[0]
    device = noise.device

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return denoise_fn(input_x_t, current_timestep)

        if rtc_enabled:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=execution_horizon,
            )
        else:
            v_t = denoise_step_partial_call(x_t)

        x_t = x_t + dt * v_t

        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

    return x_t
