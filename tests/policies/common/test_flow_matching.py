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

"""Behavior-pinning tests for the shared flow-matching sampling primitives.

``euler_integrate`` is compared against a verbatim copy of the historical pi0/pi05/
smolvla sampling loop (including its RTC hook semantics): any divergence from that
reference is a behavior change for released checkpoints.
"""

import pytest
import torch

from lerobot.policies.common.flow_matching import (
    FlowConvention,
    euler_integrate,
    make_flow_matching_inputs,
    sample_beta,
    sample_noise,
    sample_time_beta,
)


def test_sample_beta_range_dtype_and_reproducibility():
    torch.manual_seed(0)
    s1 = sample_beta(1.5, 1.0, 4096, "cpu")
    torch.manual_seed(0)
    s2 = sample_beta(1.5, 1.0, 4096, "cpu")
    assert torch.equal(s1, s2)
    assert s1.shape == (4096,) and s1.dtype == torch.float32
    assert s1.min() >= 0.0 and s1.max() <= 1.0
    # Beta(1.5, 1.0) mean is 1.5/2.5 = 0.6.
    assert abs(s1.mean().item() - 0.6) < 0.02


def test_sample_time_beta_openpi_convention():
    torch.manual_seed(1)
    time = sample_time_beta(4096, "cpu", alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    assert time.dtype == torch.float32
    assert time.min() >= 0.001 and time.max() <= 1.0
    # Exact composition: Beta sample * scale + offset, same RNG stream.
    torch.manual_seed(1)
    expected = sample_beta(1.5, 1.0, 4096, "cpu") * 0.999 + 0.001
    torch.testing.assert_close(time, expected, rtol=0, atol=0)


def test_sample_noise_seeded():
    torch.manual_seed(2)
    n1 = sample_noise((2, 8, 4), "cpu")
    torch.manual_seed(2)
    n2 = sample_noise((2, 8, 4), "cpu")
    assert torch.equal(n1, n2)
    assert n1.dtype == torch.float32 and n1.shape == (2, 8, 4)


def test_euler_integrate_constant_velocity_is_exact():
    # With v_t == c constant, x_0 = x_1 + sum(dt * c) = x_1 - c exactly (num_steps * dt = -1).
    noise = torch.randn(3, 5, 2)
    c = torch.randn(3, 5, 2)
    out = euler_integrate(lambda x_t, time: c, noise, num_steps=10)
    torch.testing.assert_close(out, noise - c, rtol=0, atol=1e-6)


def _reference_pi0_loop(denoise_fn, noise, num_steps, rtc_enabled, rtc_processor, kw):
    """Verbatim structure of the historical pi0/pi05/smolvla sample_actions loop."""
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
                prev_chunk_left_over=kw.get("prev_chunk_left_over"),
                inference_delay=kw.get("inference_delay"),
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=kw.get("execution_horizon"),
            )
        else:
            v_t = denoise_step_partial_call(x_t)
        x_t = x_t + dt * v_t
        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
    return x_t


class _StubRTCProcessor:
    def __init__(self, debug_enabled: bool):
        self._debug = debug_enabled
        self.tracked = []
        self.guidance_calls = []

    def is_debug_enabled(self):
        return self._debug

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon,
    ):
        self.guidance_calls.append(
            {
                "time": time,
                "inference_delay": inference_delay,
                "execution_horizon": execution_horizon,
                "x_t": x_t.clone(),
            }
        )
        return original_denoise_step_partial(x_t) * 0.5

    def track(self, time, x_t, v_t):
        self.tracked.append({"time": time, "x_t": x_t.clone(), "v_t": v_t.clone()})


def _make_denoise_fn():
    weight = torch.randn(4, 4) * 0.1

    def denoise_fn(x_t, time_tensor):
        return x_t @ weight + time_tensor[:, None, None]

    return denoise_fn


def test_euler_integrate_matches_historical_loop():
    torch.manual_seed(3)
    denoise_fn = _make_denoise_fn()
    noise = torch.randn(2, 6, 4)
    ref = _reference_pi0_loop(denoise_fn, noise, 10, rtc_enabled=False, rtc_processor=None, kw={})
    out = euler_integrate(denoise_fn, noise, 10)
    assert torch.equal(out, ref)


def test_euler_integrate_rtc_guidance_and_kwarg_forwarding():
    torch.manual_seed(4)
    denoise_fn = _make_denoise_fn()
    noise = torch.randn(2, 6, 4)
    leftover = torch.randn(2, 6, 4)
    kw = {"inference_delay": 3, "prev_chunk_left_over": leftover, "execution_horizon": 25}

    ref_proc, new_proc = _StubRTCProcessor(False), _StubRTCProcessor(False)
    ref = _reference_pi0_loop(denoise_fn, noise, 6, rtc_enabled=True, rtc_processor=ref_proc, kw=kw)
    out = euler_integrate(
        denoise_fn,
        noise,
        6,
        rtc_processor=new_proc,
        rtc_enabled=True,
        inference_delay=3,
        prev_chunk_left_over=leftover,
        execution_horizon=25,
    )
    assert torch.equal(out, ref)
    assert len(new_proc.guidance_calls) == 6
    for ref_call, new_call in zip(ref_proc.guidance_calls, new_proc.guidance_calls, strict=True):
        assert ref_call["time"] == new_call["time"]
        assert new_call["inference_delay"] == 3 and new_call["execution_horizon"] == 25
        # Guidance sees the PRE-update x_t.
        assert torch.equal(ref_call["x_t"], new_call["x_t"])


def test_euler_integrate_debug_tracking_fires_even_when_rtc_disabled():
    # Historical behavior: track() fires whenever the processor exists and has debugging
    # enabled, independent of whether RTC guidance is active.
    torch.manual_seed(5)
    denoise_fn = _make_denoise_fn()
    noise = torch.randn(2, 6, 4)
    proc = _StubRTCProcessor(True)
    out = euler_integrate(denoise_fn, noise, 4, rtc_processor=proc, rtc_enabled=False)
    assert len(proc.guidance_calls) == 0
    assert len(proc.tracked) == 4
    # track() receives the POST-update x_t; the last one is the returned sample.
    assert torch.equal(proc.tracked[-1]["x_t"], out)


# ---------------------------------------------------------------- make_flow_matching_inputs
# Xbotics: ported from upstream main's tests (the helper came with LingBot-VLA 2.0). Left out:
# test_make_inputs_target_integrates_back_to_the_actions, which needs upstream's euler_integrate(convention=...).


def _historical_noise_at_one_inputs(actions, noise, time):
    """pi0 / smolvla / eo1 training-time noising, pre-adoption."""
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    return x_t, u_t


def _historical_pi05_inputs(actions, noise, time, prefix_mask):
    """pi05's `_build_flow_matching_inputs` plus its target, pre-adoption."""
    if prefix_mask is None:
        model_time = time
        expanded_time = time[:, None, None]
    else:
        model_time = time[:, None].expand_as(prefix_mask)
        model_time = torch.where(prefix_mask, torch.zeros_like(model_time), model_time)
        expanded_time = model_time.unsqueeze(-1)
    x_t = expanded_time * noise + (1 - expanded_time) * actions
    return x_t, noise - actions, model_time


def _historical_groot_inputs(actions, noise, time, num_timestep_buckets):
    """groot_n1_7 training-time noising, pre-adoption."""
    t = time[:, None, None]
    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise
    t_discretized = (t[:, 0, 0] * num_timestep_buckets).long()
    return noisy_trajectory, velocity, t_discretized


def _historical_wall_x_inputs(action_chunk, noise, time):
    """wall_x ActionHead training-time noising, pre-adoption (float32 throughout)."""
    t = time.unsqueeze(-1).unsqueeze(-1)
    action_chunk_f32 = action_chunk.to(torch.float32)
    noisy_action = (1 - t) * noise + t * action_chunk_f32
    flow = action_chunk_f32 - noise
    return noisy_action, flow


def _historical_evo1_inputs(actions_gt_seq, noise_seq, time, batch_size):
    """evo1 flow-matching head training-time noising, pre-adoption."""
    t_broadcast = time.view(batch_size, 1, 1)
    return (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq


def test_make_inputs_noise_at_one_matches_historical_pi_family():
    torch.manual_seed(20)
    actions, noise = torch.randn(3, 6, 4), torch.randn(3, 6, 4)
    time = torch.rand(3)

    x_t, u_t, model_time = make_flow_matching_inputs(
        actions, noise, time, convention=FlowConvention.NOISE_AT_ONE
    )
    ref_x_t, ref_u_t = _historical_noise_at_one_inputs(actions, noise, time)

    assert torch.equal(x_t, ref_x_t)
    assert torch.equal(u_t, ref_u_t)
    # Without a prefix the model sees one scalar timestep per batch element, unchanged.
    assert model_time is time


def test_make_inputs_noise_at_zero_matches_historical_groot():
    torch.manual_seed(21)
    actions, noise = torch.randn(3, 6, 4), torch.randn(3, 6, 4)
    time = torch.rand(3)
    buckets = 1000

    x_t, velocity, model_time = make_flow_matching_inputs(
        actions, noise, time, convention=FlowConvention.NOISE_AT_ZERO
    )
    ref_x_t, ref_velocity, ref_buckets = _historical_groot_inputs(actions, noise, time, buckets)

    assert torch.equal(x_t, ref_x_t)
    assert torch.equal(velocity, ref_velocity)
    # groot reads its timestep buckets off the same scalar time.
    assert torch.equal((model_time * buckets).long(), ref_buckets)


def test_make_inputs_noise_at_zero_matches_historical_wall_x():
    torch.manual_seed(22)
    action_chunk = torch.randn(2, 5, 7, dtype=torch.float32)
    noise = torch.randn_like(action_chunk, dtype=torch.float32)
    time = torch.rand(2)

    x_t, flow, _ = make_flow_matching_inputs(
        action_chunk.to(torch.float32), noise, time, convention=FlowConvention.NOISE_AT_ZERO
    )
    ref_x_t, ref_flow = _historical_wall_x_inputs(action_chunk, noise, time)

    assert x_t.dtype == torch.float32 and flow.dtype == torch.float32
    assert torch.equal(x_t, ref_x_t)
    assert torch.equal(flow, ref_flow)


def test_make_inputs_noise_at_zero_matches_historical_evo1():
    torch.manual_seed(23)
    batch_size, horizon, per_action_dim = 4, 3, 5
    actions_gt_seq = torch.randn(batch_size, horizon, per_action_dim)
    noise_seq = torch.rand_like(actions_gt_seq) * 2 - 1
    time = torch.distributions.Beta(2, 2).sample((batch_size,)).clamp(0.02, 0.98)

    x_t, _, _ = make_flow_matching_inputs(
        actions_gt_seq, noise_seq, time, convention=FlowConvention.NOISE_AT_ZERO
    )
    ref = _historical_evo1_inputs(actions_gt_seq, noise_seq, time, batch_size)

    assert torch.equal(x_t, ref)


def test_make_inputs_target_sign_flips_with_the_convention():
    actions, noise = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    time = torch.rand(2)
    _, backward_target, _ = make_flow_matching_inputs(
        actions, noise, time, convention=FlowConvention.NOISE_AT_ONE
    )
    _, forward_target, _ = make_flow_matching_inputs(
        actions, noise, time, convention=FlowConvention.NOISE_AT_ZERO
    )
    assert torch.equal(backward_target, -forward_target)


@pytest.mark.parametrize("convention", [FlowConvention.NOISE_AT_ONE, FlowConvention.NOISE_AT_ZERO])
def test_make_inputs_endpoints_are_the_noise_and_the_actions(convention):
    actions, noise = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    noise_end = 1.0 if convention is FlowConvention.NOISE_AT_ONE else 0.0

    at_noise, _, _ = make_flow_matching_inputs(
        actions, noise, torch.full((2,), noise_end), convention=convention
    )
    at_actions, _, _ = make_flow_matching_inputs(
        actions, noise, torch.full((2,), 1.0 - noise_end), convention=convention
    )
    torch.testing.assert_close(at_noise, noise, rtol=0, atol=1e-6)
    torch.testing.assert_close(at_actions, actions, rtol=0, atol=1e-6)


def test_make_inputs_prefix_matches_historical_pi05():
    torch.manual_seed(24)
    actions, noise = torch.randn(4, 6, 3), torch.randn(4, 6, 3)
    time = torch.rand(4)
    delays = torch.tensor([0, 2, 6, 3])
    prefix_mask = torch.arange(6).unsqueeze(0) < delays.unsqueeze(1)

    x_t, u_t, model_time = make_flow_matching_inputs(
        actions, noise, time, convention=FlowConvention.NOISE_AT_ONE, prefix_mask=prefix_mask
    )
    ref_x_t, ref_u_t, ref_model_time = _historical_pi05_inputs(actions, noise, time, prefix_mask)

    assert torch.equal(x_t, ref_x_t)
    assert torch.equal(u_t, ref_u_t)
    assert torch.equal(model_time, ref_model_time)
    assert model_time.shape == (4, 6)


@pytest.mark.parametrize(
    ("convention", "clean_time"),
    [(FlowConvention.NOISE_AT_ONE, 0.0), (FlowConvention.NOISE_AT_ZERO, 1.0)],
)
def test_make_inputs_prefix_is_clean_and_gets_the_clean_end_time(convention, clean_time):
    torch.manual_seed(25)
    actions, noise = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
    time = torch.rand(2) * 0.8 + 0.1
    prefix_mask = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])

    x_t, _, model_time = make_flow_matching_inputs(
        actions, noise, time, convention=convention, prefix_mask=prefix_mask
    )

    # Prefix positions carry the clean action, untouched by the noise.
    torch.testing.assert_close(x_t[prefix_mask], actions[prefix_mask], rtol=0, atol=1e-6)
    assert torch.equal(model_time[prefix_mask], torch.full_like(model_time[prefix_mask], clean_time))
    # Non-prefix positions keep their sampled time and are genuinely noised.
    assert torch.equal(model_time[~prefix_mask], time[:, None].expand_as(prefix_mask)[~prefix_mask])
    assert not torch.allclose(x_t[~prefix_mask], actions[~prefix_mask])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_make_inputs_preserves_dtype(dtype):
    actions = torch.randn(2, 4, 3, dtype=dtype)
    noise = torch.randn(2, 4, 3, dtype=dtype)
    x_t, target, _ = make_flow_matching_inputs(
        actions, noise, torch.rand(2, dtype=dtype), convention=FlowConvention.NOISE_AT_ONE
    )
    assert x_t.dtype == dtype and target.dtype == dtype


@pytest.mark.parametrize("convention", ["noise_at_one", "noise_at_zero", "noise_at_on", None])
def test_make_inputs_rejects_a_convention_that_is_not_the_enum(convention):
    # FlowConvention is a str enum, so "noise_at_one" == NOISE_AT_ONE, yet the `is` check
    # would send it down the NOISE_AT_ZERO branch.
    actions = torch.randn(2, 4, 3)
    with pytest.raises(TypeError, match="convention must be a FlowConvention"):
        make_flow_matching_inputs(actions, torch.randn_like(actions), torch.rand(2), convention=convention)


def test_make_inputs_rejects_shapes_that_would_broadcast_silently():
    actions = torch.randn(2, 4, 3)
    noise = torch.randn_like(actions)
    # lawam keeps its time pre-expanded as (B, 1, 1); passed in, x_t would broadcast to (B, 1, B, H, D).
    with pytest.raises(ValueError, match="time must have shape"):
        make_flow_matching_inputs(actions, noise, torch.rand(2, 1, 1), convention=FlowConvention.NOISE_AT_ONE)
    # A (B, H, D) mask is the shape of euler_integrate's hard_prefix_mask, not this one.
    with pytest.raises(ValueError, match="prefix_mask must have shape"):
        make_flow_matching_inputs(
            actions,
            noise,
            torch.rand(2),
            convention=FlowConvention.NOISE_AT_ONE,
            prefix_mask=torch.zeros(2, 4, 3, dtype=torch.bool),
        )
