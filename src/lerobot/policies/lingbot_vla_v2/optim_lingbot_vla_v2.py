# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Optimizer of the upstream LingBot-VLA 2.0 post-training recipe, with fp32 master weights (Xbotics).

Upstream (``lingbotvla/optim/muon.py`` and ``optimizer.py::build_muon_optimizer``) trains the 2D weight
matrices and the 3D MoE expert stacks with Muon (Newton-Schulz orthogonalized momentum, one matrix per
expert) and everything else with AdamW, always on fp32 parameters: FSDP2 keeps fp32 shards and computes
in bf16. This port runs the whole model in ``config.dtype`` (bf16), so the optimizer keeps the fp32 master
copy itself: each update is applied to the master and then rounded into the bf16 parameter. Updating the
bf16 parameters directly loses most updates — at a learning rate of 1e-5, the step is below half a bf16
ulp for about 92% of the action-expert weights of the released checkpoint, so they never move.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from lerobot.optim import OptimizerConfig

# Quintic Newton-Schulz coefficients, step count and epsilon of torch.optim.Muon (and upstream).
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-7

# A parameter may carry the exact fp32 checkpoint values the master copy should start from (set by
# ``LingbotVLAV2Policy.get_optim_params`` when the checkpoint stores fp32); consumed on the first step.
MASTER_INIT_ATTR = "_lingbot_fp32_master_init"


@torch.no_grad()
def batched_newton_schulz(
    grad: Tensor, ns_steps: int = 5, eps: float = NS_EPS, compute_dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    """Orthogonalize each trailing ``[M, K]`` matrix of ``grad`` (upstream ``batched_newton_schulz``)."""
    a, b, c = NS_COEFFICIENTS
    original_dtype = grad.dtype
    ortho = grad.to(compute_dtype)
    transposed = ortho.size(-2) > ortho.size(-1)
    if transposed:
        ortho = ortho.mT
    norm = ortho.norm() if ortho.ndim == 2 else ortho.norm(dim=(-2, -1), keepdim=True)
    ortho = ortho / norm.clamp(min=eps)
    # Fused addmm/baddbmm exactly as upstream and torch.optim.Muon, so the bf16 rounding matches.
    for _ in range(ns_steps):
        gram = ortho @ ortho.mT
        if gram.ndim == 2:
            ortho = torch.addmm(ortho, torch.addmm(gram, gram, gram, beta=b, alpha=c), ortho, beta=a)
        else:
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            ortho = torch.baddbmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.mT
    return ortho.to(original_dtype)


def adjust_muon_lr(lr: float, adjust_lr_fn: str | None, shape: torch.Size) -> float:
    """Per-matrix learning rate of torch.optim.Muon, on the trailing ``[out, in]`` dims (as upstream)."""
    fan_out, fan_in = shape[-2:]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1.0, fan_out / fan_in))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
    raise ValueError(f"Unsupported Muon adjust_lr_fn {adjust_lr_fn!r}.")


class LingbotMuonAdamW(Optimizer):
    """Muon on the groups flagged ``use_muon``, AdamW on the others, both on fp32 master weights.

    Muon follows upstream ``DistributedMuon`` on unsharded tensors: fp32 momentum (lerp), Nesterov
    look-ahead, Newton-Schulz in bf16, then ``adjust_muon_lr``. AdamW is ``torch.optim.AdamW`` without
    amsgrad. Weight decay is decoupled in both. With ``master_weights`` a parameter stored below fp32 is
    updated through an fp32 copy kept in the optimizer state, which is saved and restored with it.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = "match_rms_adamw",
        master_weights: bool = True,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "master_weights": master_weights,
            "use_muon": False,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["use_muon"]:
                shapes = [tuple(p.shape) for p in group["params"] if p.ndim not in (2, 3)]
                if shapes:
                    raise ValueError(f"Muon groups take 2D/3D parameters only, got shapes {shapes}.")

    def _master(self, param: Tensor, state: dict, group: dict) -> Tensor:
        """The fp32 tensor the update is applied to: the parameter itself, or its master copy."""
        init = param.__dict__.pop(MASTER_INIT_ATTR, None)
        if not group["master_weights"] or param.dtype == torch.float32:
            return param
        if "master" not in state:
            source = init if init is not None else param.detach()
            state["master"] = source.to(device=param.device, dtype=torch.float32)
        return state["master"]

    @staticmethod
    def _muon_step(master: Tensor, grad: Tensor, state: dict, group: dict) -> None:
        momentum = group["momentum"]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(master)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if group["nesterov"] else buf
        ortho = batched_newton_schulz(update, group["ns_steps"])
        master.add_(ortho, alpha=-adjust_muon_lr(group["lr"], group["adjust_lr_fn"], master.shape))

    @staticmethod
    def _adamw_step(master: Tensor, grad: Tensor, state: dict, group: dict) -> None:
        beta1, beta2 = group["betas"]
        if "step" not in state:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(master)
            state["exp_avg_sq"] = torch.zeros_like(master)
        state["step"] += 1
        step = state["step"].item()
        state["exp_avg"].lerp_(grad, 1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = (state["exp_avg_sq"].sqrt() / (1 - beta2**step) ** 0.5).add_(group["eps"])
        master.addcdiv_(state["exp_avg"], denom, value=-group["lr"] / (1 - beta1**step))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                master = self._master(param, state, group)
                grad = param.grad.to(torch.float32)
                if group["weight_decay"]:
                    master.mul_(1 - group["lr"] * group["weight_decay"])
                if group["use_muon"]:
                    self._muon_step(master, grad, state, group)
                else:
                    self._adamw_step(master, grad, state, group)
                if master is not param:
                    param.copy_(master)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load as ``Optimizer`` does, then restore the saved precision of the state tensors.

        ``Optimizer.load_state_dict`` casts every floating-point state tensor to its parameter's dtype,
        which would turn the fp32 master copies and moments of bf16 parameters into bf16.
        """
        super().load_state_dict(state_dict)
        saved_ids = chain.from_iterable(group["params"] for group in state_dict["param_groups"])
        params = chain.from_iterable(group["params"] for group in self.param_groups)
        for param_id, param in zip(saved_ids, params, strict=True):
            for key, value in state_dict["state"].get(param_id, {}).items():
                if key != "step" and torch.is_tensor(value) and value.is_floating_point():
                    self.state[param][key] = value.to(device=param.device, copy=True)


@OptimizerConfig.register_subclass("lingbot_muon_adamw")
@dataclass
class LingbotMuonAdamWConfig(OptimizerConfig):
    """``LingbotMuonAdamW``. Each param group says whether it uses Muon (``use_muon``) and its own lr."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr_fn: str | None = "match_rms_adamw"
    master_weights: bool = True

    def build(self, params) -> torch.optim.Optimizer:
        return LingbotMuonAdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
            momentum=self.momentum,
            nesterov=self.nesterov,
            ns_steps=self.ns_steps,
            adjust_lr_fn=self.adjust_lr_fn,
            master_weights=self.master_weights,
        )
