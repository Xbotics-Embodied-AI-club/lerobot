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

"""LingbotMuonAdamW (Xbotics): fp32 master weights, Muon parity with torch.optim.Muon, save/load."""

import pytest
import torch
from torch import nn

from lerobot.optim import load_optimizer_state, save_optimizer_state
from lerobot.policies.lingbot_vla_v2.optim_lingbot_vla_v2 import MASTER_INIT_ATTR, LingbotMuonAdamW

ADAMW = {"betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0}
MUON = {"momentum": 0.95, "nesterov": True, "ns_steps": 5, "adjust_lr_fn": "match_rms_adamw"}


def _grads(shape, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g) for _ in range(n)]


def test_master_weights_follow_fp32_adamw():
    """A bf16 parameter trained through the master copy follows fp32 AdamW; trained directly it barely moves."""
    init = torch.randn(64, 48, generator=torch.Generator().manual_seed(1)) * 0.02
    grads = [g.bfloat16() for g in _grads(init.shape, 20)]

    reference = nn.Parameter(init.clone())
    reference_opt = torch.optim.AdamW([reference], lr=1e-5, **ADAMW)
    direct = nn.Parameter(init.bfloat16())
    direct_opt = torch.optim.AdamW([direct], lr=1e-5, **ADAMW)
    param = nn.Parameter(init.bfloat16())
    param.__dict__[MASTER_INIT_ATTR] = init.clone()  # the fp32 checkpoint values
    opt = LingbotMuonAdamW([{"params": [param], "use_muon": False}], lr=1e-5, **ADAMW)

    for grad in grads:
        for p, o in ((reference, reference_opt), (direct, direct_opt), (param, opt)):
            p.grad = grad.to(p.dtype)
            o.step()

    master = opt.state[param]["master"]
    assert master.dtype == torch.float32
    torch.testing.assert_close(master, reference.detach(), rtol=0, atol=1e-8)
    assert torch.equal(param.detach(), master.bfloat16())
    assert not hasattr(param, MASTER_INIT_ATTR)
    # The failure this fixes: most lr-sized updates round away on a bf16 parameter.
    assert (direct.detach() == init.bfloat16()).float().mean() > 0.5


@pytest.mark.parametrize("shape", [(48, 80), (80, 48)], ids=["wide", "tall"])
def test_muon_matches_torch_muon(shape):
    """2D Muon equals torch.optim.Muon (same Newton-Schulz, momentum and match_rms_adamw lr)."""
    init = torch.randn(shape, generator=torch.Generator().manual_seed(2))
    reference = nn.Parameter(init.clone())
    reference_opt = torch.optim.Muon([reference], lr=1e-3, weight_decay=0.0, **MUON)
    param = nn.Parameter(init.clone())
    opt = LingbotMuonAdamW([{"params": [param], "use_muon": True}], lr=1e-3, **MUON)
    for grad in _grads(shape, 5):
        reference.grad, param.grad = grad.clone(), grad.clone()
        reference_opt.step()
        opt.step()
    torch.testing.assert_close(param.detach(), reference.detach(), rtol=0, atol=1e-6)


def test_muon_on_expert_stack_is_per_expert():
    """A 3D MoE expert stack is orthogonalized one [out, in] matrix per expert, as upstream."""
    init = torch.randn(4, 16, 24, generator=torch.Generator().manual_seed(3))
    stack = nn.Parameter(init.clone())
    slices = [nn.Parameter(init[e].clone()) for e in range(4)]
    stack_opt = LingbotMuonAdamW([{"params": [stack], "use_muon": True}], lr=1e-3, **MUON)
    slice_opt = LingbotMuonAdamW([{"params": slices, "use_muon": True}], lr=1e-3, **MUON)
    for grad in _grads(init.shape, 3):
        stack.grad = grad.clone()
        for e, p in enumerate(slices):
            p.grad = grad[e].clone()
        stack_opt.step()
        slice_opt.step()
    torch.testing.assert_close(stack.detach(), torch.stack([p.detach() for p in slices]), rtol=0, atol=1e-5)


def test_muon_rejects_vectors():
    with pytest.raises(ValueError, match="2D/3D"):
        LingbotMuonAdamW([{"params": [nn.Parameter(torch.zeros(8))], "use_muon": True}])


def test_fp32_state_survives_save_and_load(tmp_path):
    """Masters and moments of bf16 params come back in fp32 through lerobot's checkpoint format."""

    def build(matrix, vector):
        groups = [{"params": [matrix], "use_muon": True}, {"params": [vector], "use_muon": False}]
        return LingbotMuonAdamW(groups, lr=1e-3, **ADAMW, **MUON)

    def train(opt, params, steps):
        for grads in zip(*(_grads(p.shape, steps, seed=i) for i, p in enumerate(params)), strict=True):
            for p, grad in zip(params, grads, strict=True):
                p.grad = grad.bfloat16()
            opt.step()

    init = torch.Generator().manual_seed(4)
    params = [
        nn.Parameter(torch.randn(16, 8, generator=init).bfloat16()),
        nn.Parameter(torch.zeros(8).bfloat16()),
    ]
    opt = build(*params)
    train(opt, params, 3)
    save_optimizer_state(opt, tmp_path)

    restored = [nn.Parameter(p.detach().clone()) for p in params]
    restored_opt = load_optimizer_state(build(*restored), tmp_path)
    for p, q in zip(params, restored, strict=True):
        for key, value in opt.state[p].items():
            assert restored_opt.state[q][key].dtype == value.dtype, key
            assert torch.equal(restored_opt.state[q][key], value), key
    assert restored_opt.state[restored[0]]["master"].dtype == torch.float32

    train(opt, params, 2)
    train(restored_opt, restored, 2)
    for p, q in zip(params, restored, strict=True):
        assert torch.equal(p, q)
