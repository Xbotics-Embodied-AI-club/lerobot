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

"""Gradient accumulation in update_policy (Xbotics): one optimizer update per window of batches."""

import pytest
import torch
from torch import nn

pytest.importorskip("accelerate")

from accelerate import Accelerator  # noqa: E402

from lerobot.scripts.lerobot_train import update_policy  # noqa: E402
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker  # noqa: E402


class LinearPolicy(nn.Module):
    """Stand-in policy: ``forward(batch) -> (loss, outputs)`` on a linear least-squares problem."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))

    def forward(self, batch, reduction="mean"):
        return ((batch["x"] @ self.weight - batch["y"]) ** 2).mean(), {}


def _tracker(accelerator) -> MetricsTracker:
    names = ("loss", "grad_norm", "lr", "update_s", "dataloading_s", "gpu_mem_gb")
    return MetricsTracker(
        2, 8, 1, {name: AverageMeter(name, ":.3f") for name in names}, accelerator=accelerator
    )


@pytest.mark.parametrize("grad_clip_norm", [0.0, 100.0])
def test_one_update_per_accumulation_window(grad_clip_norm):
    """Two accumulated batches give the update of their averaged loss, applied on the second batch only."""
    g = torch.Generator().manual_seed(0)
    batches = [{"x": torch.randn(2, 3, generator=g), "y": torch.randn(2, generator=g)} for _ in range(2)]

    reference = LinearPolicy()
    reference_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    (sum(reference(batch)[0] for batch in batches) / len(batches)).backward()
    reference_opt.step()

    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = LinearPolicy()
    policy, optimizer = accelerator.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    tracker = _tracker(accelerator)

    update_policy(tracker, policy, batches[0], optimizer, grad_clip_norm, accelerator=accelerator)
    assert torch.equal(model.weight.detach(), torch.zeros(3))
    assert tracker.grad_norm.count == 0  # no clipping on a partial gradient

    update_policy(tracker, policy, batches[1], optimizer, grad_clip_norm, accelerator=accelerator)
    torch.testing.assert_close(model.weight.detach(), reference.weight.detach())
    assert tracker.grad_norm.count == 1
