"""Held-out action error of a LingBot-VLA 2.0 checkpoint on RoboTwin, in raw joint units (Xbotics).

Training losses are not comparable across normalizations — MEAN_STD and QUANTILES give the normalized
action different scales — so checkpoints are compared on the action chunk the policy actually outputs:
the postprocessed (raw-unit) chunk against the recorded one, as mean absolute error over the valid
(non-padded) steps, on frames spread evenly over the held-out episodes, with fixed denoising noise.
Arm joints are in radians, grippers in RoboTwin's 0-1 opening. The "hold" baseline repeats the current
state over the chunk, so a policy must beat it to be predicting motion at all.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

GRIPPERS = (6, 13)


def summarize(abs_err: np.ndarray, valid: np.ndarray) -> dict:
    """``abs_err`` [N, T, D] and ``valid`` [N, T] -> MAE over valid steps: arms, grippers, first step."""
    arm = [d for d in range(abs_err.shape[-1]) if d not in GRIPPERS]
    steps = abs_err[valid]
    return {
        "arm_mae_rad": float(steps[:, arm].mean()),
        "gripper_mae": float(steps[:, list(GRIPPERS)].mean()),
        "first_step_arm_mae_rad": float(abs_err[:, 0][valid[:, 0]][:, arm].mean()),
        "per_joint_mae": steps.mean(0).round(5).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="a pretrained_model directory")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="local/robotwin_clean50")
    parser.add_argument(
        "--split",
        required=True,
        help="split.json written by robotwin_clean_to_lerobot.py",
    )
    parser.add_argument("--num-frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path, cfg.device = checkpoint, "cuda"
    held_out = json.loads(Path(args.split).read_text())["held_out"]
    probe = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=held_out[:1])
    chunk = [i / probe.fps for i in range(cfg.chunk_size)]
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=held_out,
        delta_timestamps={ACTION: chunk},
    )
    # The camera renaming used in training is recorded in the checkpoint's train_config.json.
    rename_map = json.loads((checkpoint / "train_config.json").read_text()).get("rename_map", {})
    policy = make_policy(cfg, ds_meta=dataset.meta, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    indices = np.linspace(0, len(dataset) - 1, args.num_frames).round().astype(int)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    errors, holds, valids = [], [], []
    for start in range(0, len(indices), args.batch_size):
        samples = [dataset[int(i)] for i in indices[start : start + args.batch_size]]
        batch = {
            key: torch.stack([s[key] for s in samples])
            for key in samples[0]
            if torch.is_tensor(samples[0][key])
        }
        batch["task"] = [s["task"] for s in samples]
        truth, state, valid = (
            batch[ACTION],
            batch[OBS_STATE],
            ~batch[f"{ACTION}_is_pad"],
        )
        model_batch = preprocessor({k: v for k, v in batch.items() if k != ACTION})
        noise = torch.randn(
            (len(samples), cfg.chunk_size, cfg.max_action_dim),
            generator=generator,
            device="cuda",
        ).to(next(policy.parameters()).dtype)  # the model runs in its config dtype (bf16)
        with torch.no_grad():
            predicted = postprocessor(policy.predict_action_chunk(model_batch, noise=noise))
        errors.append((predicted.float().cpu() - truth).abs().numpy())
        holds.append((state[:, None] - truth).abs().numpy())
        valids.append(valid.numpy())

    error, hold, valid = (np.concatenate(x) for x in (errors, holds, valids))
    result = {
        "checkpoint": str(checkpoint),
        "frames": len(indices),
        "held_out_episodes": len(held_out),
        "policy": summarize(error, valid),
        "hold_baseline": summarize(hold, valid),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_joint_mae"}, indent=2))


if __name__ == "__main__":
    main()
