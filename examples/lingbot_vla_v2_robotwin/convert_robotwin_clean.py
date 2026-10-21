"""RoboTwin 2.0 clean demos -> LeRobot v3, prepared the way LingBot-VLA 2.0 trains on them (Xbotics).

Upstream LingBot-VLA 2.0 trains on RoboTwin data converted by RoboTwin's pi0 scripts
(``policy/pi0/scripts/process_data.py`` + ``examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py``
at RoboTwin commit 13c3c47, the one LingBot pins): for every frame t < T-1, ``observation.state`` is
[left arm 6, left gripper, right arm 6, right gripper] at t and ``action`` the same vector at t+1; the
head / left / right cameras become ``cam_high`` / ``cam_left_wrist`` / ``cam_right_wrist``; 50 fps; each
episode gets one "seen" instruction drawn at random. RoboTwin stores RGB arrays through ``cv2.imencode``,
so ``cv2.imdecode`` gives RGB back, as upstream uses it. One deliberate difference: frames keep the
cameras' native 320x240 instead of being upscaled to 640x480, because the RoboTwin evaluation feeds
native frames and the policy resizes both to 224x224.

Only the clean set is read (``aloha-agilex_clean_50``, 50 episodes per task). Three steps:

1. ``convert`` writes one dataset per worker (run several workers in parallel);
2. merge them with ``lerobot-edit-dataset --operation.type=merge``;
3. ``stats`` replaces the merged state/action statistics with exact global ones (upstream computes its
   normalization over the whole training set), and ``split`` lists train / held-out episodes per task.
"""

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "TianxingChen/RoboTwin2.0"
CLEAN_SET = "aloha-agilex_clean_50"
FPS = 50
MOTORS = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    )
]
CAMERAS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
HEIGHT, WIDTH = 240, 320
EPISODE_MEMBER = re.compile(r"/(data/episode(\d+)\.hdf5|instructions/episode(\d+)\.json)$")
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def list_tasks() -> list[str]:
    tree = HfApi().list_repo_tree(REPO_ID, path_in_repo="dataset", repo_type="dataset")
    return sorted(Path(entry.path).name for entry in tree)


def features() -> dict:
    vector = {"dtype": "float32", "shape": (len(MOTORS),), "names": MOTORS}
    cameras = {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": (HEIGHT, WIDTH, 3),
            "names": ["height", "width", "channels"],
        }
        for name in CAMERAS
    }
    return {"observation.state": vector, "action": vector, **cameras}


def extract_task(task: str, download_dir: Path, work_dir: Path) -> Path:
    """Download the task's clean zip, keep only the episode HDF5 + instruction files, drop the zip."""
    zip_path = Path(
        hf_hub_download(
            REPO_ID,
            f"dataset/{task}/{CLEAN_SET}.zip",
            repo_type="dataset",
            local_dir=download_dir,
        )
    )
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if EPISODE_MEMBER.search(name)]
        archive.extractall(work_dir, members)
    zip_path.unlink()
    return next(work_dir.rglob("data/episode0.hdf5")).parent.parent


def add_episode(dataset: LeRobotDataset, hdf5_path: Path, instruction: str) -> int:
    with h5py.File(hdf5_path, "r") as episode:
        joint = episode["joint_action"]
        state = np.concatenate(
            [
                joint["left_arm"][()],
                joint["left_gripper"][()][:, None],
                joint["right_arm"][()],
                joint["right_gripper"][()][:, None],
            ],
            axis=1,
        ).astype(np.float32)
        frames = {
            name: [
                cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                for jpeg in episode[f"observation/{source}/rgb"][:-1]
            ]
            for name, source in CAMERAS.items()
        }
    for t in range(len(state) - 1):
        frame = {
            "observation.state": state[t],
            "action": state[t + 1],
            "task": instruction,
        }
        frame.update({f"observation.images.{name}": images[t] for name, images in frames.items()})
        dataset.add_frame(frame)
    dataset.save_episode()
    return len(state) - 1


def convert(args: argparse.Namespace) -> None:
    tasks = list_tasks() if args.tasks == "all" else args.tasks.split(",")
    mine = tasks[args.worker_index :: args.num_workers]
    root = Path(args.out_root) / f"part-{args.worker_index:02d}"
    dataset = LeRobotDataset.create(
        repo_id=f"local/robotwin_clean50_part{args.worker_index:02d}",
        fps=FPS,
        features=features(),
        root=root,
        robot_type="aloha",
        use_videos=True,
    )
    # Episode instructions differ, so the RoboTwin task of each episode is recorded beside the part.
    task_log = Path(args.out_root) / f"part-{args.worker_index:02d}.tasks.json"
    written = []
    for task in mine:
        task_index = tasks.index(task)
        work_dir = Path(args.work_dir) / task
        source = extract_task(task, Path(args.download_dir), work_dir)
        frames = 0
        episodes = sorted(
            (source / "data").glob("episode*.hdf5"),
            key=lambda p: int(p.stem.removeprefix("episode")),
        )
        for path in episodes:
            index = int(path.stem.removeprefix("episode"))
            seen = json.loads((source / "instructions" / f"episode{index}.json").read_text())["seen"]
            rng = np.random.default_rng([args.seed, task_index, index])
            frames += add_episode(dataset, path, str(rng.choice(seen)))
        shutil.rmtree(work_dir)
        written.append({"task": task, "episodes": len(episodes), "frames": frames})
        task_log.write_text(json.dumps(written, indent=2))
        print(
            f"[worker {args.worker_index}] {task}: {len(episodes)} episodes, {frames} frames",
            flush=True,
        )
    dataset.finalize()
    print(f"[worker {args.worker_index}] done: {root}", flush=True)


def split(args: argparse.Namespace) -> None:
    """Global train / held-out episode indices of the dataset merged from parts 0..N-1 in order.

    LeRobot's ``eval_split`` groups episodes by their instruction, which is unique per RoboTwin
    episode, so the last ``holdout`` episodes of every RoboTwin task are held out here instead.
    """
    out_root = Path(args.out_root)
    train, held_out, offset = [], [], 0
    for part in range(args.parts):
        for entry in json.loads((out_root / f"part-{part:02d}.tasks.json").read_text()):
            indices = list(range(offset, offset + entry["episodes"]))
            train += indices[: -args.holdout]
            held_out += indices[-args.holdout :]
            offset += entry["episodes"]
    (out_root / "split.json").write_text(json.dumps({"train": train, "held_out": held_out}))
    print(f"{len(train)} train / {len(held_out)} held-out episodes -> {out_root / 'split.json'}")


def exact_stats(args: argparse.Namespace) -> None:
    """Replace state/action stats of a (merged) dataset with exact global ones, quantiles included."""
    root = Path(args.root)
    data = pd.concat(
        pd.read_parquet(path, columns=["observation.state", "action"])
        for path in sorted((root / "data").rglob("*.parquet"))
    )
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for key in ("observation.state", "action"):
        values = np.stack(data[key].to_numpy()).astype(np.float64)
        stats[key] = {
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "count": [len(values)],
            **{f"q{round(q * 100):02d}": np.quantile(values, q, axis=0).tolist() for q in QUANTILES},
        }
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"exact stats over {len(data)} frames written to {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert")
    conv.add_argument(
        "--tasks",
        default="all",
        help='comma-separated task names, or "all" (the 50 tasks)',
    )
    conv.add_argument("--worker-index", type=int, default=0)
    conv.add_argument("--num-workers", type=int, default=1)
    conv.add_argument("--out-root", required=True)
    conv.add_argument("--download-dir", required=True)
    conv.add_argument("--work-dir", required=True)
    conv.add_argument("--seed", type=int, default=0)
    stats = sub.add_parser("stats")
    stats.add_argument("--root", required=True)
    held_out = sub.add_parser("split")
    held_out.add_argument("--out-root", required=True)
    held_out.add_argument("--parts", type=int, required=True)
    held_out.add_argument("--holdout", type=int, default=2)
    args = parser.parse_args()
    {"convert": convert, "stats": exact_stats, "split": split}[args.command](args)


if __name__ == "__main__":
    main()
