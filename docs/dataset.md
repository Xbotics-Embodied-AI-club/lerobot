# 第 6 章 数据集

录完的数据先检查一遍，再拿去训练。这一章讲数据集长什么样、怎么查看和整理，以及怎么传到 x86 电脑上。

## 6.1 数据集的结构

LeRobot 数据集是一个目录，关节数据存在 Parquet 文件里，相机画面存成视频，描述信息存在 `meta/` 下：

```text
pick_cube/
├── meta/        # 特征定义、任务描述、集的索引、统计量
├── data/        # 每帧的 observation.state 和 action（Parquet）
└── videos/      # observation.images.top、observation.images.wrist（MP4）
```

两个字段最重要。`observation.state` 是从臂此刻的关节读数，`action` 是主臂发出的目标。五个臂关节的单位是度，夹爪是 0 到 100 的行程百分比。模型学的就是：看到相机画面和 `observation.state`，输出 `action`。

## 6.2 传到 x86 电脑

查看、整理和训练都在 x86 电脑上做。在电脑的终端里，把板子上的数据拷过来：

```bash
mkdir -p ~/data && scp -r root@<板子IP>:data/pick_cube ~/data/
```

提示输入密码时填板子的默认密码 `123456`。

## 6.3 查看和回放

在 x86 电脑上用 rerun 看某一集，能同时看到相机画面和关节曲线。这条命令会打开一个桌面窗口：

```bash
lerobot-dataset-viz --repo-id local/pick_cube --root ~/data/pick_cube --episode-index 0
```

<video controls muted playsinline preload="metadata" src="_static/media/rerun.mp4"></video>

把某一集的动作重放到机械臂上，能直观地检查数据。这一步在接着机械臂的板子上做：

```bash
lerobot-replay --robot.type=so101_follower --robot.port=/dev/xbotics/follower --robot.id=follower \
  --dataset.repo_id=local/pick_cube --dataset.root=$HOME/data/pick_cube --dataset.episode=0
```

没有真机时，在 x86 电脑上把同一集的动作灌进仿真，`--robot.video_path` 会存下画面。下面两段就是同一集动作在真机和仿真上的样子：

```bash
lerobot-replay --robot.type=so101_sim --robot.discover_packages_path=so101_sim \
  --robot.video_path=replay.mp4 \
  --dataset.repo_id=local/pick_cube --dataset.root=$HOME/data/pick_cube --dataset.episode=0
```

<div class="video-row">
<video controls muted playsinline preload="metadata" src="_static/media/replay_real.mp4"></video>
<video controls muted playsinline preload="metadata" src="_static/media/replay_sim.mp4"></video>
</div>

## 6.4 删集与合并

发现坏的集，用 `lerobot-edit-dataset` 删掉，它会生成一份新的数据集：

```bash
lerobot-edit-dataset --repo_id local/pick_cube --root ~/data/pick_cube \
  --new_repo_id local/pick_cube_trim --new_root ~/data/pick_cube_trim \
  --operation.type=delete_episodes --operation.episode_indices='[1]'
```

几批数据合成一份：

```bash
lerobot-edit-dataset --operation.type=merge \
  --operation.repo_ids='[local/demo_a,local/demo_b]' --operation.roots="[$HOME/data/demo_a,$HOME/data/demo_b]" \
  --new_repo_id=local/demo_merged --new_root=$HOME/data/demo_merged
```

`demo_a` 和 `demo_b` 是 5.5 节录的两份数据。合并的前提是这几份数据的相机名字、关节单位、分辨率和帧率都一致。

(public-dataset)=
## 6.5 使用开源数据

没有录数据，也可以直接用我们录制并开源的 SO-101 真机数据。它有九个任务：抓电池、易拉罐、方块、橡皮、高尔夫球、药瓶、毛绒玩具放进料盒，以及两个堆叠任务。

<video controls muted playsinline preload="metadata" src="_static/media/ds9.mp4"></video>

下载方块任务（约 500 MB），要别的任务就换 `--include` 里的目录名，去掉 `--include` 是全部九个任务（约 3.5 GB）：

```bash
uvx modelscope download --dataset zhuzhuangtian/so101-pick-place-tasks \
  --include 'pick_up_a_cube_and_place_in_the_bin/*' --local_dir ~/data/so101-pick-place-tasks
```

每个任务是一个子目录，可以直接当 `--dataset.root` 用，例如 `~/data/so101-pick-place-tasks/pick_up_a_cube_and_place_in_the_bin`。后面的命令里出现 `$HOME/data/pick_cube` 时，换成这个目录就能直接跑，`repo_id` 不用改。

仿真数据在 Hugging Face 上，有 `cube40`、`cube20`、`cylinder40` 三个场景，下载 4 cm 方块场景：

```bash
hf download Harrysunshine/so101-sim-pickplace-v2 --repo-type dataset --include 'cube40/*' --local-dir ~/data/so101-sim
```

真机数据和仿真数据可以按 6.4 节合并成一份一起训练。任务描述写法相同的集会算作同一个任务。
