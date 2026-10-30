# 第 4 章 遥操作

遥操作是人握着主臂运动，从臂实时跟随。录数据时人就是这样示范动作的，所以先单独练一遍：确认两条臂都工作正常，也熟悉一下手感。

## 4.1 主臂带从臂

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/xbotics/follower --robot.id=follower \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader --teleop.port=/dev/xbotics/leader --teleop.id=leader \
  --display_data=true
```

`--robot` 是从臂，`--teleop` 是主臂，`id` 要和标定时一致。按 `Ctrl+C` 停下。

## 4.2 看相机和关节曲线

带了 `--display_data=true`，板子会自己开一个网页预览，终端里打出地址。浏览器打开 `http://<板子IP>:9090`，能看到两路相机画面和关节曲线。录制开始后要等十几秒才打得开。

<video controls muted playsinline preload="metadata" src="_static/media/rerun.mp4"></video>

## 4.3 上手检查

- 从臂跟得上主臂，没有明显延迟和抖动
- 每个关节都能转到需要的位置，夹爪能完全张开和闭合
- 顶视相机能看到整个工作区：机械臂、物体和料盒都在画面里
- 腕部相机能看到夹爪和要抓的物体

有一项不对就回到第 3 章重新检查或标定。

## 4.4 没有机械臂时用仿真

SO-101 仿真可以当一台机器人用，回放遥操器 `so101_dataset_player` 把一份已录数据集的动作逐帧播出来，代替人手握主臂。这一节在 x86 电脑上做，并用到我们开源的九任务数据，下载方法见 [6.5 节](#public-dataset)。

```bash
REAL=~/data/so101-pick-place-tasks/pick_up_a_cube_and_place_in_the_bin
lerobot-teleoperate \
  --robot.type=so101_sim --robot.discover_packages_path=so101_sim \
  --teleop.type=so101_dataset_player --teleop.discover_packages_path=so101_sim \
  --teleop.repo_id=local/pick_cube --teleop.root=$REAL --teleop.episode=0 \
  --teleop_time_s=10 --fps=30
```

和真机的写法相比，只换了 `--robot` 和 `--teleop`。后面录数据、部署也是这样。
