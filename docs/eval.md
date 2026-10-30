# 第 8 章 评测

把模型放到真机上之前，可以先在仿真里评测。仿真能在同样的条件下跑很多局，给出一个可比较的成功率，也不用担心撞坏机械臂。这一章的命令在 x86 电脑上敲。

仿真评测适用于训练数据里含有仿真数据的模型。只用真机数据训练的模型没见过仿真的画面，在仿真里成绩会很低，这不代表它在真机上做不好；这类模型直接进入第 9 章，看真机表现。

## 8.1 跑一次评测

先用我们发布的 ACT 模型跑通。它在仿真和真机混合的数据上训练，任务是把 4 cm 方块放进料盒：

```bash
lerobot-eval --policy.path=Harrysunshine/so101-act-sim-real-cube \
  --env.type=so101_sim --env.task=SO101PickPlaceCube40-v1 --env.control_mode=pd_joint_pos \
  --env.observation_width=640 --env.observation_height=480 --env.fps=30 --env.episode_length=500 \
  --eval.n_episodes=50 --eval.batch_size=1 --output_dir=$HOME/outputs/eval_act
```

结果写在 `--output_dir` 下的 `eval_info.json`，其中 `overall` 里的 `pc_success` 是成功率。每一局的录像存在 `videos/` 下。这个模型在这个场景跑 50 局的成功率是 76%。

<video controls muted playsinline preload="metadata" src="_static/media/eval.mp4"></video>

评自己的模型，把 `--policy.path` 换成权重目录，例如 `$HOME/outputs/act_pick_cube/checkpoints/last/pretrained_model`。

SmolVLA 还要加上训练时的相机对应关系：

```bash
lerobot-eval --policy.path=Harrysunshine/so101-smolvla-sim-real-10task \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --env.type=so101_sim --env.task=SO101PickPlaceCube40-v1 --env.control_mode=pd_joint_pos \
  --env.observation_width=640 --env.observation_height=480 --env.fps=30 --env.episode_length=500 \
  --eval.n_episodes=50 --eval.batch_size=1 --output_dir=$HOME/outputs/eval_smolvla
```

## 8.2 三个场景

| `--env.task` | 场景 |
| --- | --- |
| `SO101PickPlaceCube40-v1` | 抓 4 cm 方块放进料盒 |
| `SO101PickPlaceCube20-v1` | 抓 2 cm 方块放进料盒 |
| `SO101PickPlaceCylinder40-v1` | 抓 4 cm 圆柱放进料盒 |

## 8.3 评测设置要和数据对齐

仿真的设置和数据来源不一致时，命令不会报错，只会得到一个很低的成功率，看起来像模型没学会。下面几项要和数据一致：

| 参数 | 取值 | 对不上时 |
| --- | --- | --- |
| `--env.control_mode` | `pd_joint_pos`，绝对关节角 | 动作被当成增量，机械臂乱动 |
| `--env.observation_width`、`--env.observation_height` | `640`、`480` | 画面和相机视野都对不上 |
| `--env.fps` | `30` | 动作执行快慢不对 |
| `--env.episode_length` | `500`，要装得下一整集 | 任务没做完就被截断 |

关节单位默认已经按真机对齐：五个臂关节是度，夹爪是 0 到 100 的行程百分比。如果用别人的数据集，要先确认它是不是这个单位。

## 8.4 上真机前的判断

- **局数够不够**：10 局的成功率只能精确到 10 个百分点，出结论至少跑 50 局。
- **看录像**：成功率之外，看失败的局是怎么失败的。总是抓不到说明定位有问题，抓到了放不进去说明后半段数据不够。
- **比较几个存点**：用同样的局数评测，选最好的那个。

用含仿真数据训练的模型，在仿真里能稳定完成任务，就可以进入下一章。
