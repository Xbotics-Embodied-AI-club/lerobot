# 第 7 章 训练

有了数据，就可以训练模型了。这一章先讲 `lerobot-train` 的命令结构，再用两个实例走一遍：ACT 从零开始训一个单任务模型，SmolVLA 在预训练权重上微调，能听懂任务描述。这一章的命令都在 x86 电脑上敲。没有自己录的数据时，把 `$HOME/data/pick_cube` 换成开源数据的目录（见 [6.5 节](#public-dataset)）。

## 7.1 命令结构

一条训练命令回答三个问题：用什么数据、训什么模型、结果放哪。

| 参数 | 问题 |
| --- | --- |
| `--dataset.repo_id`、`--dataset.root` | 数据在哪 |
| `--policy.type` 或 `--policy.path` | 从零训一种模型，或者从已有权重接着训 |
| `--output_dir` | 权重和日志放哪 |
| `--steps`、`--batch_size` | 训多少步，每步用多少帧 |

## 7.2 实例一：ACT 单任务

ACT 一次预测未来一段动作（动作块），对硬件要求低，是入门的好选择。它从随机初始化开始训，不需要预训练权重：

```bash
lerobot-train --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=local/pick_cube --dataset.root=$HOME/data/pick_cube \
  --steps=50000 --batch_size=64 --save_freq=10000 --log_freq=100 --num_workers=4 \
  --output_dir=$HOME/outputs/act_pick_cube
```

这里的步数和批量就是我们发布的 ACT 模型用的取值。显存不够时把 `--batch_size` 调小。训练时终端会打出步数和 loss：

<video controls muted playsinline preload="metadata" src="_static/media/train.mp4"></video>

## 7.3 实例二：SmolVLA 多任务

SmolVLA 是一个 VLA（视觉-语言-动作）模型，除了相机画面还读任务描述，同一个模型能做多个任务。它从预训练权重 `lerobot/smolvla_base` 微调：

```bash
lerobot-train --policy.path=lerobot/smolvla_base --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.optimizer_lr=5e-5 --policy.scheduler_warmup_steps=2000 \
  --dataset.repo_id=local/pick_cube --dataset.root=$HOME/data/pick_cube \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --steps=30000 --batch_size=16 --save_freq=10000 --log_freq=100 --num_workers=4 \
  --output_dir=$HOME/outputs/smolvla_pick_cube
```

和 ACT 相比多了两样东西：

- **`--rename_map`**：预训练权重认的相机名字是 `camera1`、`camera2`，这里把数据集里的 `top`、`wrist` 对应过去。部署和评测时也要写同样的对应关系。
- **解冻和学习率**：`freeze_vision_encoder=false` 和 `train_expert_only=false` 让整个模型一起微调，所以学习率要配小一些。

多任务时，把几个任务的数据合并成一份（见 6.4 节），每一集的任务描述会保留在数据里。

## 7.4 换其他模型

命令结构都一样，主要改 `--policy.type` 或 `--policy.path`。其他模型要多装对应的可选组：把组名加进 `~/xbotics/pyproject.toml` 的 `lerobot[...]` 方括号里，再敲 `uv sync`。

| 模型 | 写法 | 可选组 | 适合 |
| --- | --- | --- | --- |
| ACT | `--policy.type=act` | 无 | 单任务，入门 |
| Diffusion Policy | `--policy.type=diffusion` | `diffusion` | 单任务，动作有多种做法时 |
| SmolVLA | `--policy.path=lerobot/smolvla_base` | `smolvla` | 多任务，语言指令 |
| PI0 | `--policy.path=lerobot/pi0_base` | `pi` | 多任务，模型大，显存要求高 |

PI0 和 SmolVLA 一样从预训练权重微调，相机名字要对应到它认的名字。它依赖 Hugging Face 上的 `google/paligemma-3b-pt-224`，要先在那个页面同意使用条款，并登录 Hugging Face。显存紧张时打开梯度检查点和半精度：

```bash
lerobot-train --policy.path=lerobot/pi0_base --policy.push_to_hub=false --policy.device=cuda \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
  --dataset.repo_id=local/pick_cube --dataset.root=$HOME/data/pick_cube \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --steps=30000 --batch_size=2 --save_freq=10000 --log_freq=100 --num_workers=2 \
  --output_dir=$HOME/outputs/pi0_pick_cube
```

## 7.5 看训练过程

**loss：** 终端每隔 `--log_freq` 步打一次。loss 整体下降、后面趋于平稳是正常的。但 loss 低不等于任务能完成，最终要看第 8 章的评测。

**wandb：** 命令里加 `--wandb.enable=true`，在网页上看曲线，方便比较几次训练。

**权重：** 每隔 `--save_freq` 步存一份，存在 `<output_dir>/checkpoints/<步数>/pretrained_model`，`checkpoints/last` 指向最新的一份。

**续训：** 训练中断后接着训，`--steps` 写总步数：

```bash
lerobot-train --config_path=$HOME/outputs/act_pick_cube/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=50000
```

**选权重：** 不要默认最后一步最好。把几个存点都拿去评测，选成绩最好的。

## 7.6 多卡训练

有多张显卡时，用 `accelerate launch` 启动同一条命令：

```bash
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 $(which lerobot-train) \
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=local/pick_cube --dataset.root=$HOME/data/pick_cube \
  --steps=50000 --batch_size=64 --output_dir=$HOME/outputs/act_multi_gpu
```

`--batch_size` 是每张卡的批量。
