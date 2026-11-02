# 第 2 章 训练电脑

训练、仿真评测和推理都在一台带 NVIDIA 显卡的 x86 电脑上做，系统用 Ubuntu。这一章把 LeRobot 和 SO-101 仿真装到这台电脑上，让板子连得上它，再把训练曲线打开。

## 2.1 显卡驱动

```bash
nvidia-smi
```

能列出显卡，右上角的 `CUDA Version` 不低于 12.8，就可以往下走。版本更低时先升级驱动。

## 2.2 建环境

用 uv 管理 Python 和依赖。先装系统工具和 uv：

```bash
sudo apt install -y git ffmpeg curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

建一个项目目录，写清楚要装什么：

```bash
mkdir -p ~/xbotics && cd ~/xbotics
cat > pyproject.toml <<'EOF'
[project]
name = "xbotics-train"
version = "0.0.0"
requires-python = "==3.12.*"
dependencies = ["lerobot[training,evaluation,dataset,viz,async,smolvla,feetech,xbotics_tools]"]

[tool.uv]
package = false

[tool.uv.sources]
lerobot = { git = "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", tag = "xbotics-v0.6.1" }
EOF
uv sync
source .venv/bin/activate
```

几个可选组各管一样：

| 组 | 作用 |
| --- | --- |
| `training`、`evaluation` | 训练和评测，训练曲线用的 TensorBoard 也在 `training` 里 |
| `dataset`、`viz` | 读写数据集、可视化 |
| `async` | 远程推理服务端 |
| `smolvla` | SmolVLA 模型 |
| `feetech` | 舵机驱动 |
| `xbotics_tools` | 网页硬件检测、网页标定、推理控制台 |

SO-101 仿真随 LeRobot 一起装好，不用另外安装。以后每次打开新终端，进目录后敲 `source .venv/bin/activate` 进入环境。

## 2.3 检查

```bash
python -c "import lerobot, torch, so101_sim; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python -c "import gymnasium as gym, so101_sim; gym.make('SO101PickPlaceCube40-v1').reset(seed=0); print('ok')"
```

第一行应当打出 `lerobot 0.6.1` 和 `cuda True`，第二行打出 `ok`，说明仿真能跑。出现 `cuda False` 时先回到 2.1 查驱动。

## 2.4 登录 Hugging Face

训练要从 Hugging Face 下载预训练权重，模型和数据集也可以传上去。建议登录，其中 PI0 模型必须登录：

```bash
hf auth login
```

(train-env-ssh)=
## 2.5 让板子连得上这台电脑

板子经 SSH 连这台电脑（[1.4 节](#connect-server)），所以先装好 SSH 服务，并记下这台电脑的 IP：

```bash
sudo apt install -y openssh-server
hostname -I
```

`hostname -I` 打出的第一个地址，就是板子网络设置页「服务器地址」要填的；账号和密码就是你登录这台电脑用的。推理服务和 TensorBoard 照默认只在这台电脑本机监听，板子经 SSH 连进来，电脑不用对外开放别的端口。

## 2.6 打开训练曲线

`lerobot-train` 训练时，默认把曲线写在输出目录下的 `tensorboard` 里，不用账号、不用联网。训练开始后，日志里会打出一行查看命令：

```text
Watch this run --> tensorboard --logdir outputs/train/<这次训练>/tensorboard
```

另开一个终端，进环境后敲：

```bash
tensorboard --logdir outputs/train
```

浏览器打开 `http://localhost:6006`，几秒内就能看到新的点，训练没结束也能看。`--logdir` 写上一级目录，就能把几次训练放在一起比。在板子那边的电脑上看，就用 [1.4 节](#connect-server) 连上这台电脑，再打开 `http://<板子IP>:6006`。

![训练曲线：损失、梯度、吞吐和显存](images/use-17-tensorboard.png)

记下来的曲线有：

| 曲线 | 是什么 |
| --- | --- |
| `train/loss` 及模型自己的分项损失 | 损失，整体下降、后面趋于平稳是正常的 |
| `train/grad_norm`、`train/lr` | 梯度大小、当前学习率 |
| `train/samples_per_s`、`train/update_s`、`train/dataloading_s` | 每秒训练多少帧，每步花在更新模型和读数据上的秒数 |
| `train/gpu_mem_gb` | 占用的显存 |
| `eval/...` | 训练中做评测时的成绩，如仿真成功率 `eval/pc_success` |

训练用的完整配置记在 TEXT 页。用 wandb 的话，命令里加 `--wandb.enable=true`，两边同时记。

(train-env-upgrade)=
## 2.7 升级

板子升级时，这台电脑也要一起升，保持同一个版本：

```bash
cd ~/xbotics && uv lock --upgrade-package lerobot && uv sync
```
