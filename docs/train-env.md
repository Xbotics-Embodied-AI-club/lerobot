# 第 2 章 训练环境

训练、仿真评测和推理都在一台带 NVIDIA 显卡的 x86 电脑上做，系统用 Ubuntu。这一章把 LeRobot 和 SO-101 仿真装到这台电脑上。

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
| `training`、`evaluation` | 训练和评测 |
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

## 2.4 登录 Hugging Face 和 wandb

训练要从 Hugging Face 下载预训练权重，模型和数据集也可以传上去。训练曲线用 wandb 看。两个都建议登录，其中 PI0 模型必须登录 Hugging Face：

```bash
hf auth login
wandb login
```

## 2.5 升级

板子升级时，这台电脑也要一起升，保持同一个提交：

```bash
cd ~/xbotics && uv lock --upgrade-package lerobot && uv sync
```
