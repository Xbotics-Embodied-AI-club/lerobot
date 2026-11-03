# 第 2 章 训练电脑

训练、仿真评测与推理都在一台配有 NVIDIA 显卡的 x86 电脑上进行，操作系统为 Ubuntu。本章在这台电脑上安装 LeRobot 与 SO-101 仿真，并使端侧计算平台能够连接它。

## 2.1 显卡驱动

```bash
nvidia-smi
```

能列出显卡，且右上角的 `CUDA Version` 不低于 12.8，即可继续。版本较低时，请先升级驱动。

(train-env-setup)=
## 2.2 建立环境

使用 uv 管理 Python 与依赖。先安装系统工具与 uv：

```bash
sudo apt install -y git ffmpeg curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

建立项目目录，并写明需要安装的内容：

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

各可选组的作用：

| 组 | 作用 |
| --- | --- |
| `training`、`evaluation` | 训练与评测，查看训练曲线所用的 TensorBoard 与 wandb 也包含在 `training` 中 |
| `dataset`、`viz` | 读写数据集、可视化 |
| `async` | 远程推理服务端 |
| `smolvla` | SmolVLA 模型 |
| `feetech` | 舵机驱动 |
| `xbotics_tools` | 网页硬件检测、网页标定、推理控制台 |

SO-101 仿真随 LeRobot 一同安装，无需另行安装。此后每次打开新终端，进入该目录后执行 `source .venv/bin/activate` 进入环境。

## 2.3 检查

```bash
python -c "import lerobot, torch, so101_sim; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python -c "import gymnasium as gym, so101_sim; gym.make('SO101PickPlaceCube40-v1').reset(seed=0); print('ok')"
```

第一行应输出 `lerobot 0.6.1` 与 `cuda True`，第二行输出 `ok`，表示仿真可以运行。若出现 `cuda False`，请回到 2.1 检查驱动。

## 2.4 登录 Hugging Face

训练需要从 Hugging Face 下载预训练权重，模型与数据集也可以上传到 Hugging Face。建议登录，使用 PI0 模型时必须登录：

```bash
hf auth login
```

(train-env-ssh)=
## 2.5 允许端侧计算平台连接这台电脑

端侧计算平台通过 SSH 连接这台电脑（[5.2 节](#connect-server)），因此需先安装 SSH 服务，并记下这台电脑的 IP。推理与训练分在两台电脑上时，两台都按本节设置：

```bash
sudo apt install -y openssh-server
hostname -I
```

`hostname -I` 输出的第一个地址，即端侧计算平台网络设置页中「服务器地址」应填写的内容；账号与密码即你登录这台电脑所用的账号与密码。推理服务与 TensorBoard 默认只在本机监听，端侧计算平台经 SSH 接入，电脑无需对外开放其他端口。

(train-env-upgrade)=
## 2.6 升级

端侧计算平台升级时（[1.1 节](#board-upgrade)），这台电脑也需同步升级，保持同一版本：

```bash
cd ~/xbotics && uv lock --upgrade-package lerobot && uv sync
```
