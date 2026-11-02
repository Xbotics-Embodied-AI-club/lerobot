# Xbotics LeRobot：让 SO-101 自己完成任务

<video controls muted playsinline preload="metadata" poster="_static/poster/hook.jpg" src="_static/media/hook.mp4"></video>

Xbotics 维护的 LeRobot 把具身智能的整条流程做成了开箱就能用的一套：在星禾套件上用主臂带着从臂示范动作、录成数据，在一台带 NVIDIA 显卡的电脑上训练模型，先在与真机对齐的 SO-101 仿真里评测，再让模型经网络控制真机。上面这段是走完流程后的样子：模型只看两路相机画面和关节读数，自己把方块放进料盒。

这个站点介绍它能做什么，并带你完成基础配置：板子连好网、连上训练电脑，两条机械臂标定好、主臂能带着从臂动。

```{rubric} 用到的硬件
```

| 部件 | 是什么 |
| --- | --- |
| 主臂、从臂（星禾套件） | 一对 SO-101 机械臂：人握着主臂示范，从臂跟着做 |
| 端侧板（星禾套件） | 瑞芯微 RK3566 板子，接两条臂和两路相机，负责录数据和执行动作 |
| 相机（星禾套件） | 顶视、腕部两路 USB 相机 |
| 训练电脑 | 你自己的 x86 电脑，带 NVIDIA 显卡：训练、仿真评测、推理 |

## 它能做什么

### 看着画面，自己完成抓取

模型每次读两路相机画面和六个关节的读数，算出接下来一段动作；板子逐个执行，赶在这一段用完之前拿到下一段，机械臂一气呵成，每秒约 30 次动作。

<video controls muted playsinline preload="metadata" poster="_static/poster/deploy.jpg" src="_static/media/deploy.mp4"></video>

### 仿真与真机对齐

SO-101 仿真按真机搭建：机械臂的模型、相机的位置和视野、控制频率、关节的单位都与真机一致。下面是同一集录好的动作，左边灌给真机，右边灌给仿真：

<div class="video-row">
<video controls muted playsinline preload="metadata" poster="_static/poster/replay_real.jpg" src="_static/media/replay_real.mp4"></video>
<video controls muted playsinline preload="metadata" poster="_static/poster/replay_sim.jpg" src="_static/media/replay_sim.mp4"></video>
</div>

所以仿真既能当评测环境，同样的条件跑几十局给出一个可比较的成功率；也能当一台机器人，没有机械臂时照样走通录数据和部署。我们发布的 ACT 模型用真机加仿真的数据训练，在 4 cm 方块场景跑 50 局，成功率 76%：

<video controls muted playsinline preload="metadata" poster="_static/poster/eval.jpg" src="_static/media/eval.mp4"></video>

### 开源的数据和模型

我们录制并开源了九个任务的 SO-101 真机数据：抓电池、易拉罐、方块、橡皮、高尔夫球、药瓶、毛绒玩具放进料盒，以及两个堆叠任务，共 2200 集、约 78 万帧，每集都带顶视和腕部两路画面。

<video controls muted playsinline preload="metadata" poster="_static/poster/ds9.jpg" src="_static/media/ds9.mp4"></video>

| 开源的 | 在哪 |
| --- | --- |
| 九任务真机数据 | ModelScope [`zhuzhuangtian/so101-pick-place-tasks`](https://modelscope.cn/datasets/zhuzhuangtian/so101-pick-place-tasks) |
| 仿真数据（4 cm 方块、2 cm 方块、4 cm 圆柱三个场景） | Hugging Face [`Harrysunshine/so101-sim-pickplace-v2`](https://huggingface.co/datasets/Harrysunshine/so101-sim-pickplace-v2) |
| ACT 模型（真机 + 仿真数据，方块任务） | Hugging Face [`Harrysunshine/so101-act-sim-real-cube`](https://huggingface.co/Harrysunshine/so101-act-sim-real-cube) |
| SmolVLA 模型（真机 + 仿真数据，十个任务，听语言指令） | Hugging Face [`Harrysunshine/so101-smolvla-sim-real-10task`](https://huggingface.co/Harrysunshine/so101-smolvla-sim-real-10task) |

### 板子上只要一个浏览器

端侧板不接显示器，所有操作都在浏览器里：连上板子的热点，网络设置页自动弹出；之后网页终端、浏览器版 VS Code、JupyterLab、硬件检测、网页标定都用一个网址打开，你的电脑上不用装任何软件。

<div class="shot-grid">
<figure><img src="_static/showcase/network-page.png" alt="网络设置页"><figcaption>网络设置：连 Wi-Fi、设网线，所有网页一处打开</figcaption></figure>
<figure><img src="_static/showcase/use-12-hardware-check.png" alt="硬件检测"><figcaption>硬件检测：两路相机实时画面，一键读出每个舵机的状态</figcaption></figure>
<figure><img src="_static/showcase/use-13-calibrate-web.png" alt="网页标定"><figcaption>网页标定：照着 3D 模型一个关节一个关节来</figcaption></figure>
<figure><img src="_static/showcase/use-07-vscode-run.png" alt="浏览器版 VS Code"><figcaption>浏览器版 VS Code：看、改、运行板子上的代码</figcaption></figure>
</div>

### 连上训练电脑，一处填好

推理服务和训练曲线都在训练电脑上。板子的网络设置页里填上电脑的地址、账号和密码（或私钥），板子就把电脑上的推理服务和 TensorBoard 接到自己身上：板子上的程序直接连本机，你在任何一台电脑上打开板子的地址就能看训练曲线。断网、重启后自己连回。

<div class="shot-grid">
<figure><img src="_static/showcase/connect-server.png" alt="连服务器"><figcaption>连服务器：每一路映射通没通，一眼看清</figcaption></figure>
<figure><img src="_static/showcase/use-17-tensorboard.png" alt="TensorBoard"><figcaption>训练曲线：损失、梯度、吞吐和显存，边训边看</figcaption></figure>
</div>

### 小板子也能稳定录数据

RK3566 这类板子算力有限，原版 LeRobot 用软件编码视频，在上面每秒只编得动约 11 帧。维护版在板子上自动改用芯片自带的硬件编码器、每 10 帧算一次画面统计量、预览改成网页，两路 640×480 相机实测平均每秒 28.8 帧，命令不用加任何参数。录的时候在浏览器里看两路画面和关节曲线：

<video controls muted playsinline preload="metadata" poster="_static/poster/rerun.jpg" src="_static/media/rerun.mp4"></video>

### 网络差也跑得满

部署时相机画面压缩后在后台发送，控制循环不等网络。板子和电脑之间实测（ACT，两路 640×480 相机）：

| 网络 | 动作频率 | 动作间隔 p95 | 板子平均上行 |
| --- | --- | --- | --- |
| 百兆局域网 | 29.4 Hz | 44 ms | 1.4 Mbps |
| 上行限 5 Mbps | 29.5 Hz | 44 ms | 1.0 Mbps |

原版同步发送原图时，同样限到 5 Mbps 只剩每秒 3.9 次动作。

### 一条命令装好，一条命令升级

板子上的全部东西（系统设置、机械臂与相机的固定名字、六个网页服务、LeRobot 环境）都随 LeRobot 一起发布。新板子敲一条命令装好，以后敲 `xbotics-config update` 一起升级：服务开关、改过的密码、自己装的包都保留。

<img class="shot" src="_static/showcase/install-02-update.png" alt="xbotics-config update">

### 支持的模型

| 模型 | 特点 |
| --- | --- |
| ACT | 一次预测一段动作，对显卡要求低，单任务入门首选 |
| Diffusion Policy | 用扩散模型生成动作，同一任务有多种做法时更稳 |
| SmolVLA | 视觉-语言-动作模型，读任务描述，一个模型做多个任务 |
| PI0 | 更大的视觉-语言-动作模型，多任务，显存要求高 |

## 从这里开始

三章基础配置，照着做完，你的板子连好网、连上训练电脑，两条机械臂标定好，主臂能带着从臂动：

1. **[板子上手](board.md)**：烧写镜像、一条命令装好、配网、连训练电脑、网页终端和编辑器
2. **[训练电脑](train-env.md)**：在带 NVIDIA 显卡的电脑上装好 LeRobot 和 SO-101 仿真，打开训练曲线
3. **[硬件准备](hardware.md)**：接线、硬件检测、标定，用遥操作检查一遍

录数据怎么设计任务、训练怎么选模型和调参数、评测与上真机怎么判断，是 Xbotics 具身智能课程的内容，在课程里带着你一步步做。

## 用的版本

全部内容用 Xbotics 维护的 LeRobot：分支 `xbotics-0.6`，标签 `xbotics-v0.6.1`，基于上游 LeRobot 0.6.1。板子和训练电脑要用同一个标签。代码与改动说明在 [GitHub](https://github.com/Xbotics-Embodied-AI-club/lerobot/tree/xbotics-0.6)。

```{toctree}
:hidden:
:maxdepth: 2

board
train-env
hardware
```
