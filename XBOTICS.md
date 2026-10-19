# 这个 fork 加了什么

Xbotics 维护的 lerobot 分支。这里是 **0.6 线：基线 `v0.6.1`，分支 `xbotics-0.6`**。

## 两条版本线，按需选

| 线 | 分支 | 标签 | 基线 | 谁在用 |
|---|---|---|---|---|
| 0.5 线 | `main` | `xbotics-v0.5.1` | 上游 `v0.5.1` | 工作坊学员机、套件板子现在钉的版本 |
| 0.6 线 | `xbotics-0.6` | `xbotics-v0.6.1` | 上游 `v0.6.1` | 新项目、星禾套件板子；工作坊之后升到这条线 |

```bash
# 克隆：-b 选线（分支跟着修，标签钉死不动）
git clone -b xbotics-0.6 https://github.com/Xbotics-Embodied-AI-club/lerobot.git    # 0.6 线
git clone -b main        https://github.com/Xbotics-Embodied-AI-club/lerobot.git    # 0.5 线

# 当依赖装（uv 项目的 pyproject.toml 里写；要复现就钉标签）
#   lerobot = { git = "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", tag = "xbotics-v0.6.1" }
#   lerobot = { git = "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", tag = "xbotics-v0.5.1" }
pip install "lerobot[all] @ git+https://github.com/Xbotics-Embodied-AI-club/lerobot.git@xbotics-v0.6.1"
```

标签 `xbotics-v0.6.1` / `xbotics-v0.5.1` 指向各条线当前推荐的版本，随修随前移（目前还没有外部使用者，不另起 `.postN`）；要钉死某一版就钉提交号。

功能是**一起装的**，没有「挑几个」这回事 —— 下面几项在本分支上一律可用。

## 改动

| commit | 内容 |
|---|---|
| `feat(policies)` | **VLA-0**（`--policy.type=vla0_smol`）。动作离散成整数当文本 token 让 VLM 自回归输出，VLM 本体零改装，xgrammar 约束解码。骨干 SmolVLM2，512 bin。 |
| `feat(policies)` | **OpenVLA**（`--policy.type=openvla`）。7B prismatic，256 bin 动作 token，`forward()` 自带 CE loss 所以训练路径即原生 SFT。 |
| `feat(envs)` | **SO-101 仿真评测口**（`--env.type=so101_sim`）。 |
| `feat(async_inference)` | **policy_server 接受第三方插件策略**（`lerobot_policy_*`）。上游 0.6.1 的 policy_server 仍只认一张固定白名单。 |
| `fix(envs)` | **异步评测环境找不到 so101_sim**。0.6 的 `lerobot-eval` 默认开异步环境、子进程用 forkserver 起，子进程里没有父进程 import 出来的环境注册，`--env.type=so101_sim` 一启动就 `NameNotFound`；现在子进程里补导一次。 |
| `feat(async_inference)` | **观测压 JPEG + 后台发送**，见下「远程推理」。 |
| `feat(datasets)` | **瑞芯微 MPP 硬件编码**（`h264_rkmpp`）与**边录边编的统计抽样**（`stats_every`），见下「接臂板子录数据」。 |
| `feat(utils)` | **网页预览**（`--display_mode=web`；没有桌面时 `rerun` 自动改走它），见下「接臂板子录数据」。 |
| `feat(policies)` | **LingBot-VLA 2.0**（`--policy.type=lingbot_vla_v2`，蚂蚁灵波）。只在 0.6 线，见下「LingBot-VLA 2.0」。 |
| `feat(board)` | **端侧板按接口位置固定设备名**（udev 规则，RK3566）：从臂串口、顶视相机都有固定名字，见下「端侧板：按接口位置固定设备名」。 |

0.5 线上另有四个 commit 在 0.6 线**不再需要**（上游已修，修法各异）：GR00T N1.5 配置去 `@dataclass`
（上游换成了 N1.7）、`add_envs_task` 绕开 gymnasium 包装器（上游删了这个函数，评测直接 `env.call`）、
diffusers 上界放到 0.41（上游 `diffusers-dep` 是 `>=0.38,<0.40`，0.38 起那个 import 期的 bug 已修）、
PEFT 续训适配器被冻与重复包装（上游两处都修了）。

## 远程推理：观测压 JPEG、后台发送（默认开）

上游 `robot_client` 在控制循环里把两路原图 pickle 后同步发给服务端（640×480 两路约 1.8 MB/次），
发完才执行下一个动作 —— 百兆网口上就一卡一卡，云服务器 5 Mbps 上行下根本发不动。本分支改为：

- 相机画面压 JPEG 再发（`--observation_jpeg_quality=90`；设 0 即发原图）；
- 控制循环只把最新一条观测放进待发位，由后台线程压缩、发送（`--send_observations_in_background=true`）；
  发得慢时旧观测被新的顶掉，服务端总拿到最新的画面。

两项默认都开，命令不用改。RK3566 板子 ↔ server 实测（ACT，两路 640×480 相机，70 s，两条线结果相同）：

| | 动作频率 | 动作间隔 p95 | 板子平均上行 |
|---|---|---|---|
| 原图同步发，百兆局域网 | 16.6 Hz | 322 ms | — |
| JPEG + 后台发，百兆局域网 | 29.4 Hz | 44 ms | 1.4 Mbps |
| 原图同步发，上行限 5 Mbps | 3.9 Hz | 3 s | 2.4 Mbps |
| JPEG + 后台发，上行限 5 Mbps | 29.5 Hz | 44 ms | 1.0 Mbps |

⚠️ 新服务端收得了老客户端发的原图；新客户端（默认发 JPEG）连老服务端不行 —— 板子与服务端一起升级。

## 接臂板子录数据：硬件编码、统计抽样、网页预览（板子上默认生效）

在 RK3566 这类板子上，上游的录数据有三处跟不上：软件 AV1 编码约 11 fps；边录边编时逐帧算图像统计，每路每帧约 15 ms；
`--display_data` 要拉起原生 rerun 查看器，而板子不开桌面、学员手里只有浏览器。本分支在板子上自动改掉这三处，命令不用加参数：

| 项 | 板子上的默认 | 其他机器 | 开关 |
|---|---|---|---|
| 视频编码 | 默认的 libsvtav1（以及 `auto`）自动换成芯片自带的 MPP 硬件 H.264（`h264_rkmpp`，经 GStreamer，带 `GST_MPP_NO_RGA=1`） | 不变 | `LEROBOT_RKMPP=0` 关掉切换；也可显式 `--dataset.rgb_encoder.vcodec=...` |
| 图像统计 | 边录边编时每 10 帧算一次（前两帧总算） | 逐帧（不变） | `--dataset.rgb_encoder.stats_every=N` |
| 预览 | `--display_data=true` 且没有桌面时，板子自己出网页：浏览器打开 `http://<板子>:9090`（只敲这个短地址） | 有桌面时仍是原生 rerun | `--display_mode=web` 显式指定；`LEROBOT_WEB_PREVIEW_HZ`（默认 10）、`LEROBOT_WEB_PREVIEW_PORT`（默认 9090） |

实测（RK3566 4 GB，两路 640×480 MJPG 相机，`lerobot-record` 40 s，`--dataset.streaming_encoding=true --display_data=true`，
不加任何板子专用参数）：自动选中 MPP、预览自动改为网页，录下 1152 帧，平均 28.8 fps，读回正常，图像统计正常；
网页预览 10 次/秒，电脑上只开 Firefox、只敲短地址即出两路画面与关节曲线。

- MPP 判定：ARM + `/proc/device-tree/compatible` 含 rockchip + 装了 `gst-launch-1.0` 与 GStreamer 的 `mpph264enc` 插件；
  板子上装 `gstreamer1.0-tools` 与瑞芯微的 GStreamer 插件包即可。
- `GST_MPP_NO_RGA=1` 不能省：插件默认借 RGA 搬帧，板子有一段物理内存在 4 GB 以上，RGA 碰到就写坏内存、整机挂死。
- 每 10 帧一次的统计与逐帧相比：均值差 ≤0.00025、q99 差 ≤0.012。
- 网页预览要两件事才能只靠浏览器：9090 是跳转页（按浏览器访问板子用的主机名拼出数据地址，跳到 9091 的查看器）；
  rerun 0.3x 的数据服务默认只放行 localhost 页面的跨源请求，这里放行查看器端口。推送在子进程里，不拖录制。
- 录数据建议加 `--dataset.streaming_encoding=true`（边录边编，存盘几乎不用等）；不加时画面先存 PNG、每段结束再用 MPP 编。
- 「有没有桌面」按显示服务是否真在跑来判：本机显示 `DISPLAY=:N` 要有 `/tmp/.X11-unix/XN`，Wayland 要有它的 socket；远端显示（如 SSH 转发的 `localhost:10.0`）照信。板子厂家镜像在每个登录 shell 里写死 `DISPLAY=:0`，桌面关了也照设，网页终端又是登录 shell —— 只看变量就会去拉原生查看器，拉不起来时 rerun 把每帧原图攒在内存里，一分多钟吃光 4 GB 被系统杀掉。录制时的按键监听（`n` / `r` / `q`）用同一个判断，网页终端里按键才管用。
- GStreamer 编码子进程的标准输出接到 `/dev/null`：接着终端时 `gst-launch-1.0` 每秒刷好多行管道位置（`-q` 关不掉），网页终端会被刷屏。

## LingBot-VLA 2.0（`lingbot_vla_v2`，只在 0.6 线）

蚂蚁灵波（Robbyant）开源的 6B VLA：Qwen3-VL-4B 骨干 + 稀疏 MoE 的 Qwen2 动作专家，flow matching 出动作块，
各种机器人都映射到统一的 55 维状态 / 动作槽位上。策略代码取自上游 LeRobot **待合并**的 PR
[#4897](https://github.com/huggingface/lerobot/pull/4897)（head `de61f262`，官方维护者在社区移植 #3967 的基础上重做，
权重逐张量与原版核对过），`src/lerobot/policies/lingbot_vla_v2/` 与它的测试、文档**逐字未改**；
转换好的权重直接用上游发布的 `lerobot/lingbot_vla_v2_base`（微调起点）与 `lerobot/lingbot_vla_v2_robotwin`。

```toml
# uv 项目里装：多加 lingbot_vla_v2 组
dependencies = ["lerobot[lingbot_vla_v2,training,dataset]"]
```

SO-101（5 个关节 + 夹爪，`top` / `wrist` 两路相机）微调：

```bash
lerobot-train \
  --dataset.repo_id=<数据集> \
  --policy.path=lerobot/lingbot_vla_v2_base \
  --policy.state_slots='{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 5}}]}, "observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 6}}]}}' \
  --policy.action_slots='{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 5}}]}, "action.effector.position": {"origin_keys": [{"action": {"start": 5, "end": 6}}]}}' \
  --rename_map='{"observation.images.top": "observation.images.camera_top", "observation.images.wrist": "observation.images.camera_wrist_left"}' \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.push_to_hub=false --batch_size=1 --steps=30000
```

⚠️ `--policy.device=cuda` 要写：`lerobot/lingbot_vla_v2_base` 的 config.json 里记的是 `device: cpu`，不写就在 CPU 上训，不报错，只是每步十几秒。

显存不够时加 `--policy.train_expert_only=true --policy.gradient_checkpointing=true`（只训动作专家）。
微调出来的模型照常由 `policy_server` 加载，板子上 `robot_client --policy_type=lingbot_vla_v2`。
完整参数、推理加速开关见 `docs/source/lingbot_vla_v2.mdx`。

本线上实测（2026-10-18，server 与别的任务共用；数据 = SO-101 真机「把方块放进盒子」300 集，5 关节 + 夹爪，`top` / `wrist`）：

| 项 | 结果 |
|---|---|
| 基础权重按 SO-101 槽位加载、真帧出动作 | 6 维、全有限；GPU 每块 50 步约 0.9 s（fp32、不开加速），CPU 约 10 s |
| `lerobot-train` 短微调（只训动作专家 + 梯度检查点 + bf16 + batch 1，蒸馏开，200 步） | 每步约 0.55 s、显存 26.9 GB；四项蒸馏损失一路降（深度 3.23→1.03 等），动作损失在这个量级下看不出降 |
| 微调出的检查点 → `policy_server` → RK3566 板子 `robot_client`（板子上没升级、不加参数） | 动作 29.0 Hz、间隔中位 33.5 ms / p95 45 ms；服务端单次推理中位 0.29 s（bf16）、加载 60 s |

上游 PR 依赖 v0.6.1 之后 main 上的几处改动，本线只补了它用到的那部分，均照搬上游；除了下面「存出来的格式」那一条，其他策略的行为不变：

| 补的 | 上游出处 | 作用 |
|---|---|---|
| `flow_matching.FlowConvention` / `make_flow_matching_inputs` | #4075、#4631 | 训练时构造加噪动作与速度目标（其他策略仍走各自原写法） |
| 处理流水的「产物」（`ProcessorStep.save_artifacts`、加载时 `_resolve_artifact_paths`） | #4182 | 权重目录里带的 `image_processor/`、`chat_template_tokenizer/`、`tokenizer/` 按相对路径存、加载时就地解析，从 Hub 加载时自动下载这几个子目录 |
| `TokenizerProcessorStep.save_artifacts` | #4182 | 分词器随前后处理一起存进权重目录 |
| `load_pretrained_policy_processors`、按约定调用 `make_<类型>_pre_post_processors_from_pretrained` | #4701 | 从权重加载前后处理时，让策略自己换上新机器人的槽位映射 |
| `PreTrainedConfig` 注册 `torch.dtype` 的 draccus 编解码（不加基类字段） | main | 策略配置里的 `dtype` 能存能读 |
| `resolve_delta_timestamps` 认按模态给的帧序（`image_observation_delta_indices` 等）与 `rename_map` | main | 训练时给每路相机取当前帧与 49 帧后的帧（深度 / 视频蒸馏要用）；没定义这些属性的策略照旧 |
| `async_inference` 白名单加 `lingbot_vla_v2` | — | 推理服务端能加载它 |
| 远程推理认得训练时的相机改名 | 本线自修（上游 main 已删掉整个 `async_inference`） | 见下 |

**远程推理与相机改名。** 上游 0.6.1 的 `policy_server` 无条件用客户端发来的改名表覆盖权重里存的那份，
而 `robot_client` 从来不填（恒为空）；缩放图像时又按策略自己的相机名去找尺寸。于是**训练时用过 `--rename_map` 的策略都没法远程推理**
（LingBot-VLA 2.0 的相机必须叫 `camera_top` / `camera_wrist_left`，服务端每帧报 `KeyError: 'observation.images.top'`）。
本线改为：客户端真给了改名表才覆盖，否则用权重里存的；缩放按实际生效的改名表找尺寸，图像仍由前处理去改名。
没用改名训练的策略行为不变，板子上的客户端不用升级。

⚠️ **存出来的格式变了一点**：本线起，带分词器的策略（SmolVLA、π0 / π0.5 等）存前后处理时会把分词器一并存进权重目录的
`tokenizer/` 子目录、配置里记相对路径（与上游 main 一致）。这样存的权重用本线或上游 main 读没问题；用原版 0.6.1 读会找不到分词器。
以前存的权重不受影响。

许可证：代码与转换后的权重是 Apache-2.0。**训练**时的深度 / 视频蒸馏要下载三个冻结的教师模型，其中 DINO-Video 来自 Meta 的 DINOv3，
受 DINOv3 License 约束（发表要致谢 DINOv3、限制部分用途）；**推理**不加载教师模型。

## 仿真器是独立仓

仿真器本体不在这里，在 [**Xbotics-SO101-Sim**](https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim)，
由本 fork 作**核心依赖**装进来。依赖方向单向：仿真包不 import lerobot，是 lerobot 认识它。

```bash
lerobot-eval --env.type=so101_sim --env.task=SO101PickPlaceCube40-v1 \
             --env.control_mode=pd_joint_pos --eval.n_episodes=20
```

三个分发场景 `SO101PickPlace{Cube40,Cube20,Cylinder40}-v1`，**这就是全部环境 id**
（一个场景一个环境；RL 训练改 reward / 放宽步数走包装器与构造参数，不新注册）。配套公开数据集
<https://huggingface.co/datasets/Harrysunshine/so101-sim-pickplace>。

### ⚠️ 评测口径必须与数据来源对齐

**默认值已经全部对齐真机，正常评测不需要再手传任何 `--env.*`。** 下面这些是「为什么是这些
默认值」以及「什么时候需要改」—— 每一条配错都**不报错**，只给出一个看着像「策略没学会」
的低成功率。

| `--env.` 参数 | 默认 | 依据 |
|---|---|---|
| `unit_convention` | `real` | 真机口径（见下）；`maniskill` 是原生弧度 |
| `control_mode` | `pd_joint_pos` | 已发布数据集录的是**绝对关节角** |
| `observation_width/height` | `640` / `480` | 相机竖直视野角是 ChArUco 在 640×480 下标定的（fovy 59.17°） |
| `episode_length` | `500` | 已发布数据最长 444 帧，短了会静默截断 |
| `fps` | `30` | 仿真 `control_freq` 与数据帧率都是 30 |

#### 真机口径是「混的」，不是统一的角度

真机数据由 `lerobot-record` 采集，走 `so_follower`，而它**逐关节配的归一化模式不一样**：

| 通道 | 真机口径 | 出处 |
|---|---|---|
| 5 个臂关节 | **度** | `SOFollowerConfig.use_degrees` 默认 `True` ⇒ `MotorNormMode.DEGREES` |
| `gripper` | **0~100 行程百分比** | `so_follower` **写死**为 `MotorNormMode.RANGE_0_100`，与 `use_degrees` 无关 |

所以「统一到真机」不是一个单位换算，是**逐通道**换算。`unit_convention="real"` 就这么做：
臂关节 弧度→度；夹爪 弧度→`(deg − deg_lo)/(deg_hi − deg_lo) × 100`（界取自夹爪关节自己的限位）。

**夹爪这一处特别容易错**：度数与百分比的量级恰好撞车（物理行程约 0~100 度），看数值看不出，
只表现为**抓取这一环学不动**。校验点：仿真「张开到位」44.95° → 49.95%，真机实测 50.6%。

⚠️ 社区的 SO-101 数据集**不都是这个口径**：官方 `lerobot/svla_so101_pickplace` 是归一化 ±100
（`action` 恰好触到 ±100.00，那是 clamp 的签名）。对齐只能以**你自己那台真机**为基准。

这些都真实踩过：一次 ACT 评测拿到 0%、一次 SmolVLA 六个 checkpoint 全 0，
两次都排查到最后才发现是口径而不是策略。

## aarch64 板端（接机械臂的板子：瑞芯微 RK3566 / RK3576、地瓜 RDK 等）

仿真依赖带了平台标记 —— mani-skill 的物理后端 `sapien` 不发 aarch64 轮子，板端不跑仿真，安装不受影响。
板端只做舵机、相机、录数据与远程推理客户端，0.6 线要装 **`lerobot[feetech,async,dataset,viz]`**
（0.6 起 datasets 与 rerun 都不在核心依赖里了）。硬件编码、统计抽样、网页预览三项自动生效（见上一节）。另有几处要在板子自己的环境里处理：

- **torch**：上游 aarch64 轮子从 2.10 起在 ARMv8.0 核（A53 / A72）上一导入就 `Illegal instruction`；
  这类板子钉 `torch==2.9.1`。A55（RK3566，ARMv8.2）上 2.10 实测可用。
- **torchcodec**：`dataset` 组在 aarch64 上要求 `>=0.11`，而那份 aarch64 轮子是对着 CUDA 13 编的，
  没有 NVIDIA 显卡的板子上必然加载失败（找 `libnvrtc.so.13`）、刷一屏报错后退回 PyAV ⇒ 板子环境里直接
  排除它（uv：`override-dependencies = ["torchcodec; sys_platform == 'never'"]`）。
- **rerun-sdk**：板子出网页预览（学员只用浏览器看）时钉 `rerun-sdk==0.26.2`。0.33.1 的网页查看器在没有显卡加速的
  浏览器里一收到数据就卡死；0.26.2 同一浏览器正常。`viz` 组声明的是 `>=0.24,<0.34`，0.26.2 在范围内。

BPU 上板推理用地瓜官方 [`rdk_LeRobot_tools`](https://github.com/D-Robotics/rdk_LeRobot_tools)，不在本 fork 内。

## 端侧板：按接口位置固定设备名（RK3566）

板子上的 `/dev/videoN` 是按枚举顺序分的，换个口、重启一次都可能变。
[`xbotics/udev/99-xbotics-rk3566.rules`](xbotics/udev/99-xbotics-rk3566.rules) 按**插在板子的哪个口**起固定名字，
不按序号，也不按相机型号或序列号 —— 同型号的相机插哪个口就叫哪个名字，换相机不用改规则：

| 固定名字 | 是什么 | 按什么匹配 |
|---|---|---|
| `/dev/xbotics/follower` | 从臂舵机（板载串口，系统里叫 `ttyS3`） | 串口控制器的设备树节点 `fe6b0000.serial` |
| `/dev/xbotics/top` | 顶视相机 | USB 接口路径 `platform-xhci-hcd.4.auto-usb-0:1:1.0`（xhci-hcd.4 控制器的 1 号口） |

腕部相机的口还没定，定了再加一行。只适用于 ZTL A566 这一款板子。

装（板子上，root）：

```bash
curl -fsSL https://raw.githubusercontent.com/Xbotics-Embodied-AI-club/lerobot/xbotics-v0.6.1/xbotics/udev/99-xbotics-rk3566.rules \
  -o /etc/udev/rules.d/99-xbotics-rk3566.rules
udevadm control --reload
udevadm trigger --action=add --subsystem-match=tty --subsystem-match=video4linux
ls -l /dev/xbotics/
```

最后一行应当列出 `follower -> ../ttyS3`、`top -> ../videoN`。之后命令里直接写这些名字：
`--robot.port=/dev/xbotics/follower`，相机 `index_or_path: /dev/xbotics/top`。

实测（2026-10-19，ZTL A566）：经固定名字，从臂 6 个 STS3215 在 1 Mbps 下全部读到（同步读位置 300 次 0 失败、中位 1.3 ms）；
顶视相机出 640×480、约 29 fps。

**换一款板子**：这份规则只认这款板子的串口地址和 USB 控制器名。把设备插好，读出位置再照着写一份：

```bash
udevadm info -q property -n /dev/videoN | grep '^ID_PATH='                 # 相机：取 index 为 0 的那个节点
udevadm info -a -n /dev/ttyS3 | grep -m1 'KERNELS==".*\.serial"'           # 板载串口
```

## 从 0.5 线迁到 0.6 线要改什么

上游 0.5.1 → 0.6.1 之间有 321 个 commit。我们自己的代码（工作坊教学仓、DM0.5 插件、套件实验、讲义）
用到的 76 个 LeRobot 导入里，0.6 线只少了 `add_envs_task`；其余变化在用法上：

| 变化 | 0.5 线 | 0.6 线 |
|---|---|---|
| 依赖 | datasets / accelerate / diffusers 是核心依赖 | 挪进可选组：`dataset`、`accelerate-dep`、`diffusers-dep`；rerun 在 `viz`；transformers 在 `transformers-dep`；`lerobot-eval` 要装 `evaluation` **和** `dataset`（上游 `evaluation` 组只有 av，评测脚本却无条件导入数据集） |
| 录数据选编码器（命令行） | `--dataset.vcodec=h264` | `--dataset.rgb_encoder.vcodec=h264` |
| 录数据选编码器（Python） | `LeRobotDataset.create(..., vcodec="h264")` | `create(..., rgb_encoder=RGBEncoderConfig(vcodec="h264"))`（`lerobot.configs.video`） |
| 模块路径 | `lerobot.types` | `lerobot.lerobot_types`；`PreTrainedConfig` 可从 `lerobot.configs` 导，`get_policy_class` / `make_pre_post_processors` 可从 `lerobot.policies` 导 |
| 环境里取任务指令 | `lerobot.envs.utils.add_envs_task` | 已删除，改用 `env.call("task_description")` |
| 策略的注册 | `factory.py` 里逐个 if/elif | 配置类 `@PreTrainedConfig.register_subclass` + 命名约定（`XConfig` → `modeling_*.XPolicy` / `processor_*.make_x_pre_post_processors`），内置策略靠 `lerobot/policies/__init__.py` 导入配置类注册 |
| GR00T | N1.5 | N1.7 |
| rerun-sdk | 0.26.2 | 0.33.1（浏览器 / 拉流端要用同一大版本；板子出网页预览钉 0.26.2，见上） |
| torch（锁文件） | 2.10 | 2.11（ARMv8.0 板子见上一节） |

没变的：机器人 / 相机 / 第三方插件的注册方式；`robot_client` 与 `policy_server` 的命令行参数
（工作坊那组实测两条线通用）。边录边编的统计在 0.6 线按平台抽样（板子每 10 帧，其余逐帧），0.5 线仍逐帧。

⚠️ **板子与推理服务端必须用同一条线、同一个标签**：两端经 gRPC 互传 pickle 后的 LeRobot 对象，
0.5 客户端连 0.6 服务端（或反过来）没有验证过；JPEG 上传要两端都是带它的版本（工作坊升级时两端一起换到 `xbotics-v0.6.1`）。

## 跟上游的关系

`upstream` = `huggingface/lerobot`。本分支是在上游 `v0.6.1` 上逐个重放 0.5 线的 commit 得到的
（不 squash、无合并提交）；移植时改写过的 commit，提交说明里写了改了什么、为什么。
上游再出新版时照同样的办法开新分支，0.5 / 0.6 两条线都保留。
