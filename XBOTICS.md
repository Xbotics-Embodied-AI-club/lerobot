# 这个 fork 加了什么

Xbotics 维护的 lerobot 分支。这里是 **0.6 线：基线 `v0.6.1`，分支 `xbotics-0.6`**。

## 版本

只维护 0.6 线：分支 `xbotics-0.6`，标签 `xbotics-v0.6.1`，基线上游 `v0.6.1`。0.5 线（`main` / `xbotics-v0.5.1`）停在原处，不再更新。

```bash
git clone -b xbotics-0.6 https://github.com/Xbotics-Embodied-AI-club/lerobot.git

# 作为依赖安装（写在 uv 项目的 pyproject.toml 中；需要复现时固定到标签或提交号）
#   lerobot = { git = "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", tag = "xbotics-v0.6.1" }
```

标签 `xbotics-v0.6.1` 指向当前推荐的版本，随修订前移（目前尚无外部使用者，不另起 `.postN`）；需固定某一版本时，请固定提交号。

功能**一并安装**，不可单独选装 —— 下列各项在本分支上一律可用。

## 改动

| commit | 内容 |
|---|---|
| `feat(policies)` | **VLA-0**（`--policy.type=vla0_smol`）。动作离散为整数，作为文本 token 由 VLM 自回归输出，VLM 本体无需改动，并采用 xgrammar 约束解码。骨干 SmolVLM2，512 bin。 |
| `feat(policies)` | **OpenVLA**（`--policy.type=openvla`）。7B prismatic，256 bin 动作 token，`forward()` 自带 CE loss 因此训练路径即原生 SFT。 |
| `feat(envs)` | **SO-101 仿真评测口**（`--env.type=so101_sim`）。 |
| `feat(async_inference)` | **policy_server 接受第三方插件策略**（`lerobot_policy_*`）。上游 0.6.1 的 policy_server 仍只认一张固定白名单。 |
| `fix(envs)` | **异步评测环境找不到 so101_sim**。0.6 的 `lerobot-eval` 默认开启异步环境，子进程以 forkserver 方式启动，子进程中没有父进程导入的环境注册，`--env.type=so101_sim` 启动即报 `NameNotFound`；现已在子进程中补充导入一次。 |
| `feat(async_inference)` | **观测压 JPEG + 后台发送**，见下「远程推理」。 |
| `feat(datasets)` | **瑞芯微 MPP 硬件编码**（`h264_rkmpp`）与**边录边编码时的统计抽样**（`stats_every`），见下「端侧板录数据」。 |
| `feat(utils)` | **网页预览**（`--display_mode=web`；无桌面时 `rerun` 自动改用它），见下「端侧板录数据」。 |
| `feat(policies)` | **LingBot-VLA 2.0**（`--policy.type=lingbot_vla_v2`，蚂蚁灵波）。只在 0.6 线，见下「LingBot-VLA 2.0」。 |
| `feat(lingbot_vla_v2)` | **LingBot 训练改为官方配方**：fp32 主权重、Muon、分位数归一化；`lerobot-train --gradient_accumulation_steps`；RoboTwin clean 转换脚本。见下「LingBot-VLA 2.0」。 |
| `feat(board)` | **端侧板按接口位置固定设备名**（udev 规则，RK3566）：主从两条臂的串口、顶视与腕部两路相机都有固定名字，见下「端侧板：按接口位置固定设备名」。 |
| `feat(board)` | **端侧板网络设置页**（RK3566）：端侧板持续开启热点，连接后在浏览器中为端侧板配置 Wi-Fi 与网线，结果实时显示，见下「端侧板：网络设置页」。 |
| `fix(async_inference)` | **policy_server 两处防停滞**：推理成功后才记录「这一步已推理」，失败时把「上次推理过的观测」还原；机械臂不动、观测全部被相似度检查滤掉时，5 秒内无观测入队则强制放入最新一帧（`--similarity_stall_timeout`）。见下「网页工具」。 |
| `feat(train)` | **训练默认写 TensorBoard**（`<output_dir>/tensorboard`），训练过程中即可查看曲线。见下「训练看 TensorBoard」。 |
| `feat(xbotics)` | **端侧板上需维护的内容全部收入包内**：一条命令即可完成新端侧板的安装，`xbotics-config update` 升级 LeRobot，并同步升级系统设置、网页与服务。见下「端侧板：安装与升级」。 |
| `feat(xbotics)` | **网络设置页新增「连接服务器」**：通过 SSH 隧道将服务器上的推理服务、TensorBoard 映射到端侧板的 0.0.0.0，支持账号密码或私钥登录，断线自动重连。见下「端侧板：连接服务器」。 |
| `feat(xbotics)` | **三个网页工具**：硬件检测 `lerobot-hardware-check`（8005）、SO101 网页标定 `lerobot-calibrate-web`（8006）、带 Web 控制台与 RTC 的推理客户端 `lerobot-inference-console`（8007），需安装 `lerobot[xbotics_tools]`。见下「网页工具」。 |

0.5 线上另有四个 commit 在 0.6 线**不再需要**（上游已修复，修法各异）：GR00T N1.5 配置移除 `@dataclass`
（上游已换成 N1.7）、`add_envs_task` 绕开 gymnasium 包装器（上游已删除该函数，评测直接 `env.call`）、
diffusers 上界放宽到 0.41（上游 `diffusers-dep` 是 `>=0.38,<0.40`，0.38 起该导入期的 bug 已修复）、
PEFT 续训适配器被冻结与重复包装（上游两处均已修复）。

## 训练看 TensorBoard（默认开启）

`lerobot-train` 把训练曲线写进 `<output_dir>/tensorboard`，训练过程中另开一个终端即可查看，无需账号、无需联网：

```bash
tensorboard --logdir outputs/train/<这次训练>/tensorboard          # 浏览器打开 http://localhost:6006
tensorboard --logdir outputs/train --bind_all                     # 同时查看所有训练，并允许局域网中的其他电脑访问
```

- 记录的指标与 W&B 同名：`train/loss`、`train/grad_norm`、`train/lr`、`train/update_s`、`train/samples_per_s`、`train/gpu_mem_gb`，
  策略自己的子损失（ACT 的 `l1_loss` / `kld_loss` 等），留出集的 `eval/eval_loss`，仿真评测的 `eval/pc_success`、`eval/avg_sum_reward` 及按场景分组的同名指标；训练配置记录在 TEXT 页。
- 每 `log_freq` 步写入一次并立即落盘，已开启的 TensorBoard 数秒内显示新数据点。
- 从检查点续训（`--resume=true`）时，中断前写过检查点之后的那段数据点作废，曲线从续训的步数处接续。
- 关闭：`--tensorboard.enable=false`；更改目录：`--tensorboard.log_dir=<目录>`。可与 `--wandb.enable=true` 同时开启。
- 训练在服务器上、需在端侧板一侧的电脑上查看时：通过端侧板网络设置页的「连接服务器」将 6006 端口映射过来，见下「端侧板：连接服务器」。

实测（server，ACT，`so101_sim/pick_place_cube_40mm`，400 步，每 20 步记一次，第 200、400 步各做一次 2 局仿真评测）：事件文件中有 13 条训练曲线、14 条评测曲线，
`tensorboard` 读回的步数与日志一致。做法沿用上游 PR #154（未合并）：使用 torch 自带的 `SummaryWriter`；`tensorboard` 已包含在 `training` 组中。

## 远程推理：观测压缩 JPEG、后台发送（默认开启）

上游 `robot_client` 在控制循环中将两路原图 pickle 后同步发送给服务端（640×480 两路约 1.8 MB/次），
发送完成后才执行下一个动作 —— 百兆网口上运行不流畅，在云服务器 5 Mbps 上行带宽下几乎无法发送。本分支改为：

- 相机画面压缩为 JPEG 后发送（`--observation_jpeg_quality=90`；设为 0 则发送原图）；
- 控制循环只将最新一条观测放入待发送位，由后台线程压缩并发送（`--send_observations_in_background=true`）；
  发送较慢时旧观测被新观测替换，服务端始终获得最新画面。

两项默认开启，命令无需修改。RK3566 端侧板 ↔ server 实测（ACT，两路 640×480 相机，70 s，两条线结果相同）：

| | 动作频率 | 动作间隔 p95 | 端侧板平均上行 |
|---|---|---|---|
| 原图同步发送，百兆局域网 | 16.6 Hz | 322 ms | — |
| JPEG + 后台发送，百兆局域网 | 29.4 Hz | 44 ms | 1.4 Mbps |
| 原图同步发送，上行限 5 Mbps | 3.9 Hz | 3 s | 2.4 Mbps |
| JPEG + 后台发送，上行限 5 Mbps | 29.5 Hz | 44 ms | 1.0 Mbps |

⚠️ 新服务端可接收旧客户端发送的原图；新客户端（默认发送 JPEG）无法连接旧服务端 —— 端侧板与服务端须同时升级。

## 端侧板录数据：硬件编码、统计抽样、网页预览（端侧板上默认生效）

在 RK3566 这类端侧板上，上游的数据录制有三处性能不足：软件 AV1 编码约 11 fps；边录边编码时逐帧计算图像统计，每路每帧约 15 ms；
`--display_data` 需启动原生 rerun 查看器，而端侧板不开启桌面、学员只有浏览器可用。本分支在端侧板上自动修改这三处，命令无需添加参数：

| 项 | 端侧板上的默认 | 其他机器 | 开关 |
|---|---|---|---|
| 视频编码 | 默认的 libsvtav1（以及 `auto`）自动切换为芯片自带的 MPP 硬件 H.264（`h264_rkmpp`，经 GStreamer，带 `GST_MPP_NO_RGA=1`） | 不变 | `LEROBOT_RKMPP=0` 关闭切换；也可显式 `--dataset.rgb_encoder.vcodec=...` |
| 图像统计 | 边录边编码时每 10 帧计算一次（前两帧必算） | 逐帧（不变） | `--dataset.rgb_encoder.stats_every=N` |
| 预览 | `--display_data=true` 且无桌面时，端侧板自行提供网页：用浏览器打开 `http://<板子>:9090`（仅需输入该短地址） | 有桌面时仍使用原生 rerun | `--display_mode=web` 显式指定；`LEROBOT_WEB_PREVIEW_HZ`（默认 10）、`LEROBOT_WEB_PREVIEW_PORT`（默认 9090） |

实测（RK3566 4 GB，两路 640×480 MJPG 相机，`lerobot-record` 40 s，`--dataset.streaming_encoding=true --display_data=true`，
不添加任何端侧板专用参数）：自动选用 MPP，预览自动改为网页，共录制 1152 帧，平均 28.8 fps，读回正常，图像统计正常；
网页预览 10 次/秒，电脑上仅需打开 Firefox 并输入短地址，即可显示两路画面与关节曲线。

- MPP 判定：ARM + `/proc/device-tree/compatible` 含 rockchip + 已安装 `gst-launch-1.0` 与 GStreamer 的 `mpph264enc` 插件；
  端侧板上安装 `gstreamer1.0-tools` 与瑞芯微的 GStreamer 插件包即可。
- `GST_MPP_NO_RGA=1` 不可省略：插件默认借助 RGA 搬运帧，端侧板有一段物理内存位于 4 GB 以上，RGA 访问该区域会写坏内存，导致整机停止响应。
- 每 10 帧计算一次的统计与逐帧计算相比：均值差 ≤0.00025、q99 差 ≤0.012。
- 网页预览要做到只靠浏览器，需要两项处理：9090 是跳转页（按浏览器访问端侧板所用的主机名拼出数据地址，再跳转到 9091 的查看器）；
  rerun 0.3x 的数据服务默认只放行 localhost 页面的跨源请求，此处放行查看器端口。推送在子进程中进行，不拖慢录制。
- 录数据建议添加 `--dataset.streaming_encoding=true`（边录边编码，存盘几乎无需等待）；不添加时画面先存为 PNG，每段结束后再用 MPP 编码。
- 「是否有桌面」按显示服务是否正在运行来判定：本机显示 `DISPLAY=:N` 需存在 `/tmp/.X11-unix/XN`，Wayland 需存在其 socket；远端显示（如 SSH 转发的 `localhost:10.0`）直接采信。端侧板厂家镜像在每个登录 shell 中固定设置 `DISPLAY=:0`，桌面关闭后仍保留该设置，而网页终端属于登录 shell —— 仅检查变量会去启动原生查看器；启动失败时 rerun 会把每帧原图缓存在内存中，一分多钟即耗尽 4 GB 内存而被系统终止。录制时的按键监听（`n` / `r` / `q`）使用同一判定，因此在网页终端中按键方能生效。
- GStreamer 编码子进程的标准输出重定向到 `/dev/null`：连接终端时 `gst-launch-1.0` 每秒输出多行管道位置信息（`-q` 无法关闭），会使网页终端输出大量信息。

## LingBot-VLA 2.0（`lingbot_vla_v2`，只在 0.6 线）

蚂蚁灵波（Robbyant）开源的 6B VLA：Qwen3-VL-4B 骨干 + 稀疏 MoE 的 Qwen2 动作专家，flow matching 生成动作块，
各类机器人统一映射到 55 维状态 / 动作槽位。策略代码取自上游 LeRobot **尚未合并**的 PR
[#4897](https://github.com/huggingface/lerobot/pull/4897)（head `de61f262`，官方维护者在社区移植 #3967 的基础上重新实现，
权重已逐张量与原版核对）；转换后的权重直接使用上游发布的 `lerobot/lingbot_vla_v2_base`（微调起点）与
`lerobot/lingbot_vla_v2_robotwin`。PR 原样移入之后，本线仅修改了训练部分（见下「训练改为官方配方」），推理与模型结构未改动。

**训练改为官方配方（2026-10-21，`cf4c8bcd`）。** PR 原版将整个模型转为 bf16 后直接用 AdamW 更新：学习率为 1e-5 时，
更新量不足 bf16 一个精度单位的一半，原版权重中约 92% 的动作专家参数完全无法更新（外部用户按手册训练，loss 停在 0.2 左右）。
官方训练一律保留 fp32 参数。本线改为：

- 优化器 `lingbot_muon_adamw`：矩阵参数（含三维 MoE 专家堆叠，逐专家正交化）使用 Muon，其余使用 AdamW，与官方
  `lingbotvla/optim/muon.py` 一致；每个低精度可训练参数在优化器中持有 fp32 主副本，更新作用于主副本后再写回 bf16，
  主副本随 `training_state` 保存与读取，续训不损失精度；基础权重为 fp32 时，主副本直接以文件中的原值初始化。
- 新配置的默认值沿用官方 `robotwin.yaml`：lr 1e-4 余弦衰减到 5e-5、不预热、状态与动作 `QUANTILES`（= 官方 `bounds_99_woclip`）。
  ⚠️ 从 `lerobot/lingbot_vla_v2_base` 微调时，其 config.json 中保存的旧值（1e-5 恒定、预热 1000、`MEAN_STD`）仍然生效，
  需在命令行用 `--policy.normalization_mapping` / `--policy.optimizer_lr` / `--policy.scheduler_*` 覆盖；新增的
  `optimizer_type`（默认 `muon`）与 `master_weights`（默认开启）无需指定。
- `lerobot-train --gradient_accumulation_steps=N`：沿用上游 main 的语义，`steps` / `log_freq` / `save_freq` / 学习率调度均按小批次计数。
- 多卡使用 `accelerate launch --multi_gpu`（DDP）。不要使用 FSDP：Muon 需对整块矩阵做正交化，FSDP 切分后的参数无法正确计算。
- RoboTwin 2.0 clean 数据的转换（口径与官方训练一致）与留出集动作误差：`examples/lingbot_vla_v2_robotwin/`。

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
  --policy.normalization_mapping='{"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}' \
  --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=5e-5 --policy.scheduler_warmup_steps=0 --policy.scheduler_decay_steps=30000 \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.push_to_hub=false --batch_size=1 --steps=30000
```

⚠️ 必须指定 `--policy.device=cuda`：`lerobot/lingbot_vla_v2_base` 的 config.json 中记录的是 `device: cpu`，不指定则在 CPU 上训练，不报错，但每步需十几秒。

显存不足时添加 `--policy.train_expert_only=true --policy.gradient_checkpointing=true`（仅训练动作专家）。
微调得到的模型照常由 `policy_server` 加载，端侧板上运行 `robot_client --policy_type=lingbot_vla_v2`。
完整参数、推理加速开关见 `docs/source/lingbot_vla_v2.mdx`。

本线上实测（2026-10-18，server 与其他任务共用；数据 = SO-101 真机「把方块放进盒子」300 集，5 关节 + 夹爪，`top` / `wrist`）：

| 项 | 结果 |
|---|---|
| 基础权重按 SO-101 槽位加载、真实帧输出动作 | 6 维、数值均有限；GPU 每块 50 步约 0.9 s（fp32、未开启加速），CPU 约 10 s |
| `lerobot-train` 短微调（仅训练动作专家 + 梯度检查点 + bf16 + batch 1，蒸馏开启，200 步） | 每步约 0.55 s、显存 26.9 GB；四项蒸馏损失持续下降（深度 3.23→1.03 等），动作损失在该量级下未见明显下降 |
| 微调出的检查点 → `policy_server` → RK3566 端侧板 `robot_client`（端侧板未升级、不加参数） | 动作 29.0 Hz、间隔中位 33.5 ms / p95 45 ms；服务端单次推理中位 0.29 s（bf16）、加载 60 s |

上游 PR 依赖 v0.6.1 之后 main 上的几处改动，本线仅补入其用到的部分，均沿用上游实现；除下文「保存格式的变化」一条外，其他策略的行为不变：

| 补的 | 上游出处 | 作用 |
|---|---|---|
| `flow_matching.FlowConvention` / `make_flow_matching_inputs` | #4075、#4631 | 训练时构造加噪动作与速度目标（其他策略仍沿用各自原写法） |
| 处理流水的「产物」（`ProcessorStep.save_artifacts`、加载时 `_resolve_artifact_paths`） | #4182 | 权重目录里带的 `image_processor/`、`chat_template_tokenizer/`、`tokenizer/` 按相对路径保存，加载时就地解析，从 Hub 加载时自动下载这些子目录 |
| `TokenizerProcessorStep.save_artifacts` | #4182 | 分词器随前后处理一起保存到权重目录 |
| `load_pretrained_policy_processors`、按约定调用 `make_<类型>_pre_post_processors_from_pretrained` | #4701 | 从权重加载前后处理时，由策略自行替换为新机器人的槽位映射 |
| `PreTrainedConfig` 注册 `torch.dtype` 的 draccus 编解码（不新增基类字段） | main | 策略配置中的 `dtype` 可保存与读取 |
| `resolve_delta_timestamps` 认按模态给的帧序（`image_observation_delta_indices` 等）与 `rename_map` | main | 训练时给每路相机取当前帧与 49 帧后的帧（深度 / 视频蒸馏需要）；未定义这些属性的策略不受影响 |
| `async_inference` 白名单加 `lingbot_vla_v2` | — | 推理服务端可加载该策略 |
| 远程推理认得训练时的相机改名 | 本线自行修复（上游 main 已删除整个 `async_inference`） | 见下 |

**远程推理与相机改名。** 上游 0.6.1 的 `policy_server` 无条件使用客户端发来的改名表覆盖权重中保存的那份，
而 `robot_client` 从不填写（恒为空）；缩放图像时又按策略自身的相机名查找尺寸。因此**训练时使用过 `--rename_map` 的策略均无法远程推理**
（LingBot-VLA 2.0 的相机必须命名为 `camera_top` / `camera_wrist_left`，服务端每帧报 `KeyError: 'observation.images.top'`）。
本线改为：仅当客户端确实提供改名表时才覆盖，否则使用权重中保存的；缩放按实际生效的改名表查找尺寸，图像仍由前处理完成改名。
未使用改名训练的策略行为不变，端侧板上的客户端无需升级。

⚠️ **保存格式有少许变化**：本线起，带分词器的策略（SmolVLA、π0 / π0.5 等）保存前后处理时，会将分词器一并保存到权重目录的
`tokenizer/` 子目录，并在配置中记录相对路径（与上游 main 一致）。以此方式保存的权重可由本线或上游 main 正常读取；用原版 0.6.1 读取则找不到分词器。
此前保存的权重不受影响。

许可证：代码与转换后的权重是 Apache-2.0。**训练**时的深度 / 视频蒸馏需要下载三个冻结的教师模型，其中 DINO-Video 来自 Meta 的 DINOv3，
受 DINOv3 License 约束（发表时须致谢 DINOv3，并限制部分用途）；**推理**不加载教师模型。

## 仿真器是独立仓

仿真器本体不在这里，在 [**Xbotics-SO101-Sim**](https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim)，
由本 fork 作为**核心依赖**安装。依赖方向单向：仿真包不 import lerobot，由 lerobot 识别该包。

```bash
lerobot-eval --env.type=so101_sim --env.task=SO101PickPlaceCube40-v1 \
             --env.control_mode=pd_joint_pos --eval.n_episodes=20
```

三个分发场景 `SO101PickPlace{Cube40,Cube20,Cylinder40}-v1`，**这就是全部环境 id**
（一个场景对应一个环境；RL 训练如需修改 reward 或放宽步数，通过包装器与构造参数实现，不新注册环境）。配套公开数据集
<https://huggingface.co/datasets/Harrysunshine/so101-sim-pickplace>。

### ⚠️ 评测口径必须与数据来源对齐

**默认值已全部对齐真机，正常评测无需再手动传入任何 `--env.*`。** 下列各项说明各默认值的依据
以及需要修改的情形 —— 任何一项配置错误都**不报错**，只会得到一个看似「策略没学会」
的低成功率。

| `--env.` 参数 | 默认 | 依据 |
|---|---|---|
| `unit_convention` | `real` | 真机口径（见下）；`maniskill` 为原生弧度 |
| `control_mode` | `pd_joint_pos` | 已发布数据集录制的是**绝对关节角** |
| `observation_width/height` | `640` / `480` | 相机竖直视野角由 ChArUco 在 640×480 下标定得到（fovy 59.17°） |
| `episode_length` | `500` | 已发布数据最长 444 帧，设得过短会静默截断 |
| `fps` | `30` | 仿真 `control_freq` 与数据帧率均为 30 |

#### 真机口径逐通道不同，并非统一的角度

真机数据由 `lerobot-record` 采集，经 `so_follower`，其**逐关节配置的归一化模式不同**：

| 通道 | 真机口径 | 出处 |
|---|---|---|
| 5 个臂关节 | **度** | `SOFollowerConfig.use_degrees` 默认 `True` ⇒ `MotorNormMode.DEGREES` |
| `gripper` | **0~100 行程百分比** | `so_follower` **固定**为 `MotorNormMode.RANGE_0_100`，与 `use_degrees` 无关 |

因此「统一到真机」不是单一的单位换算，而是**逐通道**换算。`unit_convention="real"` 即按此执行：
臂关节 弧度→度；夹爪 弧度→`(deg − deg_lo)/(deg_hi − deg_lo) × 100`（上下界取自夹爪关节自身的限位）。

**夹爪一项尤其容易出错**：度数与百分比的量级恰好重合（物理行程约 0~100 度），从数值上无法分辨，
只表现为**抓取环节无法学会**。校验点：仿真「张开到位」44.95° → 49.95%，真机实测 50.6%。

⚠️ 社区的 SO-101 数据集**并非都是这一口径**：官方 `lerobot/svla_so101_pickplace` 是归一化 ±100
（`action` 恰好达到 ±100.00，这是 clamp 的特征）。对齐只能以**自己那台真机**为基准。

上述问题均实际发生过：一次 ACT 评测得到 0%，一次 SmolVLA 六个 checkpoint 全为 0，
两次都是排查到最后才确认原因在口径而非策略。

## aarch64 板端（接机械臂的端侧板：瑞芯微 RK3566 / RK3576、地瓜 RDK 等）

仿真依赖带有平台标记 —— mani-skill 的物理后端 `sapien` 不提供 aarch64 轮子，板端不运行仿真，安装不受影响。
板端只负责舵机、相机、录数据与远程推理客户端，0.6 线需安装 **`lerobot[feetech,async,dataset,viz]`**
（0.6 起 datasets 与 rerun 不再属于核心依赖）；使用下文「网页工具」一节的三个网页时，另加 `xbotics_tools`。硬件编码、统计抽样、网页预览三项自动生效（见上一节）。另有几处需在端侧板自身的环境中处理：

- **torch**：上游 aarch64 轮子从 2.10 起在 ARMv8.0 核（A53 / A72）上一导入即报 `Illegal instruction`；
  这类端侧板固定 `torch==2.9.1`。A55（RK3566，ARMv8.2）上 2.10 实测可用。
- **torchcodec**：`dataset` 组在 aarch64 上要求 `>=0.11`，而那份 aarch64 轮子是针对 CUDA 13 编译的，
  在没有 NVIDIA 显卡的端侧板上必然加载失败（查找 `libnvrtc.so.13` 失败），输出大量报错后退回 PyAV ⇒ 端侧板环境中直接
  排除它（uv：`override-dependencies = ["torchcodec; sys_platform == 'never'"]`）。
- **rerun-sdk**：端侧板提供网页预览（学员仅用浏览器查看）时固定 `rerun-sdk==0.26.2`。0.33.1 的网页查看器在没有显卡加速的
  浏览器中一收到数据就停止响应；0.26.2 在同一浏览器中正常。`viz` 组声明的是 `>=0.24,<0.34`，0.26.2 在范围内。

BPU 上板推理用地瓜官方 [`rdk_LeRobot_tools`](https://github.com/D-Robotics/rdk_LeRobot_tools)，不在本 fork 内。

## 端侧板：安装与升级（RK3566）

端侧板上需维护的内容全部随 LeRobot 包发布（[`src/lerobot/xbotics/board/`](src/lerobot/xbotics/board/)），升级 LeRobot 即升级全部内容：

| 安装内容 | 包内位置 |
|---|---|
| 开机不进入桌面、CPU 与内存锁定最高频率、相机不自动降帧 | `system/`、`udev/` |
| 机械臂与相机的固定设备名 | `udev/99-xbotics-rk3566.rules` |
| 网络设置页（8001，热点配网 + 连接服务器） | `provision/` |
| 网页终端（8002）、JupyterLab（8003）、浏览器版 VS Code（8004，code-server 版本固定在安装器中）的配置与服务 | `jupyter/`、`vscode/`、`install.py` |
| 网页硬件检测（8005）、网页标定（8006）的服务及其开关命令 `xbotics-config` | `config/` |
| 端侧板的 LeRobot 环境 `/root/xbotics` 的 pyproject（固定 torch 2.9.1、rerun 0.26.2，不安装 torchcodec） | `board_env/` |

**新端侧板**（厂家 Ubuntu 22.04 镜像，root 登录后执行一行命令；按旧手册手工安装过的端侧板同样适用）：

```bash
curl -fsSL https://raw.githubusercontent.com/Xbotics-Embodied-AI-club/lerobot/xbotics-v0.6.1/src/lerobot/xbotics/board/bootstrap.sh | bash
```

它安装 uv、建立 LeRobot 环境，再由环境中的 LeRobot 运行 `python -m lerobot.xbotics.board install` 安装上表全部内容；首次安装会开启 8002–8006 五个网页服务。

**升级全部**：

```bash
xbotics-config update
```

获取标签当前指向的 LeRobot，再把上表内容重新安装一遍，仅修改有变化的部分：

- 保留 `xbotics-config` 中的服务开关；已设置的 JupyterLab、VS Code 密码不覆盖。
- 端侧板环境的 pyproject 只修改由维护版管理的几项（Python 版本、`lerobot` 与 `jupyterlab` 两个依赖、override-dependencies、LeRobot 的源），用 tomlkit 逐键修改：学员自行 `uv add` 的包、为 lerobot 添加的可选组、源、索引、依赖组原样保留；LeRobot 的源为标签时换成新版标签，指向分支或提交的测试端侧板不改动。
- 正在运行的服务有改动时才重启；JupyterLab、硬件检测、网页标定在 LeRobot 版本变化时重启，未变化则不动，不会无故打断正在运行的 notebook。
- 升级在单独的 systemd 单元中运行，重启延后 3 秒：从网页终端或 VS Code 终端中执行也不会中断升级本身，页面短暂断开后刷新即可。

实测（2026-11-02，ZTL A566，按 10-27 手册手工安装过的端侧板）：一键安装约 1 分钟（LeRobot 只替换一个包），六个网页服务照常运行、
页面更新为新版；之后 `xbotics-config update` 在 LeRobot 提交变化时重启上述三项，未变化时不重启任何服务，各约 20 秒。

## 端侧板：按接口位置固定设备名（RK3566）

端侧板上的 `/dev/videoN` 按枚举顺序分配，更换接口或重启都可能使其变化。
[`99-xbotics-rk3566.rules`](src/lerobot/xbotics/board/udev/99-xbotics-rk3566.rules) 按**设备插在端侧板的哪个接口**生成固定名字，
不依据序号，也不依据相机型号或序列号 —— 相机插在哪个接口，就使用该接口对应的名字，更换相机无需修改规则：

| 固定名字 | 是什么 | 匹配依据 |
|---|---|---|
| `/dev/xbotics/follower` | 从臂舵机（板载串口，系统中名为 `ttyS3`） | 串口控制器的设备树节点 `fe6b0000.serial` |
| `/dev/xbotics/leader` | 主臂舵机（板载串口，系统中名为 `ttyS1`） | 串口控制器的设备树节点 `fe6a0000.serial` |
| `/dev/xbotics/top` | 顶视相机 | USB 接口路径 `platform-xhci-hcd.4.auto-usb-0:1:1.0`（xhci-hcd.4 控制器的 1 号口） |
| `/dev/xbotics/wrist` | 腕部相机 | USB 接口路径 `platform-xhci-hcd.7.auto-usb-0:1:1.0`（xhci-hcd.7 控制器的 1 号口） |

规则识别的是**接口**，而非设备：主臂插到从臂的串口座上，即命名为 `follower`；顶视相机插到腕部的 USB 口上，即命名为 `wrist`。
设备拔出后，对应的名字随之消失。仅适用于 ZTL A566 这一款端侧板。

规则随 LeRobot 包发布，由上文「安装与升级」一并安装。之后命令中直接使用这些名字：
`--robot.port=/dev/xbotics/follower`、`--teleop.port=/dev/xbotics/leader`，相机 `index_or_path: /dev/xbotics/top`、`/dev/xbotics/wrist`。
`ls -l /dev/xbotics/` 列出当前已连接的设备，四样设备都连接时为 `follower -> ../ttyS3`、`leader -> ../ttyS1`、`top`、`wrist` 各指向一个 `videoN`。

实测（2026-10-19，ZTL A566）：通过固定名字，两个板载串口上的 6 个 STS3215 在 1 Mbps 下均全部读取成功（同步读位置 300 次，0 次失败，中位 1.3 ms）；
两路 USB 相机各输出 640×480、约 29 fps；相机拔出后其名字随之消失。

**更换端侧板型号**：这份规则仅识别这款端侧板的串口地址和 USB 控制器名。把设备接好，读出其位置，再仿照写一份：

```bash
udevadm info -q property -n /dev/videoN | grep '^ID_PATH='                 # 相机：取 index 为 0 的那个节点
udevadm info -a -n /dev/ttyS3 | grep -m1 'KERNELS==".*\.serial"'           # 板载串口
```

## 端侧板：网络设置页（RK3566）

网络设置页使端侧板持续开启一个热点 `Xbotics-XXXX`（后四位取自端侧板 Wi-Fi 网卡的地址，每块端侧板不同），
电脑或手机连接该热点后，系统会弹出网络设置页（未弹出时，在浏览器中打开 `http://10.42.0.1`）。页面可为端侧板连接 Wi-Fi、设置网线（自动获取或固定 IP），
网线、Wi-Fi、热点三者的状态以及能否访问外网每两秒刷新一次。热点不设密码、不通外网，仅供该页面使用；
同一页面也可在局域网中通过 `http://<板子IP>:8001` 访问。

- **热点为何持续开启**：AIC8800 可同时开启热点并连接 Wi-Fi（`iw list`：热点与连接 Wi-Fi 各一个接口，两者可在不同信道）。
  端侧板连接 Wi-Fi 时热点不中断，是否连接成功、获得的地址均可在页面上查看；配置有误时仍可通过热点进入页面修改。
- **弹出机制**：热点的 DNS 把所有域名都解析为端侧板自身（`xbotics-captive.conf`），页面服务将其他网址一律跳转到网络设置页；
  再关闭热点接口的转发，使连接的设备检测联网失败，系统便弹出该页面。
- 仅使用系统自带的 Python 3 标准库和 `nmcli`，另安装 `iw`，用于创建热点所用的虚拟接口 `ap0`。
- **端口**：套件端侧板上的四个网页服务使用连号端口，便于记忆 —— 8001 网络设置页、8002 网页终端（ttyd）、8003 JupyterLab、
  8004 浏览器版 VS Code（code-server）；页面上「从你自己的网络打开」一栏按这四个端口列出地址。
  80 端口同样提供该页面：连接热点的设备检测联网只访问 80 端口，弹出页面即依赖于此，因此 `http://10.42.0.1` 不写端口也可打开。

代码在 [`src/lerobot/xbotics/board/provision/`](src/lerobot/xbotics/board/provision/)，由上文「安装与升级」一并安装（仅使用系统 Python 与 `nmcli`，不进入 LeRobot 环境）。

NetworkManager 1.36 存在两个问题，代码中已规避：

- 连接失败时 `nmcli` 仍返回退出码 0 ⇒ 以网卡实际连接的是哪条连接判断成败。
- 对已保存的 Wi-Fi 直接 `nmcli device wifi connect … password …` 会改写旧配置的密码，输错一次后原本可用的配置也会失效 ⇒
  先用临时配置名 `xbotics-try` 试连，成功后替换旧配置，失败则删除临时配置并换回旧配置。

实测（2026-10-19，ZTL A566）：热点开启时连接 Wi-Fi、输错密码、删除 Wi-Fi、将网线改为固定 IP，热点均未中断；输错密码不影响旧配置；
携带 Android、Apple、Windows 联网检测地址的请求都跳转到网络设置页；重启后热点、页面、Wi-Fi 与网线设置均自动恢复。
2026-10-21 改为 80 与 8001 同时开启：两个端口都返回页面，联网检测请求在 80 端口上仍跳转到网络设置页。

## 端侧板：连接服务器（RK3566）

推理服务与 TensorBoard 运行在服务器上，端侧板连接它们。网络设置页（`http://<板子IP>:8001`）的「连接服务器」一栏填写服务器地址、SSH 端口、账号，
登录方式二选一：**账号密码**，或**私钥**（整段粘贴或选择文件，可带口令）。点击「保存并连接」后，端侧板通过 SSH 隧道把服务器上的端口映射到自身的 **0.0.0.0**：

| 默认映射 | 端侧板上 | 服务器上 |
|---|---|---|
| 推理服务（`policy_server`） | `0.0.0.0:8080` | `127.0.0.1:8080` |
| TensorBoard | `0.0.0.0:6006` | `127.0.0.1:6006` |

端侧板上的程序连接 `127.0.0.1:8080`（`robot_client`、推理控制台的 `--server_host=127.0.0.1` 保持不变），同一网络中的电脑打开 `http://<板子IP>:6006` 即可查看训练曲线。
映射可增删；服务器端口写成 `地址:端口` 时，转发到服务器可访问的另一台机器。页面为每条映射标出「畅通 / 服务器上该端口无服务」，网页类映射提供「打开」链接。

- 隧道即 OpenSSH 客户端的本地转发 `ssh -N -L 0.0.0.0:<板子端口>:<地址>:<端口>`，由 systemd 服务 `xbotics-tunnel` 启动：断开 5 秒后重连，
  `ServerAliveInterval` 使断网在 30 秒内被发现；设置完成后开机自动连接，「断开」按钮同时取消开机自动连接。
- 密码与私钥口令经 OpenSSH 自带的 `SSH_ASKPASS` 交给 ssh（OpenSSH 8.4 起），每次重连均可自动登录，无需 sshpass。设置保存在 `/etc/xbotics/`，仅 root 可读。
- ssh 的英文报错译为中文说明显示在页面上：登录被拒、私钥口令不对、地址找不到、端口不对、端侧板端口被占用、服务器不允许转发（`AllowTcpForwarding`）等。
- 账号、密码或私钥不对时不再重试（持续使用错误密码尝试，会使服务器上的 fail2ban 等机制锁定账号），修改正确后再点击「保存并连接」；每次保存都会重新确认服务器指纹，因此服务器重装系统后也能连接。
- 每条映射的「畅通 / 服务器上该端口无服务」判定：端侧板连接自身的该端口但不发送任何数据，服务器端无服务时 ssh 会立即断开这条连接，有服务时连接保持开启，因此对 gRPC 推理服务也能准确判定。
- 服务器端只需开启 SSH 并允许 TCP 转发（OpenSSH 默认允许），推理服务与 TensorBoard 按默认仅监听 127.0.0.1 即可，无需对外开放端口。

实测（2026-11-02，ZTL A566 ↔ 同网段的一台 aarch64 机器作为服务器）：在网页上填写账号密码并保存，约 3 秒连接成功，`0.0.0.0:8080`、`0.0.0.0:6006` 由 ssh 监听，
其他电脑经 `http://<板子IP>:6006` 打开了服务器上仅监听本机的 TensorBoard；换成带口令的 ed25519 私钥同样连接成功（服务器日志 `Accepted publickey`）；
终止 ssh 进程后 6.5 秒自动重新连接；输错密码时页面显示「登录被拒：账号、密码或私钥不正确」，隧道服务停止、不再重试（50 秒内重试次数为 0）。

## 网页工具：硬件检测、标定、推理控制台

端侧板不开启桌面，学员只有浏览器可用。下列三个工具在端侧板（或任何连接机械臂的机器）上运行，
学员在自己电脑上打开 `http://<板子IP>:<端口>`；端口接续端侧板上 8001–8004 的连号，默认监听全部网卡。
套件端侧板上，硬件检测与网页标定安装为服务，用 `xbotics-config` 开关（见下「端侧板上的开关」）；在其他机器上直接输入命令启动，按 Ctrl+C 关闭。
推理控制台需要指定服务端地址和模型，因此总是按需输入命令启动。移植自 gitee [`zhuzhuangtian/lerobot-v0.5.1-custom`](https://gitee.com/zhuzhuangtian/lerobot-v0.5.1-custom/tree/master/lerobot_tools)
的 `lerobot_tools`，已按本线修改，具体改动写在各自的模块说明中。

套件端侧板上由「端侧板：安装与升级」安装完成；其他机器上将 `xbotics_tools` 组加入 LeRobot 依赖（`lerobot[feetech,async,xbotics_tools]`）。

| 命令 | 端口 | 做什么 |
|---|---|---|
| `lerobot-hardware-check` | 8005 | 每个 USB 相机显示一路实时画面（标出协商得到的分辨率、格式、帧率；芯片自带的视频节点默认不列出，`--all-cameras` 全部列出）；串口列表与「插拔对比」用于查找机械臂串口（板载串口有固定名字时也一并列出）；每条串口提供一个「读电机」按钮：扫描总线上的舵机，读取一次位置、电压、温度、电流、负载、力矩开关、状态码与报警（过载、过热的舵机同样列出），只读不写 |
| `lerobot-calibrate-web` | 8006 | SO101 主臂 / 从臂网页标定：3D 模型演示每一步的摆放方式，逐关节记录行程、单独校验，可单关节重做；写入后回读核对；取消或失败时自动恢复标定前的备份，出错的会话不阻塞新会话 |
| `lerobot-inference-console` | 8007 | 远程推理客户端 + 网页控制台：相机画面、动作队列可视化、推理延迟；网页上可修改任务、启停、实时调整 RTC 参数 |

### 端侧板上的开关：`xbotics-config`

类似树莓派的 `raspi-config`：不带参数时弹出菜单，用空格勾选要开启的网页服务、回车生效（开启的服务开机自启）；也可直接输入子命令。
管理 8002 网页终端、8003 JupyterLab、8004 VS Code、8005 硬件检测、8006 网页标定五项；8001 网络设置页始终开启，不在此管理（关闭后将无法从热点进入端侧板修改网络）。

```bash
xbotics-config                            # 菜单（要 whiptail）
xbotics-config status                     # 各服务是否开机自启、是否正在运行及网址
xbotics-config enable hardware-check calibrate
xbotics-config disable calibrate          # 名字：terminal jupyter vscode hardware-check calibrate，all = 全部
```

`xbotics-config` 与两个服务文件由「端侧板：安装与升级」安装完成，`xbotics-config update` 升级全部内容。

- 在网页终端中关闭网页终端会使当前页面断开，菜单会先请求确认一次。
- 两个服务空闲时各占用约 290 MB 内存（LeRobot 同时导入 torch），两者均开启时端侧板仍有约 3 GB 可用（RK3566 4 GB 实测）。
- 硬件检测读取电机、查看相机画面时，与正在使用同一条臂或同一路相机的程序（录数据、推理、网页标定）会相互干扰：
  串口与相机均不加锁，需由使用者自行避开 —— 录数据、推理前请关闭查看相机的页面，录制时不要点击「读电机」。

### 网页标定

```bash
lerobot-calibrate-web                                     # 端侧板上：串口默认取 /dev/xbotics/leader、/dev/xbotics/follower，id 默认 leader、follower
lerobot-calibrate-web --follower-port /dev/ttyACM0 --follower-id my_follower --leader-id my_leader
lerobot-calibrate-web --demo                              # 不接硬件，把流程走一遍（模拟舵机，不碰标定文件）
```

- 步骤与 `lerobot-calibrate` 相同（卸力 → 切位置模式 → 中位记录 homing offset → 除腕部旋转外记录行程 → 写入），
  标定文件也写在同一位置（`$HF_LEROBOT_CALIBRATION/{robots,teleoperators}/<类型>/<id>.json`）——
  之后录数据、推理时，`--robot.id` / `--teleop.id` 指定同一个 id 即可直接使用。
- 与命令行的差别：逐个关节录制，每个关节单独校验（行程是否达到 50 个码值、中位是否在范围内），哪个不合格只重做哪个；
  开始前把舵机中的标定和原标定文件备份到标定文件旁的 `backups/`，取消或写入回读不一致时恢复。
- 中文语音提示使用微软 edge-tts 在线合成（需联网），无法连接时页面只显示文字，标定仍可完成；已合成的句子缓存在 `$HF_LEROBOT_HOME/calibration_tts/`。
- 3D 模型随包发布：从臂使用星禾套件的仿真模型（与 [Xbotics-SO101-Sim](https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim) 为同一份，并同时显示底板、型材架、顶视与腕部相机支架；半透明的目标姿态只绘制手臂），主臂使用官方 SO101 主臂模型。`.gitattributes` 使这些 STL 不经 LFS 管理——经 LFS 管理时，从 git 安装得到的是指针文件，页面无法加载模型。
- 修改前端：在 `xbotics/calibrate_web_frontend/` 中执行 `npm ci && npm run build`，产物直接写入 `src/lerobot/xbotics/calibrate_web/static/`，一并提交。

### 推理控制台

服务端仍为 `python -m lerobot.async_inference.policy_server`；端侧板上用它替代 `robot_client`：

```bash
lerobot-inference-console \
    --robot.type=so101_follower --robot.port=/dev/xbotics/follower --robot.id=follower \
    --robot.cameras="$CAMERAS" \
    --task="Grab the cube" \
    --policy_type=act --pretrained_name_or_path=<模型> --policy_device=cuda \
    --server_host=127.0.0.1 --server_port=8080 --actions_per_chunk=50 --fps=30
```

`$CAMERAS` 与连接服务器的 SSH 隧道按端侧板使用手册第 8.1、10.2 节配置；`--server_host=127.0.0.1` 即经由该隧道。

- 相机画面默认压缩为 JPEG 后发送（`--observation_jpeg_quality=90`，与 `robot_client` 为同一实现；设为 0 则发送原图），发送在后台线程中进行。
- **RTC**（默认开启，`--rtc_enabled`）：按最近几次推理延迟估计何时需要下一个动作块，使其在旧块用完前到达；
  新旧块重叠部分的开头 `--rtc_blend_steps` 步线性过渡；执行层做 EMA 平滑（`--rtc_action_ema_alpha`，夹爪不平滑）；
  推理较慢时，可用 `--action_interp_steps=N` 把每个动作插值为 N 个子步，控制循环仍保持高频运行。这些参数均可在网页上修改，下一个控制周期生效。
- 在网页上点击「停止任务」时，动作停止、观测照常发送，服务端返回的动作块被丢弃；指定了 `--home_action='{shoulder_pan.pos: 0, ...}'` 时同时回到该姿态。
  在 `--task_presets_file=<yaml>` 中写入 `tasks: {名字: 任务文本}`，即可在网页下拉框中选择。
- `--transport=websocket` 改为连接 LingBot-VLA 官方推理服务（msgpack 协议，`--ws_host` / `--ws_robo_name` 必填，模型由服务端决定，无需指定 `--pretrained_name_or_path`；每条观测只推送一次）；
  本线自带的 LingBot（`--policy_type=lingbot_vla_v2`）使用默认的 gRPC 即可。
- 控制台无登录验证，监听全部网卡时，同一网络中的任何人都能打开；仅在本机查看时指定 `--webui_host=127.0.0.1`。

### policy_server 防停滞

上游的 policy_server 在两种情况下会使客户端始终收不到动作块：① 收到观测就先记录「这一步已推理」，推理一旦失败，动作块丢失，
客户端仍停在这一步，重发的观测又被「已推理」挡掉；② 机械臂不动时每帧都与上一帧相似，全部被相似度检查滤掉，
客户端那条必须推理的观测（must_go）一旦丢失，就再也没有新动作块。现在改为推理和序列化都成功后才记录「已推理」，
推理失败时把「上次推理过的观测」还原为失败前那一帧（否则静止时重发的观测与失败帧相似，仍会被滤掉）；
超过 `--similarity_stall_timeout`（默认 5 秒）没有观测入队时，把最新一帧强制放入。对 `robot_client` 与推理控制台均生效。
单次推理超过 5 秒（CPU 推理、首次加载）时，兜底机制会使推理接连运行，仅造成浪费，不会出错；这种情况下应调大该值。

## 从 0.5 线迁到 0.6 线要改什么

上游 0.5.1 → 0.6.1 之间有 321 个 commit。Xbotics 自有代码（工作坊教学仓、DM0.5 插件、套件实验、讲义）
用到的 76 个 LeRobot 导入中，0.6 线只缺少 `add_envs_task`；其余变化体现在用法上：

| 变化 | 0.5 线 | 0.6 线 |
|---|---|---|
| 依赖 | datasets / accelerate / diffusers 是核心依赖 | 移入可选组：`dataset`、`accelerate-dep`、`diffusers-dep`；rerun 在 `viz`；transformers 在 `transformers-dep`；`lerobot-eval` 需安装 `evaluation` **和** `dataset`（上游 `evaluation` 组只有 av，评测脚本却无条件导入数据集） |
| 录数据选编码器（命令行） | `--dataset.vcodec=h264` | `--dataset.rgb_encoder.vcodec=h264` |
| 录数据选编码器（Python） | `LeRobotDataset.create(..., vcodec="h264")` | `create(..., rgb_encoder=RGBEncoderConfig(vcodec="h264"))`（`lerobot.configs.video`） |
| 模块路径 | `lerobot.types` | `lerobot.lerobot_types`；`PreTrainedConfig` 可从 `lerobot.configs` 导，`get_policy_class` / `make_pre_post_processors` 可从 `lerobot.policies` 导 |
| 环境里取任务指令 | `lerobot.envs.utils.add_envs_task` | 已删除，改用 `env.call("task_description")` |
| 策略的注册 | `factory.py` 里逐个 if/elif | 配置类 `@PreTrainedConfig.register_subclass` + 命名约定（`XConfig` → `modeling_*.XPolicy` / `processor_*.make_x_pre_post_processors`），内置策略依靠 `lerobot/policies/__init__.py` 导入配置类完成注册 |
| GR00T | N1.5 | N1.7 |
| rerun-sdk | 0.26.2 | 0.33.1（浏览器 / 拉流端须使用同一大版本；端侧板网页预览固定 0.26.2，见上） |
| torch（锁文件） | 2.10 | 2.11（ARMv8.0 端侧板见上一节） |

未变的有：机器人 / 相机 / 第三方插件的注册方式；`robot_client` 与 `policy_server` 的命令行参数
（工作坊那组实测两条线通用）。边录边编码的统计在 0.6 线按平台抽样（端侧板每 10 帧，其余逐帧），0.5 线仍逐帧。

⚠️ **端侧板与推理服务端必须使用同一条线、同一个标签**：两端经 gRPC 互传 pickle 后的 LeRobot 对象，
0.5 客户端连接 0.6 服务端（或反过来）尚未验证；JPEG 上传要求两端均为包含该功能的版本（工作坊升级时两端同时换到 `xbotics-v0.6.1`）。

## 跟上游的关系

`upstream` = `huggingface/lerobot`。本分支是在上游 `v0.6.1` 上逐个重放 0.5 线的 commit 得到的
（不 squash、无合并提交）；移植时改写过的 commit，提交说明中写明了修改内容与原因。
上游发布新版时，按同样的办法新建分支，0.5 / 0.6 两条线均保留。
