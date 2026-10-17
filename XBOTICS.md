# 这个 fork 加了什么

Xbotics 维护的 lerobot 分支。这里是 **0.6 线：基线 `v0.6.1`，分支 `xbotics-0.6`**。

## 两条版本线，按需选

| 线 | 分支 | 标签 | 基线 | 谁在用 |
|---|---|---|---|---|
| 0.5 线 | `main` | `xbotics-v0.5.1.post1` | 上游 `v0.5.1` | 工作坊学员机、套件板子现在钉的版本 |
| 0.6 线 | `xbotics-0.6` | `xbotics-v0.6.1.post1` | 上游 `v0.6.1` | 新项目；Xbotics 的改动都已搬过来并重新验证 |

```bash
# 克隆：-b 选线（分支跟着修，标签钉死不动）
git clone -b xbotics-0.6 https://github.com/Xbotics-Embodied-AI-club/lerobot.git    # 0.6 线
git clone -b main        https://github.com/Xbotics-Embodied-AI-club/lerobot.git    # 0.5 线

# 当依赖装（uv 项目的 pyproject.toml 里写；要复现就钉标签）
#   lerobot = { git = "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", tag = "xbotics-v0.6.1.post1" }
#   lerobot = { git = "https://github.com/Xbotics-Embodied-AI-club/lerobot.git", tag = "xbotics-v0.5.1.post1" }
pip install "lerobot[all] @ git+https://github.com/Xbotics-Embodied-AI-club/lerobot.git@xbotics-v0.6.1.post1"
```

标签只增不改：`.post1` 是在 `xbotics-v0.5.1` / `xbotics-v0.6.1` 之后加了远程推理 JPEG 上传（及 0.6 线一处评测修复）的版本，旧标签原样保留。

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
（0.6 起 datasets 与 rerun 都不在核心依赖里了）。另有两处要在板子自己的环境里处理：

- **torch**：上游 aarch64 轮子从 2.10 起在 ARMv8.0 核（A53 / A72）上一导入就 `Illegal instruction`；
  这类板子钉 `torch==2.9.1`。A55（RK3566，ARMv8.2）上 2.10 实测可用。
- **torchcodec**：`dataset` 组在 aarch64 上要求 `>=0.11`，而那份 aarch64 轮子是对着 CUDA 13 编的，
  没有 NVIDIA 显卡的板子上必然加载失败（找 `libnvrtc.so.13`）、刷一屏报错后退回 PyAV ⇒ 板子环境里直接
  排除它（uv：`override-dependencies = ["torchcodec; sys_platform == 'never'"]`）。
- **rerun-sdk**：板子出网页预览（学员只用浏览器看）时钉 `rerun-sdk==0.26.2`。0.33.1 的网页查看器在没有显卡加速的
  浏览器里一收到数据就卡死；0.26.2 同一浏览器正常。`viz` 组声明的是 `>=0.24,<0.34`，0.26.2 在范围内。

BPU 上板推理用地瓜官方 [`rdk_LeRobot_tools`](https://github.com/D-Robotics/rdk_LeRobot_tools)，不在本 fork 内。

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
（工作坊那组实测两条线通用）；边录边编的逐帧统计（0.6 线仍是每帧都算，弱 CPU 板子上照旧是瓶颈）。

⚠️ **板子与推理服务端必须用同一条线、同一个标签**：两端经 gRPC 互传 pickle 后的 LeRobot 对象，
0.5 客户端连 0.6 服务端（或反过来）没有验证过；JPEG 上传也要两端都是 `.post1` 及以后。

## 跟上游的关系

`upstream` = `huggingface/lerobot`。本分支是在上游 `v0.6.1` 上逐个重放 0.5 线的 commit 得到的
（不 squash、无合并提交）；移植时改写过的 commit，提交说明里写了改了什么、为什么。
上游再出新版时照同样的办法开新分支，0.5 / 0.6 两条线都保留。
