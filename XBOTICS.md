# 这个 fork 加了什么

Xbotics 维护的 lerobot 分支。**基线 `v0.5.1`**，工作分支 `main`。

装它：

```bash
pip install "lerobot[all] @ git+https://github.com/Xbotics-Embodied-AI-club/lerobot.git@main"
```

功能是**一起装的**，没有「挑几个」这回事 —— 下面四项在 `main` 分支上一律可用。

## 四项改动

| commit | 内容 |
|---|---|
| `fix(groot)` | `GR00TN15Config` 去掉 `@dataclass`。HF 的 `PretrainedConfig` 子类不能再套 `@dataclass`，两边都要接管 `__init__`，实例化即崩。上游 v0.5.0→v0.5.1 未修。**这条是纯 bugfix，与我们的业务无关，可以提回上游。** |
| `feat(policies)` | **VLA-0**（`--policy.type=vla0_smol`）。动作离散成整数当文本 token 让 VLM 自回归输出，VLM 本体零改装，xgrammar 约束解码。骨干 SmolVLM2，512 bin。 |
| `feat(policies)` | **OpenVLA**（`--policy.type=openvla`）。7B prismatic，256 bin 动作 token，`forward()` 自带 CE loss 所以训练路径即原生 SFT。 |
| `feat(envs)` | **SO-101 仿真评测口**（`--env.type=so101_sim`）。 |

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

## aarch64（地瓜 RDK S100 / S600 板端）

仿真依赖带了平台标记 —— mani-skill 的物理后端 `sapien` 不发 aarch64 轮子。
板端走 `lerobot[feetech]` 只做舵机与相机，不跑仿真，安装不受影响。
BPU 上板推理用地瓜官方 [`rdk_LeRobot_tools`](https://github.com/D-Robotics/rdk_LeRobot_tools)，
不在本 fork 内。

## 跟上游的关系

`upstream` = `huggingface/lerobot`。上游当前已到 v0.6.1，我们**有意停在 v0.5.1**：
全部教学与实验代码都按这个版本验证过。v0.6.1 含破坏性改名（`lerobot.types` →
`lerobot.lerobot_types`），要跟的话得单开一条分支重验。

四个 commit 各自独立、按上面表格的顺序摞在 `v0.5.1` 上；两个 policy 的 commit
都动 `factory.py`，顺序不可换。
