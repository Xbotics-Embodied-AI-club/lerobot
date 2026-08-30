# 这个 fork 加了什么

Xbotics 维护的 lerobot 分支。**基线 `v0.5.1`**，工作分支 `xbotics`。

装它：

```bash
pip install "lerobot[all] @ git+https://github.com/Xbotics-Embodied-AI-club/lerobot.git@xbotics"
```

功能是**一起装的**，没有「挑几个」这回事 —— 下面四项在 `xbotics` 分支上一律可用。

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

三个分发场景 `SO101PickPlace{Cube40,Cube20,Cylinder40}-v1` + 三个 `...Train-v1`
RL 训练孪生。配套公开数据集
<https://huggingface.co/datasets/Harrysunshine/so101-sim-pickplace>。

### ⚠️ 评测口径必须与数据来源对齐

`--env.control_mode` 与 `--env.episode_length` 要和「被评策略所训数据是怎么产生的」一致。
不一致**不报错**，只会安静地跑错，最后给出一个看着像「策略没学会」的低成功率：

- 已发布数据集录的是**绝对关节角** ⇒ 必须 `--env.control_mode=pd_joint_pos`。
  不给的话用机器人默认的归一化增量模式，绝对角（约 −2.0~1.6 rad）被当增量喂进去、
  每维 clip 到 ±1，手臂以包线最大速度朝错误方向走。
- `episode_length` 要装得下轨迹长度，否则策略在完成动作前被 TimeLimit 截断。

这两条都真实踩过：一次 ACT 评测拿到 0%，排查到最后才发现是口径而不是策略。

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
