# 第 3 章 硬件准备

本章将两条机械臂与两路相机连接到端侧板，确认硬件工作正常，完成机械臂标定，最后用遥操作检查一遍。标定决定了舵机读数与真实关节角度之间的对应关系，后续录制的数据与部署的模型都依赖它。本章的命令均在端侧板的网页终端中执行。

## 3.1 接线

两路相机接左上角的双 USB 口，两条机械臂接下沿的两个串口座。

![接口图：wrist 与 top 为两路相机，leader 与 follower 为主从两条机械臂](images/ports-board.jpg)

| 设备 | 接口 | 命令中的写法 |
| --- | --- | --- |
| 从臂（执行动作的一条） | 下沿丝印 TTYS3 的串口座 | `/dev/xbotics/follower` |
| 主臂（操作者握持的一条） | 下沿丝印 TTYS1 的串口座 | `/dev/xbotics/leader` |
| 顶视相机 | 双 USB 口右侧 | `/dev/xbotics/top` |
| 腕部相机 | 双 USB 口左侧 | `/dev/xbotics/wrist` |

名字按接口识别，而不按设备识别。两只相机的接口插反时，名字随之互换，录制的顶视与腕部画面也会互换。检查方法：

```bash
ls -l /dev/xbotics/
```

四个名字均存在且各指向一个设备，即为正确。两个串口的名字始终存在；相机未连接时，其名字不会出现。

![四个固定名字均存在](images/use-02-devices.png)

## 3.2 舵机编号

机械臂已完成编号的，可以跳过本节。自行组装的机械臂需要为每个舵机写入编号，以下命令会提示你逐个连接舵机：

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/xbotics/follower
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/xbotics/leader
```

## 3.3 硬件检测

在浏览器中打开 `http://<端侧板IP>:8005`：

- **摄像头**：每路相机一张实时画面，标题为固定名字与分辨率。两路均有画面，且名字与接口图一致，即表示相机连接正确。
- **串口**：列出两条机械臂的串口与固定名字。
- **读电机**：点击串口所在行的「读电机」，下方列出该机械臂上每个舵机的位置、电压、温度、电流、负载与报警。6 个舵机均存在、状态码均为 0、报警均为「无」，即为正常，从臂电压约 12 V。该操作只读不写，机械臂不会运动。

报警一列不为「无」时，表示该舵机过载、过热或电压异常，请先断电，待其冷却后检查是否卡住，再重新上电。提示「没有扫到舵机」时，表示该机械臂未上电或未接好。

![硬件检测：上方为两路相机画面，下方为串口与从臂 6 个舵机的读数](images/use-12-hardware-check.png)

该页面不能与录制数据、推理同时使用：页面打开期间相机被其占用，录制数据时将无法获取画面。录制前请先关闭该页面。

## 3.4 相机变量

将两路相机的设置保存在变量中，后续录制数据与部署时直接使用 `$CAMERAS`。每次重新登录后都需重新执行：

```bash
CAMERAS="{ top: {type: opencv, index_or_path: /dev/xbotics/top, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/xbotics/wrist, width: 640, height: 480, fps: 30, fourcc: MJPG}}"
```

`top` 与 `wrist` 是相机在数据集中的名字。部署时须与训练模型所用的数据集保持一致，不要随意修改。

## 3.5 标定

每条机械臂首次使用前需标定一次。标定结果按 `id` 保存在端侧板上，后续录制数据、部署时使用同一个 `id`，即会采用这次的标定。标定有网页与命令行两种方式，结果保存为同一个文件，任选其一即可。

**网页标定（推荐）：** 在浏览器中打开 `http://<端侧板IP>:8006`，依照 3D 模型逐步操作。

1. 选择「从臂」或「主臂」，串口默认为其固定名字。
2. **先用手托住机械臂**，再点击「开始标定」。连接后舵机立即卸力，不托住会使机械臂下落。
3. **中位**：依照中间的模型，将整条机械臂摆到各关节行程的中间位置，按回车。
4. **限位**：逐个关节进行，将当前关节缓慢转过两端，页面自动记录最小值与最大值，按回车记录该关节。腕部旋转可整圈转动，无需记录。转动范围不足的关节会标红，点击「重试」只重新标定该关节。
5. **写入**：全部通过后点击「写入标定」，写入舵机并读回核对，再保存为标定文件。

中途点击「取消会话」，舵机与标定文件均恢复到开始前的状态。标定期间，该机械臂不要同时用于录制数据、推理或硬件检测的「读电机」。

![网页标定：从臂处于「记录限位」步骤，半透明部分为目标姿态](images/use-13-calibrate-web.png)

**命令行标定：** 按终端中的提示，先将各关节摆到中间位置，再将每个关节转过全程。

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/xbotics/follower --robot.id=follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/xbotics/leader --teleop.id=leader
```

## 3.6 遥操作检查

遥操作是指操作者握持主臂运动，从臂实时跟随。用它检查一遍：两条机械臂、两路相机与标定均正确，即表示硬件准备完成。

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/xbotics/follower --robot.id=follower \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader --teleop.port=/dev/xbotics/leader --teleop.id=leader \
  --display_data=true
```

`--robot` 为从臂，`--teleop` 为主臂，`id` 须与标定时一致。按 `Ctrl+C` 停止。

带有 `--display_data=true` 时，端侧板会开启网页预览。在浏览器中打开 `http://<端侧板IP>:9090`，即可看到两路相机画面与关节曲线，启动后需等待十余秒才能打开：

<video controls muted playsinline preload="metadata" poster="_static/poster/rerun.jpg" src="_static/media/rerun.mp4"></video>

逐项检查：

- 从臂能跟随主臂，没有明显的延迟与抖动
- 每个关节都能转到需要的位置，夹爪能完全张开与闭合
- 顶视相机能看到整个工作区：机械臂、物体与料盒都在画面中
- 腕部相机能看到夹爪与待抓取的物体

如有一项不符合，请回到前面各节重新检查接线或标定。四项均符合，这套硬件即可用于录制数据。
