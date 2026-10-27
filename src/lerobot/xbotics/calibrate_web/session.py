"""一次标定会话：与 ``SOFollower.calibrate`` / ``SOLeader.calibrate`` 同一套步骤，拆成网页上一步一步点的状态机。

阶段：``middle``（整臂摆到中位，记 homing offset）→ ``ranges``（除腕部旋转外逐个关节走满行程，记最小 / 最大码值）
→ ``review``（复核，可单关节重试）→ ``writing``（写舵机、回读核对、原子写标定文件）→ ``done``。
任何时候可以取消，取消与写入失败都把开始前备份的舵机标定和标定文件放回去。

和命令行标定的差别：命令行一次录全部关节的行程，这里一个关节一个关节录，每个关节单独校验
（行程够不够、中位在不在范围里），哪个不对只重来哪个。写进舵机与文件的内容格式相同。
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import suppress
from dataclasses import replace

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode

from .devices import JOINT_NAMES, ArmSettings, DemoDeviceOwner, make_real_device
from .store import CalibrationBackup, atomic_write_calibration

JOINT_LABELS = {
    "shoulder_pan": "底座旋转",
    "shoulder_lift": "肩部抬升",
    "elbow_flex": "肘部弯曲",
    "wrist_flex": "腕部俯仰",
    "wrist_roll": "腕部旋转",
    "gripper": "夹爪",
}
ACTIVE_PHASES = {"middle", "ranges", "review", "writing", "error"}
FULL_TURN_JOINT = "wrist_roll"  # 与命令行标定一样，腕部旋转不录行程，直接取满圈 0–4095
SAMPLE_PERIOD_S = 0.05
# 串口偶尔丢一帧不该让整个会话报错：采样与 ping 都多试几次
READ_RETRIES = 2
PING_RETRIES = 3


class CalibrationSession:
    def __init__(self, target: str, settings: ArmSettings, demo: bool = False, port: str | None = None):
        if target not in {"leader", "follower"}:
            raise ValueError(f"未知标定对象: {target}")
        if port is not None:
            if not isinstance(port, str) or not port.strip():
                raise ValueError("请选择有效的串口")
            settings = replace(settings, port=port.strip())
        if not demo and not settings.port:
            raise ValueError("请选择串口")
        self.id = uuid.uuid4().hex[:12]
        self.target = target
        self.demo = demo
        self.port = settings.port
        self._demo_owner = DemoDeviceOwner() if demo else None
        self.device = self._demo_owner.device if self._demo_owner else make_real_device(target, settings)
        self.backup = CalibrationBackup(self.device)
        self.phase = "idle"
        self.stage = None
        self.active_joint = None
        self.connected = False
        self.torque_enabled = True
        self.error = None
        self.seq = 0
        self.joints = {
            name: {
                "id": index + 1,
                "label": JOINT_LABELS[name],
                "raw": None,
                "angle": None,
                "observed_min": None,
                "observed_max": None,
                "range_min": None,
                "range_max": None,
                "homing_offset": None,
                "status": "pending",
                "reason": None,
            }
            for index, name in enumerate(JOINT_NAMES)
        }
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._sampler = None

    def start(self):
        target_label = "主臂" if self.target == "leader" else "从臂"
        try:
            try:
                self.device.connect(calibrate=False)
            except Exception as error:
                raise ConnectionError(
                    f"无法打开{target_label}串口“{self.port}”。请检查端口是否选择正确、USB 是否连接，"
                    f"以及串口是否被其他程序占用。原始错误：{error}"
                ) from error
            if not self.device.bus.is_connected:
                raise ConnectionError(
                    f"{target_label}端口“{self.port}”未能建立连接。该端口可能不是 SO101 舵机控制板，"
                    "或设备已断开。请重新选择端口并检查 USB 连接。"
                )
            with self._lock:
                self.connected = True
                for name in JOINT_NAMES:
                    if self.device.bus.ping(name, num_retry=PING_RETRIES) is None:
                        raise ConnectionError(
                            f"{target_label}已在端口“{self.port}”打开串口，但关节 {self.joints[name]['label']}"
                            f"（ID {self.joints[name]['id']}）无响应。请确认端口连接的是 SO101，"
                            "并检查舵机供电、接线和 ID。"
                        )
                self.backup.create()
                self.device.bus.disable_torque()
                self.torque_enabled = False
                # 与命令行标定相同：卸力后统一切到位置模式，再记 homing offset
                for name in JOINT_NAMES:
                    self.device.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                self.phase = "middle"
                self._read_positions_locked()
            self._sampler = threading.Thread(
                target=self._sample_loop, daemon=True, name=f"calibration-{self.id}"
            )
            self._sampler.start()
        except Exception as error:
            with suppress(Exception):
                self.device.disconnect()
            self.connected = False
            self.cleanup()
            if isinstance(error, ConnectionError) and self.port in str(error):
                raise
            raise ConnectionError(f"{target_label}在端口“{self.port}”初始化失败：{error}") from error
        return self

    def _sample_loop(self):
        while not self._stop_event.wait(SAMPLE_PERIOD_S):
            try:
                with self._lock:
                    if self.connected and self.phase not in {"writing", "done", "cancelled"}:
                        self._read_positions_locked()
            except Exception as error:
                with self._lock:
                    self.phase = "error"
                    self.error = f"读取关节位置失败：{error}"
                    self.seq += 1
                break

    def _stop_sampler(self):
        self._stop_event.set()
        if self._sampler and self._sampler.is_alive() and self._sampler is not threading.current_thread():
            self._sampler.join(timeout=2)

    def _read_positions_locked(self):
        positions = self.device.bus.sync_read("Present_Position", normalize=False, num_retry=READ_RETRIES)
        if set(positions) != set(JOINT_NAMES):
            raise ConnectionError("舵机返回的关节集合不完整")
        now = time.monotonic()
        for name, value in positions.items():
            joint = self.joints[name]
            raw = float(value)
            joint["raw"] = raw
            joint["sampled_at"] = now
            if self.phase == "ranges" and name == self.active_joint:
                joint["observed_min"] = (
                    raw if joint["observed_min"] is None else min(joint["observed_min"], raw)
                )
                joint["observed_max"] = (
                    raw if joint["observed_max"] is None else max(joint["observed_max"], raw)
                )
            joint["angle"] = self._joint_angle(name, joint)
        self.seq += 1

    @staticmethod
    def _joint_angle(name, joint):
        """按 SO101 的口径换算成页面上显示的读数：臂关节是度（以行程中点为 0），夹爪是 0–100。"""
        raw = joint["raw"]
        range_min = joint["range_min"]
        range_max = joint["range_max"]
        if raw is None or joint["homing_offset"] is None or range_min is None or range_max is None:
            return None
        if range_max <= range_min:
            return None
        if name == "gripper":
            return (raw - range_min) / (range_max - range_min) * 100
        return (raw - (range_min + range_max) / 2) * 360 / 4095

    def _require_phase(self, phase):
        if self.phase != phase:
            raise ValueError(f"当前阶段为 {self.phase}，不能执行该操作")

    def confirm_middle(self):
        with self._lock:
            self._require_phase("middle")
            offsets = self.device.bus.set_half_turn_homings()
            for name, offset in offsets.items():
                joint = self.joints[name]
                joint["homing_offset"] = int(offset)
                joint["raw"] = 2047.0
                joint["observed_min"] = None
                joint["observed_max"] = None
            self.joints[FULL_TURN_JOINT].update(range_min=0, range_max=4095, status="pass")
            self.phase = "ranges"
            self._set_active("shoulder_pan")
            self._read_positions_locked()

    def _set_active(self, name):
        self.active_joint = name
        self.stage = "range"
        joint = self.joints[name]
        joint["status"] = "active"
        joint["reason"] = None
        joint["observed_min"] = joint["raw"]
        joint["observed_max"] = joint["raw"]
        self.seq += 1

    def confirm_limit(self):
        with self._lock:
            self._require_phase("ranges")
            name = self.active_joint
            if name is None:
                raise ValueError("当前没有等待确认的极限记录")
            joint = self.joints[name]
            if joint["status"] == "error":
                raise ValueError("请先重新标定当前关节")
            if self.demo:
                joint["observed_min"] = 1024.0
                joint["observed_max"] = 3070.0
                self.device.bus.positions[name] = 2047
            self._read_positions_locked()
            if time.monotonic() - joint.get("sampled_at", 0) > 1:
                raise ConnectionError("当前关节位置采样已过期，请检查连接")
            if joint["observed_min"] is None or joint["observed_max"] is None:
                joint["status"] = "error"
                joint["reason"] = "尚未记录到运动范围，请移动关节后重试"
                self.seq += 1
                return
            joint["range_min"] = int(round(joint["observed_min"]))
            joint["range_max"] = int(round(joint["observed_max"]))
            self._validate_range(name)
            if joint["status"] == "error":
                self.seq += 1
                return
            joint["status"] = "pass"
            next_name = next(
                (
                    candidate
                    for candidate in JOINT_NAMES
                    if candidate != FULL_TURN_JOINT and self.joints[candidate]["status"] != "pass"
                ),
                None,
            )
            if next_name:
                self._set_active(next_name)
            else:
                self.active_joint = None
                self.stage = None
                self.phase = "review"
            self.seq += 1

    def _validate_range(self, name):
        joint = self.joints[name]
        range_min = joint["range_min"]
        range_max = joint["range_max"]
        reason = None
        if not 0 <= range_min <= 4095 or not 0 <= range_max <= 4095:
            reason = "极限值超出编码器范围 0–4095"
        elif range_min >= range_max:
            reason = "最小极限必须小于最大极限"
        elif range_max - range_min < 50:
            reason = f"检测到的行程仅 {range_max - range_min}，请完整移动关节后重试"
        elif not range_min <= 2047 <= range_max:
            reason = "中位点不在记录的极限范围内"
        if reason:
            joint["status"] = "error"
            joint["reason"] = reason

    def retry_joint(self, name):
        if name not in self.joints or name == FULL_TURN_JOINT:
            raise ValueError("该关节不需要或不能单独重试")
        with self._lock:
            if self.phase not in {"ranges", "review"}:
                raise ValueError("当前阶段不能重试关节")
            joint = self.joints[name]
            joint.update(range_min=None, range_max=None, angle=None, reason=None)
            self.phase = "ranges"
            self._set_active(name)

    def write(self):
        with self._lock:
            self._require_phase("review")
            if any(joint["status"] != "pass" for joint in self.joints.values()):
                raise ValueError("仍有未通过校验的关节")
            self.phase = "writing"
            self.error = None
            try:
                candidate = {}
                for name, motor in self.device.bus.motors.items():
                    joint = self.joints[name]
                    candidate[name] = MotorCalibration(
                        id=motor.id,
                        drive_mode=0,
                        homing_offset=joint["homing_offset"],
                        range_min=joint["range_min"],
                        range_max=joint["range_max"],
                    )
                self.device.bus.write_calibration(candidate)
                readback = self.device.bus.read_calibration()
                for name, expected in candidate.items():
                    if readback.get(name) != expected:
                        raise RuntimeError(f"关节 {self.joints[name]['label']} 的舵机回读与写入值不一致")
                atomic_write_calibration(self.device.calibration_fpath, candidate)
                self.device.calibration = candidate
                self.device.disconnect()
                self.connected = False
                self.phase = "done"
            except Exception as error:
                restore_error = self._restore_locked()
                with suppress(Exception):
                    self.device.disconnect()
                self.connected = False
                self.phase = "error"
                self.error = f"写入标定失败：{error}"
                if restore_error:
                    self.error += f"；恢复备份时也发生错误：{restore_error}"
                raise RuntimeError(self.error) from error
            finally:
                self.seq += 1
        self._stop_sampler()

    def _restore_locked(self):
        try:
            # 串口已经断开（写入失败后、或中途拔线）时舵机写不进去，只放回标定文件
            self.backup.restore(motors=self.connected and self.device.bus.is_connected)
            return None
        except Exception as error:
            return str(error)

    def cancel(self):
        self._stop_sampler()
        with self._lock:
            if self.phase in {"done", "cancelled"}:
                return
            restore_error = self._restore_locked()
            with suppress(Exception):
                self.device.bus.disable_torque()
            self.torque_enabled = False
            with suppress(Exception):
                self.device.disconnect()
            self.connected = False
            if restore_error:
                self.phase = "error"
                self.error = f"恢复标定备份失败：{restore_error}"
                self.seq += 1
                raise RuntimeError(self.error)
            self.phase = "cancelled"
            self.error = None
            self.seq += 1

    def abandon(self):
        """放弃一个出错的会话：停采样、尽量断开串口、清理，让新会话能接手。"""
        self._stop_sampler()
        with self._lock:
            with suppress(Exception):
                self.device.disconnect()
            self.connected = False
        self.cleanup()

    def cleanup(self):
        self._stop_sampler()
        if self._demo_owner:
            self._demo_owner.cleanup()
            self._demo_owner = None

    @property
    def instruction(self):
        if self.phase == "middle":
            return "请按目标动画将整臂移动到各关节行程中点，确认后记录 homing offset。"
        if self.phase == "ranges" and self.active_joint:
            joint = self.joints[self.active_joint]
            if joint["status"] == "error":
                return f"{joint['label']}：{joint['reason']}。请点击“重新标定此关节”。"
            return (
                f"请缓慢移动「{joint['label']}」完整经过最小和最大极限，系统会自动记录运动范围；"
                "完成后点击“记录限位”。"
            )
        if self.phase == "review":
            return "请复核六个关节的 offset 和极限值。异常关节可单独重试，确认无误后写入。"
        if self.phase == "writing":
            return "正在写入舵机并验证回读，请勿断开电源或串口。"
        if self.phase == "done":
            return "标定已写入并通过回读验证，串口已释放。"
        if self.phase == "cancelled":
            return "已取消标定并恢复备份。"
        if self.phase == "error":
            return self.error or "标定发生异常，请取消并恢复备份后重试。"
        return "请选择主臂或从臂开始标定。"

    @property
    def speech_instruction(self):
        if self.phase == "middle":
            return "请调整整臂到中位姿态，完成后按回车。"
        if self.phase == "ranges" and self.active_joint:
            joint = self.joints[self.active_joint]
            if joint["status"] == "error":
                return f"{joint['label']}标定异常，请重新标定该关节。"
            return f"请移动{joint['label']}到两端极限，完成后按回车。"
        if self.phase == "review":
            return "请复核标定结果，确认后写入。"
        if self.phase == "writing":
            return "正在写入标定，请勿断开设备。"
        if self.phase == "done":
            return "标定完成。"
        if self.phase == "cancelled":
            return "已取消标定并恢复备份。"
        if self.phase == "error":
            return "标定异常，请检查设备后重试。"
        return "请选择主臂或从臂开始标定。"

    def snapshot(self):
        with self._lock:
            actions = []
            if self.phase == "middle":
                actions.append("confirm_middle")
            elif self.phase == "ranges":
                if self.active_joint and self.joints[self.active_joint]["status"] != "error":
                    actions.append("confirm_limit")
                actions.append("retry_joint")
            elif self.phase == "review":
                actions.extend(("write", "retry_joint"))
            return {
                "id": self.id,
                "target": self.target,
                "demo": self.demo,
                "port": self.port,
                "calibration_path": None if self.demo else str(self.device.calibration_fpath),
                "phase": self.phase,
                "stage": self.stage,
                "active_joint": self.active_joint,
                "seq": self.seq,
                "connected": self.connected,
                "torque_enabled": self.torque_enabled,
                "error": self.error,
                "instruction": self.instruction,
                "speech_instruction": self.speech_instruction,
                "available_actions": actions,
                "backup_path": str(self.backup.backup_path) if self.backup.backup_path else None,
                "joints": {
                    name: {key: value for key, value in joint.items() if key != "sampled_at"}
                    for name, joint in self.joints.items()
                },
            }
