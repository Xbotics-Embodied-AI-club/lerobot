"""标定对象：真机（SO101 主臂 / 从臂）与演示用的模拟舵机，以及当前机器上能选的串口。"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# 板子装了 Xbotics udev 规则时主从臂串口的固定名字（见 XBOTICS.md）
FIXED_PORTS = {"leader": "/dev/xbotics/leader", "follower": "/dev/xbotics/follower"}


@dataclass
class ArmSettings:
    """一条臂的标定设置：串口与 id（标定文件按 id 存，和 ``--robot.id`` / ``--teleop.id`` 同一个）。"""

    type: str
    port: str
    id: str
    calibration_dir: Path | None = None


def default_port(target: str) -> str:
    """板子上有固定名字就用它，否则留空，让页面从检测到的串口里选。"""
    fixed = FIXED_PORTS[target]
    return fixed if Path(fixed).exists() else ""


def clone_calibration(calibration: dict[str, MotorCalibration]) -> dict[str, MotorCalibration]:
    return {name: MotorCalibration(**asdict(value)) for name, value in calibration.items()}


class DemoBus:
    """不接硬件也能把整个标定流程走一遍的模拟舵机总线，接口与 ``FeetechMotorsBus`` 用到的那几个一致。"""

    def __init__(self):
        self.motors = {
            name: Motor(
                index + 1,
                "sts3215",
                MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES,
            )
            for index, name in enumerate(JOINT_NAMES)
        }
        self.positions: dict[str, int] = dict.fromkeys(JOINT_NAMES, 2047)
        self.calibration = {
            name: MotorCalibration(motor.id, 0, 0, 500, 3500) for name, motor in self.motors.items()
        }
        self.torque_enabled = True
        self.is_connected = False

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        self.is_connected = False

    def disable_torque(self, *args, **kwargs):
        self.torque_enabled = False

    def write(self, data_name, motor, value, **kwargs):
        pass

    def ping(self, motor, **kwargs):
        return self.motors[motor].id

    def sync_read(self, data_name, motors=None, **kwargs):
        if data_name != "Present_Position":
            raise ValueError(f"Demo bus does not provide {data_name}")
        names = list(self.motors) if motors is None else [motors] if isinstance(motors, str) else motors
        return {name: self.positions[name] for name in names}

    def set_half_turn_homings(self, motors=None):
        names = list(self.motors) if motors is None else [motors] if isinstance(motors, str) else motors
        offsets = {}
        for name in names:
            offsets[name] = self.positions[name] - 2047
            self.positions[name] = 2047
        self.calibration = {}
        return offsets

    def read_calibration(self):
        if not self.calibration:
            return {name: MotorCalibration(motor.id, 0, 0, 0, 4095) for name, motor in self.motors.items()}
        return clone_calibration(self.calibration)

    def write_calibration(self, calibration_dict, cache=True):
        if cache:
            self.calibration = clone_calibration(calibration_dict)


class DemoDevice:
    def __init__(self, root: Path):
        self.bus = DemoBus()
        self.calibration = clone_calibration(self.bus.calibration)
        self.calibration_fpath = root / "demo.json"
        self._connected = False

    def connect(self, calibrate=False):
        self.bus.connect()
        self._connected = True

    def disconnect(self):
        self.bus.disconnect()
        self._connected = False


class DemoDeviceOwner:
    """演示设备连同它的临时目录：标定文件与备份都写进临时目录，用完删掉，不碰真的标定。"""

    def __init__(self):
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="lerobot-calibration-demo-")
        self.device = DemoDevice(Path(self._temporary_directory.name))

    def cleanup(self):
        self._temporary_directory.cleanup()


def available_serial_ports() -> list[dict[str, Any]]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    ports = []
    for port in list_ports.comports():
        vid_pid = f"{port.vid:04X}:{port.pid:04X}" if port.vid is not None and port.pid is not None else None
        ports.append(
            {
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
                "vid_pid": vid_pid,
            }
        )
    # 固定名字不在 comports() 里（它列的是真实设备），存在就补到最前面
    for fixed in FIXED_PORTS.values():
        if Path(fixed).exists():
            target = str(Path(fixed).resolve())
            ports.insert(
                0,
                {"device": fixed, "description": f"固定名字 → {target}", "hwid": "", "vid_pid": None},
            )
    return sorted(ports, key=lambda port: (not port["device"].startswith("/dev/xbotics/"), port["device"]))


def make_real_device(target: str, settings: ArmSettings):
    if target == "leader":
        if settings.type != "so101_leader":
            raise ValueError(f"网页标定只支持 so101_leader，当前为 {settings.type}")
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        return SO101Leader(
            SO101LeaderConfig(port=settings.port, id=settings.id, calibration_dir=settings.calibration_dir)
        )
    if target == "follower":
        if settings.type != "so101_follower":
            raise ValueError(f"网页标定只支持 so101_follower，当前为 {settings.type}")
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        return SO101Follower(
            SO101FollowerConfig(port=settings.port, id=settings.id, calibration_dir=settings.calibration_dir)
        )
    raise ValueError(f"未知标定对象: {target}")
