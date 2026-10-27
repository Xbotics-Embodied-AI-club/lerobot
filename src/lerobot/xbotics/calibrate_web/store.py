"""标定文件的原子写入，以及标定开始前对舵机与标定文件的备份 / 恢复。"""

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import draccus

from lerobot.motors import MotorCalibration


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_calibration(path: Path, calibration: dict[str, MotorCalibration]) -> None:
    """与 ``Robot._save_calibration`` 写出同样格式的 JSON，但先写临时文件再改名，写一半断电也不留半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            with draccus.config_type("json"):
                draccus.dump(calibration, handle, indent=4)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


class CalibrationBackup:
    """开始标定前读下舵机里的标定与磁盘上的标定文件，取消或写入失败时原样放回。

    备份另存一份到标定文件旁的 ``backups/<id>_<时间>.json``，进程崩了也找得回来。
    """

    def __init__(self, device):
        self.device = device
        self.path = device.calibration_fpath
        self.motor_calibration = {}
        self.file_exists = False
        self.file_content = None
        self.backup_path = None

    def create(self) -> None:
        self.motor_calibration = self.device.bus.read_calibration()
        self.file_exists = self.path.is_file()
        self.file_content = self.path.read_text(encoding="utf-8") if self.file_exists else None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.backup_path = self.path.parent / "backups" / f"{self.path.stem}_{timestamp}.json"
        payload = {
            "motor_calibration": {name: asdict(value) for name, value in self.motor_calibration.items()},
            "file_exists": self.file_exists,
            "file_content": self.file_content,
        }
        atomic_write_text(self.backup_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def restore(self, motors: bool = True) -> None:
        """先放回标定文件（总是做得到），再放回舵机里的标定；串口已断开时给 ``motors=False`` 只放文件。"""
        if self.file_exists and self.file_content is not None:
            atomic_write_text(self.path, self.file_content)
        elif self.path.exists():
            self.path.unlink()
        if motors and self.motor_calibration:
            self.device.bus.write_calibration(self.motor_calibration)
            self.device.calibration = self.motor_calibration
