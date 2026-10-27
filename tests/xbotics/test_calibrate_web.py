"""网页标定（lerobot.xbotics.calibrate_web）：用演示舵机把会话状态机与 HTTP 接口走一遍。

移植自 gitee ``lerobot-v0.5.1-custom`` 的 ``lerobot_tools/tests/test_calibration.py``，按本线的接口改写，
另加：模型文件随包、不是 LFS 指针；前端构建产物在包里；卸力后切位置模式。
"""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

pytest.importorskip("flask")

from lerobot.xbotics.calibrate_web.app import MODEL_DIR, STATIC_DIR, create_app  # noqa: E402
from lerobot.xbotics.calibrate_web.devices import (  # noqa: E402
    JOINT_NAMES,
    ArmSettings,
    DemoBus,
    DemoDevice,
    clone_calibration,
)
from lerobot.xbotics.calibrate_web.session import CalibrationSession  # noqa: E402

LEADER = ArmSettings(type="so101_leader", port="/dev/ttyLeaderTest", id="test_leader")
FOLLOWER = ArmSettings(type="so101_follower", port="/dev/ttyFollowerTest", id="test_follower")
ARMS = {"leader": LEADER, "follower": FOLLOWER}


def demo_session(target="leader", **kwargs):
    return CalibrationSession(target, ARMS[target], demo=True, **kwargs)


def advance_to_review(session):
    session.confirm_middle()
    for _ in range(5):
        session.confirm_limit()
    assert session.phase == "review"


def test_complete_demo_calibration():
    session = demo_session().start()
    try:
        advance_to_review(session)
        snapshot = session.snapshot()
        assert snapshot["available_actions"] == ["write", "retry_joint"]
        assert snapshot["joints"]["wrist_roll"]["range_min"] == 0
        assert snapshot["joints"]["wrist_roll"]["range_max"] == 4095

        session.write()

        assert session.phase == "done"
        assert not session.connected
        saved = json.loads(session.device.calibration_fpath.read_text(encoding="utf-8"))
        assert set(saved) == set(JOINT_NAMES)
        assert saved["shoulder_pan"]["range_min"] == 1024
        assert saved["shoulder_pan"]["range_max"] == 3070
        assert session.backup.backup_path.is_file()
    finally:
        session.cleanup()


def test_start_switches_motors_to_position_mode():
    """与 SOFollower.calibrate 一样：卸力后每个关节写 Operating_Mode = 位置模式。"""
    writes = []

    class RecordingBus(DemoBus):
        def disable_torque(self, *args, **kwargs):
            writes.append(("disable_torque", None, None))
            super().disable_torque()

        def write(self, data_name, motor, value, **kwargs):
            writes.append((data_name, motor, value))

    session = demo_session()
    session.device.bus = RecordingBus()
    session.start()
    try:
        # 先卸力再写：力矩开着时舵机不接受 EEPROM 写入，且不报错
        assert writes[0] == ("disable_torque", None, None)
        assert [(name, motor) for name, motor, _ in writes[1:]] == [
            ("Operating_Mode", m) for m in JOINT_NAMES
        ]
        assert {value for _, _, value in writes[1:]} == {0}  # OperatingMode.POSITION
        assert not session.device.bus.torque_enabled
    finally:
        session.cancel()
        session.cleanup()


def test_limit_confirmation_uses_observed_motion_range():
    session = demo_session().start()
    try:
        session.confirm_middle()
        speech = session.snapshot()["speech_instruction"]
        assert speech == "请移动肩部旋转到两端极限，完成后按回车。"
        session.demo = False
        for position in (1500, 3000, 2047):
            session.device.bus.positions["shoulder_pan"] = position
            session._read_positions_locked()

        session.confirm_limit()

        joint = session.joints["shoulder_pan"]
        assert joint["raw"] == 2047
        assert joint["range_min"] == 1500
        assert joint["range_max"] == 3000
        assert joint["status"] == "pass"
        assert session.active_joint == "shoulder_lift"
    finally:
        session.cancel()
        session.cleanup()


def test_invalid_range_can_be_retried():
    session = demo_session("follower").start()
    try:
        session.confirm_middle()
        session.demo = False
        session.confirm_limit()  # 关节没动过：最小 = 最大

        joint = session.joints["shoulder_pan"]
        assert joint["status"] == "error"
        assert "最小极限" in joint["reason"]
        assert "confirm_limit" not in session.snapshot()["available_actions"]

        session.retry_joint("shoulder_pan")
        assert session.phase == "ranges"
        assert session.stage == "range"
        assert joint["range_min"] is None
        assert joint["range_max"] is None
        assert joint["status"] == "active"
    finally:
        session.cancel()
        session.cleanup()


def test_cancel_restores_motor_backup():
    session = demo_session().start()
    original = clone_calibration(session.backup.motor_calibration)
    try:
        session.confirm_middle()
        assert session.device.bus.read_calibration() != original

        session.cancel()

        assert session.phase == "cancelled"
        assert session.device.bus.read_calibration() == original
        assert not session.device.calibration_fpath.exists()
        assert not session.connected
    finally:
        session.cleanup()


def test_cancel_restores_existing_calibration_file():
    session = demo_session()
    session.device.calibration_fpath.write_text('{"old": true}', encoding="utf-8")
    session.start()
    try:
        advance_to_review(session)
        session.cancel()
        assert session.device.calibration_fpath.read_text(encoding="utf-8") == '{"old": true}'
    finally:
        session.cleanup()


def test_readback_mismatch_restores_backup():
    class MismatchedBus(DemoBus):
        mismatch = False

        def write_calibration(self, calibration_dict, cache=True):
            super().write_calibration(calibration_dict, cache)
            self.mismatch = True

        def read_calibration(self):
            calibration = super().read_calibration()
            if self.mismatch:
                calibration["shoulder_pan"].range_min += 1
            return calibration

    session = demo_session()
    session.device.bus = MismatchedBus()
    original = clone_calibration(session.device.bus.calibration)
    session.start()
    try:
        advance_to_review(session)
        with pytest.raises(RuntimeError, match="回读"):
            session.write()

        assert session.phase == "error"
        assert session.device.bus.calibration == original
        assert not session.device.calibration_fpath.exists()
    finally:
        session.cleanup()


def test_disconnected_port_has_precise_error(tmp_path, monkeypatch):
    device = DemoDevice(tmp_path)
    device.connect = lambda calibrate=False: None
    monkeypatch.setattr(
        "lerobot.xbotics.calibrate_web.session.make_real_device", lambda target, settings: device
    )
    client = create_app(ARMS).test_client()

    response = client.post("/api/session", json={"target": "follower", "demo": False, "port": "COM_WRONG"})

    assert response.status_code == 503
    message = response.get_json()["error"]
    assert "COM_WRONG" in message
    assert "SO101" in message
    assert "未能建立连接" in message


def test_unresponsive_joint_has_precise_error(tmp_path, monkeypatch):
    device = DemoDevice(tmp_path)
    device.bus.ping = lambda name, **kwargs: None
    monkeypatch.setattr(
        "lerobot.xbotics.calibrate_web.session.make_real_device", lambda target, settings: device
    )
    client = create_app(ARMS).test_client()

    response = client.post("/api/session", json={"target": "leader", "demo": False, "port": "COM4"})

    assert response.status_code == 503
    message = response.get_json()["error"]
    assert "COM4" in message
    assert "ID 1" in message
    assert "供电、接线和 ID" in message


def test_flask_api_and_session_exclusion():
    app = create_app(ARMS)
    app.config.update(TESTING=True)
    client = app.test_client()
    manager = app.extensions["calibration_manager"]

    response = client.post("/api/session", json={"target": "leader", "demo": True, "port": "COM_TEST"})
    assert response.status_code == 201
    assert response.get_json()["phase"] == "middle"
    assert response.get_json()["port"] == "COM_TEST"
    assert client.post("/api/session", json={"target": "follower", "demo": True}).status_code == 409

    assert client.post("/api/session/action", json={"action": "confirm_middle"}).status_code == 200
    for _ in range(5):
        response = client.post("/api/session/action", json={"action": "confirm_limit"})
        assert response.status_code == 200
    assert response.get_json()["phase"] == "review"

    response = client.post("/api/session/action", json={"action": "write"})
    assert response.status_code == 200
    assert response.get_json()["phase"] == "done"
    manager.get().cleanup()


def test_devices_report_configured_arms_and_serial_ports(monkeypatch):
    monkeypatch.setattr(
        "lerobot.xbotics.calibrate_web.app.available_serial_ports",
        lambda: [{"device": "COM9", "description": "Test servo", "hwid": "test", "vid_pid": "1A86:7523"}],
    )
    client = create_app(ARMS).test_client()

    payload = client.get("/api/devices").get_json()

    assert payload["devices"]["leader"] == {
        "type": "so101_leader",
        "port": "/dev/ttyLeaderTest",
        "id": "test_leader",
    }
    assert payload["devices"]["follower"]["id"] == "test_follower"
    assert payload["ports"][0]["device"] == "COM9"


def test_real_session_without_port_is_rejected():
    with pytest.raises(ValueError, match="请选择串口"):
        CalibrationSession("leader", ArmSettings(type="so101_leader", port="", id="x"), demo=False)


def test_blank_port_is_rejected():
    with pytest.raises(ValueError, match="有效的串口"):
        demo_session(port=" ")


def test_api_rejects_unknown_action_without_session():
    client = create_app(ARMS).test_client()
    assert client.get("/api/session").status_code == 404
    assert client.post("/api/session/action", json={"action": "unknown"}).status_code == 404


def test_urdf_models_and_meshes_are_complete():
    """从臂是星禾套件模型（连台子与相机支架），主臂是官方模型；网页只加载 <visual>，那些网格都要随包。"""
    for filename in ("kit/kit_v1_so101.urdf", "so101_leader_new_calib.urdf"):
        urdf = MODEL_DIR / filename
        robot = ET.parse(urdf).getroot()
        joints = {joint.get("name") for joint in robot.findall("joint") if joint.get("type") != "fixed"}
        assert joints == set(JOINT_NAMES)
        for mesh in robot.iterfind("link/visual/geometry/mesh"):
            assert (urdf.parent / mesh.get("filename")).is_file(), (filename, mesh.get("filename"))

    kit = ET.parse(MODEL_DIR / "kit/kit_v1_so101.urdf").getroot()
    kit_links = {link.get("name") for link in kit.findall("link")}
    assert {
        "kit_base_plate_link",
        "frame_extrusion_link",
        "top_camera_mount_link",
        "wrist_camera_mount_link",
    } <= kit_links

    leader = ET.parse(MODEL_DIR / "so101_leader_new_calib.urdf").getroot()
    assert leader.find("link[@name='trigger_link']") is not None
    assert leader.find("link[@name='moving_jaw_so101_v1_link']") is None
    gripper = leader.find("joint[@name='gripper']")
    assert gripper.find("child").get("link") == "trigger_link"
    assert float(gripper.find("limit").get("upper")) == pytest.approx(0.785398163397)


def test_meshes_are_real_files_not_lfs_pointers():
    """STL 随包发布：是 LFS 指针的话 pip 从 git 装下来的页面加载不出模型。"""
    meshes = sorted(MODEL_DIR.rglob("*.stl"))
    assert meshes
    for mesh in meshes:
        head = mesh.read_bytes()[:64]
        assert not head.startswith(b"version https://git-lfs"), mesh.name
        assert mesh.stat().st_size > 10_000, mesh.name


def test_built_frontend_and_model_are_served():
    assert (STATIC_DIR / "index.html").is_file(), (
        "前端没构建：xbotics/calibrate_web_frontend 里 npm run build"
    )
    client = create_app(ARMS).test_client()

    index = client.get("/")
    assert index.status_code == 200
    assert b"<html" in index.data.lower()
    asset = next(Path(STATIC_DIR, "assets").glob("*.js")).name
    assert client.get(f"/assets/{asset}").status_code == 200

    urdf = client.get("/model/kit/kit_v1_so101.urdf")
    assert urdf.status_code == 200
    assert b"<robot" in urdf.data
    assert client.get("/model/kit/assets/frame_extrusion.stl").status_code == 200
    assert client.get("/model/assets/Handle_SO101.stl").status_code == 200
    assert client.get("/model/../app.py").status_code == 404


class StrictBus(DemoBus):
    """像真的总线一样：断开后再读写就抛错；写入后回读对不上。"""

    mismatch = False

    def _require_connected(self):
        if not self.is_connected:
            raise ConnectionError("bus is not connected")

    def write_calibration(self, calibration_dict, cache=True):
        self._require_connected()
        super().write_calibration(calibration_dict, cache)
        self.mismatch = True

    def read_calibration(self):
        self._require_connected()
        calibration = super().read_calibration()
        if self.mismatch:
            calibration["shoulder_pan"].range_min += 1
        return calibration


def test_failed_write_can_be_cancelled_and_restarted(monkeypatch):
    """写入失败后串口已释放：取消只放回标定文件，并且能马上开新会话，不用重启服务。"""
    monkeypatch.setattr("lerobot.xbotics.calibrate_web.devices.DemoBus", StrictBus)
    app = create_app(ARMS)
    client = app.test_client()
    manager = app.extensions["calibration_manager"]

    assert client.post("/api/session", json={"target": "leader", "demo": True}).status_code == 201
    session = manager.get()
    session.device.calibration_fpath.write_text('{"old": true}', encoding="utf-8")
    session.backup.create()  # 备份要包含这份旧文件
    advance_to_review(session)

    assert client.post("/api/session/action", json={"action": "write"}).status_code == 500
    assert session.phase == "error" and not session.connected

    response = client.post("/api/session/cancel")
    assert response.status_code == 200
    assert response.get_json()["phase"] == "cancelled"
    assert session.device.calibration_fpath.read_text(encoding="utf-8") == '{"old": true}'

    assert client.post("/api/session", json={"target": "follower", "demo": True}).status_code == 201
    manager.get().cleanup()


def test_session_stuck_in_error_does_not_block_a_new_one(monkeypatch):
    app = create_app(ARMS)
    client = app.test_client()
    manager = app.extensions["calibration_manager"]
    client.post("/api/session", json={"target": "leader", "demo": True})
    stuck = manager.get()
    stuck.phase = "error"  # 例如采样时串口掉线

    assert client.post("/api/session", json={"target": "leader", "demo": True}).status_code == 201
    assert manager.get() is not stuck
    assert not stuck.connected
    manager.get().cleanup()
