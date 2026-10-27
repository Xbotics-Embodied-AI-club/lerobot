"""网页硬件检测（lerobot.xbotics.hardware_check）：固定名字的对应、HTTP 接口、端口预检。"""

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lerobot.xbotics import hardware_check


def test_fixed_names_map_real_device_to_xbotics_name(tmp_path, monkeypatch):
    real = tmp_path / "ttyS3"
    real.touch()
    fixed_dir = tmp_path / "xbotics"
    fixed_dir.mkdir()
    (fixed_dir / "follower").symlink_to(real)
    monkeypatch.setattr(hardware_check, "IS_LINUX", True)
    monkeypatch.setattr(hardware_check, "FIXED_NAME_DIR", fixed_dir)

    assert hardware_check.fixed_names() == {str(real): str(fixed_dir / "follower")}


def test_fixed_names_empty_without_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware_check, "FIXED_NAME_DIR", tmp_path / "missing")
    assert hardware_check.fixed_names() == {}


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(hardware_check, "detect_cameras", lambda: [{"source": "/dev/video0", "name": "cam"}])
    monkeypatch.setattr(
        hardware_check,
        "list_serial_ports",
        lambda: [{"device": "/dev/ttyS3", "fixed_name": "/dev/xbotics/follower"}],
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), hardware_check.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:  # nosec
        return response.read().decode("utf-8")


def test_http_endpoints(server):
    assert "拔掉 USB" in _get(server + "/")
    assert json.loads(_get(server + "/api/ports"))[0]["fixed_name"] == "/dev/xbotics/follower"
    assert json.loads(_get(server + "/api/cameras"))[0]["source"] == "/dev/video0"
    assert json.loads(_get(server + "/api/stream_stats?src=/dev/video9"))["status"] == "未启动"


def test_port_available_detects_busy_port():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert not hardware_check.port_available("127.0.0.1", port)
    assert hardware_check.port_available("127.0.0.1", port)


def test_port_available_right_after_server_closed_connections():
    """刚关掉的服务（主动关连接留下 TIME_WAIT）不该挡住同一端口重启。"""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = listener.accept()
    conn.close()  # 服务端先关 ⇒ 服务端这一侧进 TIME_WAIT
    client.close()
    listener.close()

    assert hardware_check.port_available("127.0.0.1", port)


def test_only_usb_capture_nodes_are_listed_by_default(tmp_path, monkeypatch):
    """芯片自带的 rkisp 节点、UVC 的元数据节点（index 1）都不算 USB 相机画面节点。"""
    sysfs = tmp_path / "sys"
    for node, device, index in [
        ("video0", "platform/rkisp", "0"),
        ("video18", "platform/usb1/1-1/1-1:1.0", "0"),
        ("video19", "platform/usb1/1-1/1-1:1.0", "1"),
    ]:
        target = tmp_path / "devices" / device
        target.mkdir(parents=True, exist_ok=True)
        (sysfs / node).mkdir(parents=True)
        (sysfs / node / "device").symlink_to(target)
        (sysfs / node / "index").write_text(index + "\n")
    monkeypatch.setattr(hardware_check, "V4L_SYSFS", sysfs)

    assert [n for n in ("video0", "video18", "video19") if hardware_check.is_usb_capture_node(n)] == [
        "video18"
    ]
    assert not hardware_check.is_usb_capture_node("video99")


def test_onboard_serial_ports_with_fixed_names_are_listed(monkeypatch):
    """pyserial 滤掉的板载串口（ttyS*），只要有固定名字就补进列表；相机的固定名字不算串口。"""
    serial_list_ports = pytest.importorskip("serial.tools.list_ports")
    monkeypatch.setattr(serial_list_ports, "comports", lambda: [])
    monkeypatch.setattr(
        hardware_check,
        "fixed_names",
        lambda: {"/dev/ttyS3": "/dev/xbotics/follower", "/dev/video18": "/dev/xbotics/top"},
    )

    ports = hardware_check.list_serial_ports()

    assert [(p["device"], p["fixed_name"]) for p in ports] == [("/dev/ttyS3", "/dev/xbotics/follower")]


def _fake_bus_class():
    """真的 FeetechMotorsBus（真的控制表、真的报警文字），只把串口那一层换成假的。

    1、2 号正常（2 号读温度时丢包），3 号在报警（过载，回包 error=0x20），9 号型号不认识。
    """
    pytest.importorskip("scservo_sdk")
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.feetech.tables import MODEL_NUMBER, STS_SMS_SERIES_CONTROL_TABLE

    by_addr = {addr: name for name, (addr, _) in STS_SMS_SERIES_CONTROL_TABLE.items()}
    values = {
        "Present_Position": 2047,
        "Present_Voltage": 121,
        "Present_Temperature": 35,
        "Present_Current": 10,
        "Present_Load": (1 << 10) | 25,
        "Torque_Enable": 0,
        "Status": 0,
    }

    class FakeBus(FeetechMotorsBus):
        disconnects = []
        opened = False

        def connect(self, handshake=True):
            FakeBus.opened = True

        @property
        def is_connected(self):
            return FakeBus.opened

        def set_baudrate(self, baudrate):
            assert baudrate == 1_000_000

        def _broadcast_ping(self):
            return {1: 0, 2: 0, 3: 0x20, 9: 0}, 0

        def _read(self, address, length, motor_id, *, num_retry=0, raise_on_error=True, err_msg=""):
            assert raise_on_error is False
            error = 0x20 if motor_id == 3 else 0
            if address == MODEL_NUMBER[0]:
                return (12345 if motor_id == 9 else 777), 0, error
            name = by_addr[address]
            if motor_id == 2 and name == "Present_Temperature":
                return 0, -6, 0  # 没回包
            return values[name], 0, error

        def disconnect(self, disable_torque=True):
            FakeBus.disconnects.append(disable_torque)
            FakeBus.opened = False

    return FakeBus


def test_read_motors_lists_alarmed_motors_and_never_disables_torque(monkeypatch):
    fake = _fake_bus_class()
    monkeypatch.setattr("lerobot.motors.feetech.FeetechMotorsBus", fake)

    result = hardware_check.read_motors("/dev/ttyS3")

    assert result["error"] is None
    by_id = {m["id"]: m for m in result["motors"]}
    assert sorted(by_id) == [1, 2, 3, 9]
    assert {
        k: by_id[1][k]
        for k in (
            "model",
            "position",
            "voltage_v",
            "temperature_c",
            "current_ma",
            "load_pct",
            "torque_on",
            "status",
            "alarm",
        )
    } == {
        "model": "sts3215",
        "position": 2047,
        "voltage_v": 12.1,
        "temperature_c": 35,
        "current_ma": 65.0,
        "load_pct": -2.5,
        "torque_on": 0,
        "status": 0,
        "alarm": 0,
    }  # 负载按符号位解码：负值是方向
    assert by_id[2]["temperature_c"] is None  # 单个寄存器丢包只空这一格
    assert by_id[3]["alarm"] == 0x20 and by_id[3]["alarm_text"]  # 报警的舵机照样列出、读数照样有
    assert by_id[3]["position"] == 2047
    assert by_id[9]["model"] == "未知型号 12345"
    assert fake.disconnects == [False], "只读工具断开时不能卸力"


def test_read_motors_reports_empty_bus(monkeypatch):
    fake = _fake_bus_class()
    monkeypatch.setattr(fake, "_broadcast_ping", lambda self: ({}, -6))
    monkeypatch.setattr("lerobot.motors.feetech.FeetechMotorsBus", fake)

    result = hardware_check.read_motors("/dev/ttyS1")

    assert result["motors"] == [] and "没有扫到舵机" in result["error"]


def test_motors_endpoint_only_accepts_listed_ports(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(server + "/api/motors?port=/etc/passwd")
    assert error.value.code == 400
