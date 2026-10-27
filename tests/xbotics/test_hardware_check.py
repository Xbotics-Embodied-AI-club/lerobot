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


class FakeBus:
    """假的飞特总线：记下断开时卸没卸力；寄存器读数按 ID 给。"""

    found = {1: 777, 2: 777, 9: 12345}
    disconnects = []

    def __init__(self, port, motors):
        self.port, self.motors, self.is_connected = port, motors, False

    def connect(self, handshake=True):
        self.is_connected = True

    def set_baudrate(self, baudrate):
        assert baudrate == 1_000_000

    def broadcast_ping(self, num_retry=0):
        return dict(self.found)

    def read(self, register, motor, normalize=True, num_retry=0):
        assert normalize is False
        values = {
            "Present_Position": 2047,
            "Present_Voltage": 121,
            "Present_Temperature": 35,
            "Present_Current": 10,
            "Present_Load": 25,
            "Torque_Enable": 0,
            "Status": 0,
        }
        if motor == "id2" and register == "Present_Temperature":
            raise ConnectionError("no status packet")
        return values[register]

    def disconnect(self, disable_torque=True):
        FakeBus.disconnects.append(disable_torque)
        self.is_connected = False


def test_read_motors_scans_reads_and_never_disables_torque(monkeypatch):
    pytest.importorskip("scservo_sdk")
    monkeypatch.setattr("lerobot.motors.feetech.FeetechMotorsBus", FakeBus)
    FakeBus.disconnects = []

    result = hardware_check.read_motors("/dev/ttyS3")

    assert result["error"] is None
    first, second, unknown = result["motors"]
    assert first == {
        "id": 1,
        "model": "sts3215",
        "position": 2047,
        "voltage_v": 12.1,
        "temperature_c": 35,
        "current_ma": 65.0,
        "load_pct": 2.5,
        "torque_on": 0,
        "status": 0,
    }
    assert second["temperature_c"] is None  # 单个寄存器读失败只空这一格
    assert unknown == {"id": 9, "model": "未知型号 12345"}
    assert FakeBus.disconnects and not any(FakeBus.disconnects), "只读工具断开时不能卸力"


def test_read_motors_reports_empty_bus(monkeypatch):
    pytest.importorskip("scservo_sdk")
    monkeypatch.setattr("lerobot.motors.feetech.FeetechMotorsBus", FakeBus)
    monkeypatch.setattr(FakeBus, "found", {})

    result = hardware_check.read_motors("/dev/ttyS1")

    assert result["motors"] == [] and "没有扫到舵机" in result["error"]


def test_motors_endpoint_only_accepts_listed_ports(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(server + "/api/motors?port=/etc/passwd")
    assert error.value.code == 400
