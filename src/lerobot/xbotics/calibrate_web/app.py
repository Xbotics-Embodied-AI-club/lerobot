"""网页标定的后端：Flask 出静态页面、会话接口与事件流（SSE），模型文件另从 ``model/`` 目录出。"""

import json
import logging
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from .devices import ArmSettings, available_serial_ports
from .session import ACTIVE_PHASES, CalibrationSession
from .tts import SpeechSynthesizer

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"  # 前端构建产物（xbotics/calibrate_web_frontend 里 npm run build 生成）
MODEL_DIR = PACKAGE_DIR / "model"  # SO101 主臂 / 从臂的 URDF 与 STL


class SessionManager:
    """同一时刻只允许一个进行中的标定会话（一个串口只能被一个人占着）。"""

    def __init__(self, arms: dict[str, ArmSettings]):
        self._lock = threading.Lock()
        self._session = None
        self.arms = arms

    def start(self, target, demo, port=None):
        if target not in self.arms:
            raise ValueError(f"未知标定对象: {target}")
        with self._lock:
            # 出错的会话（写入失败、串口掉线、恢复也失败）不挡新会话，否则只能重启服务
            if self._session and self._session.phase in ACTIVE_PHASES and self._session.phase != "error":
                raise RuntimeError("已有进行中的标定会话，请先完成或取消")
            if self._session:
                self._session.abandon()
            self._session = CalibrationSession(target, self.arms[target], demo=demo, port=port)
            try:
                self._session.start()
            except Exception:
                self._session = None
                raise
            return self._session

    def get(self):
        with self._lock:
            if not self._session:
                raise LookupError("尚未创建标定会话")
            return self._session


def create_app(arms: dict[str, ArmSettings], demo_default: bool = False) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
    manager = SessionManager(arms)
    app.extensions["calibration_manager"] = manager
    speech = SpeechSynthesizer()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.get("/")
    def index():
        if not (STATIC_DIR / "index.html").is_file():
            return jsonify(
                {"error": "前端尚未构建，请先在 xbotics/calibrate_web_frontend 运行 npm run build"}
            ), 503
        return app.send_static_file("index.html")

    @app.get("/model/<path:filename>")
    def model(filename):
        return send_from_directory(MODEL_DIR, filename)

    @app.get("/api/devices")
    def devices():
        return jsonify(
            {
                "devices": {
                    target: {
                        key: value for key, value in asdict(settings).items() if key != "calibration_dir"
                    }
                    for target, settings in manager.arms.items()
                },
                "ports": available_serial_ports(),
                "demo_default": demo_default,
            }
        )

    @app.post("/api/tts")
    def text_to_speech():
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        if not isinstance(text, str):
            raise ValueError("语音文本必须是字符串")
        return send_file(speech.synthesize(text), mimetype="audio/mpeg", conditional=True)

    @app.post("/api/session")
    def start_session():
        data = request.get_json(silent=True) or {}
        session = manager.start(str(data.get("target", "")), bool(data.get("demo", False)), data.get("port"))
        return jsonify(session.snapshot()), 201

    @app.get("/api/session")
    def get_session():
        return jsonify(manager.get().snapshot())

    @app.post("/api/session/action")
    def session_action():
        data = request.get_json(silent=True) or {}
        action = data.get("action")
        session = manager.get()
        if action == "confirm_middle":
            session.confirm_middle()
        elif action == "confirm_limit":
            session.confirm_limit()
        elif action == "retry_joint":
            session.retry_joint(str(data.get("joint", "")))
        elif action == "write":
            session.write()
        else:
            raise ValueError(f"不支持的操作: {action}")
        return jsonify(session.snapshot())

    @app.post("/api/session/cancel")
    def cancel_session():
        session = manager.get()
        session.cancel()
        return jsonify(session.snapshot())

    @app.get("/api/events")
    def events():
        session = manager.get()
        initial_seq = session.snapshot()["seq"]

        def stream():
            last_seq = initial_seq
            last_heartbeat = time.monotonic()
            while True:
                current = session.snapshot()
                if current["seq"] != last_seq:
                    last_seq = current["seq"]
                    yield f"data: {json.dumps(current, ensure_ascii=False)}\n\n"
                elif time.monotonic() - last_heartbeat >= 15:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                time.sleep(0.1)

        return Response(stream(), mimetype="text/event-stream")

    @app.errorhandler(LookupError)
    def not_found(error):
        return jsonify({"error": str(error)}), 404

    @app.errorhandler(ValueError)
    def bad_request(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(ConnectionError)
    def connection_error(error):
        return jsonify({"error": str(error)}), 503

    @app.errorhandler(RuntimeError)
    def runtime_error(error):
        return jsonify({"error": str(error)}), 409 if "进行中的标定" in str(error) else 500

    return app


def run_server(
    arms: dict[str, ArmSettings],
    host: str,
    port: int,
    demo: bool = False,
    open_browser: bool = False,
) -> None:
    app = create_app(arms, demo_default=demo)
    shown_host = "127.0.0.1" if host == "0.0.0.0" else host
    address = f"http://{shown_host}:{port}"
    if open_browser:
        threading.Timer(1, webbrowser.open, args=(address,)).start()
    print(f"星禾套件网页标定: {address}" + ("（局域网里用这台机器的 IP 打开）" if host == "0.0.0.0" else ""))
    app.run(host=host, port=port, threaded=True, use_reloader=False)
