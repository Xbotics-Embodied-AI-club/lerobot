"""推理客户端的 Web 控制台（Flask）：运行状态、相机画面、动作队列可视化，网页上改任务、启停、热调 RTC 参数。"""

import logging
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request

WEBUI_HTML_PATH = Path(__file__).with_name("webui.html")


def _load_index_html() -> str:
    return WEBUI_HTML_PATH.read_text(encoding="utf-8")


def _create_app(client) -> Flask:
    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.get("/")
    def index():
        return Response(_load_index_html(), mimetype="text/html")

    @app.get("/api/status")
    def status():
        return jsonify(client.get_webui_status())

    @app.get("/api/queue")
    def queue():
        return jsonify(client.get_webui_queue())

    @app.get("/api/rtc")
    def get_rtc():
        return jsonify(client.get_webui_rtc())

    @app.post("/api/rtc")
    def set_rtc():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": 'JSON body required, e.g. {"rtc_blend_steps": 8}'}), 400
        try:
            return jsonify(client.set_webui_rtc(data))
        except (ValueError, TypeError) as e:
            return jsonify({"error": str(e)}), 400

    @app.get("/api/control")
    def get_control():
        return jsonify({"task_active": client.task_active})

    @app.post("/api/control")
    def set_control():
        data = request.get_json(silent=True) or {}
        if "action" in data:
            active = str(data["action"]).strip().lower() in ("start", "run", "resume", "1", "true")
        else:
            active = data.get("task_active", True)
            if isinstance(active, str):
                active = active.strip().lower() in ("1", "true", "yes", "on")
        return jsonify(client.set_task_active(bool(active)))

    @app.get("/api/task_presets")
    def task_presets():
        return jsonify({"presets": client.get_task_presets()})

    @app.get("/api/task")
    def get_task():
        return jsonify({"task": client.current_task})

    @app.post("/api/task")
    def set_task():
        data = request.get_json(silent=True)
        task = str(data["task"]) if data and "task" in data else request.get_data(as_text=True).strip()
        client.set_task(task)
        return jsonify({"task": client.current_task})

    @app.get("/api/video")
    def get_video():
        return jsonify({"video_enabled": client.config.webui_video_enabled})

    @app.post("/api/video")
    def set_video():
        data = request.get_json(silent=True) or {}
        enabled = data.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
        return jsonify(client.set_webui_video(bool(enabled)))

    @app.get("/stream/<path:cam>")
    def stream(cam):
        if not client.config.webui_video_enabled:
            return jsonify({"error": "video push disabled"}), 503

        def gen():
            while client.running and client.config.webui_video_enabled:
                frame = client.get_webui_frame(cam)
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.1)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


class WebUIServer:
    """在守护线程中运行的最小 Web 控制台。"""

    def __init__(self, client, host: str = "0.0.0.0", port: int = 8007):
        self._app = _create_app(client)
        self._host = host
        self._port = port
        self._thread = threading.Thread(target=self._run, daemon=True, name="webui")

    def _run(self):
        self._app.run(host=self._host, port=self._port, threaded=True, use_reloader=False)

    @property
    def address(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self):
        self._thread.start()

    def stop(self):
        pass
