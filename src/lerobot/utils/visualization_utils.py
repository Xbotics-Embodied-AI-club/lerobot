# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-agnostic visualization dispatch.

Selects a visualization backend at runtime via a display-mode string (e.g. a ``--display_mode`` CLI
flag) so callers never branch on the backend. The concrete implementations live in
:mod:`lerobot.utils.rerun_visualization` and :mod:`lerobot.utils.foxglove_visualization`; importing
this module does not import ``rerun`` or ``foxglove`` (each backend imports its SDK lazily behind a
``require_package`` guard).
"""

import logging
import os

from lerobot.lerobot_types import RobotAction, RobotObservation

from .foxglove_visualization import init_foxglove, log_foxglove_data, shutdown_foxglove
from .rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun
from .rerun_web_visualization import init_rerun_web, log_rerun_web_data, shutdown_rerun_web

# Visualization backends selectable at runtime via a display-mode string (e.g. a --display_mode flag).
# "web"（Xbotics）：板子自己出网页，浏览器打开 http://<板子>:9090 看，见 rerun_web_visualization。
VISUALIZATION_MODES = ("rerun", "foxglove", "web")

# Xbotics：没有图形桌面（板子不开桌面、经 SSH / 网页终端操作）又没指定远端 rerun 服务时，
# "rerun" 拉不起原生查看器，自动改走 "web"。之后的 log / shutdown 跟着走 "web"。
_web_override = False


def _has_desktop() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _resolve(display_mode: str) -> str:
    return "web" if display_mode == "rerun" and _web_override else display_mode


def init_visualization(
    display_mode: str,
    *,
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
) -> None:
    """Initializes the visualization backend selected by ``display_mode``.

    For ``"rerun"``, ``ip``/``port`` point at an optional remote Rerun server. For ``"foxglove"``,
    ``ip`` is the interface to bind the WebSocket server to (``127.0.0.1`` for local only, ``0.0.0.0``
    for all interfaces) and ``port`` is its port.
    """

    global _web_override
    if display_mode == "rerun" and not (ip and port) and not _has_desktop():
        logging.info("没有图形桌面：rerun 改为板子出网页预览（display_mode=web）")
        _web_override = True
    display_mode = _resolve(display_mode)

    if display_mode == "rerun":
        init_rerun(session_name=session_name, ip=ip, port=port)
    elif display_mode == "foxglove":
        init_foxglove(host=ip or "127.0.0.1", port=port)
    elif display_mode == "web":
        init_rerun_web(web_port=port)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def log_visualization_data(
    display_mode: str,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Logs observation/action data to the backend selected by ``display_mode``."""

    display_mode = _resolve(display_mode)
    if display_mode == "rerun":
        log_rerun_data(observation=observation, action=action, compress_images=compress_images)
    elif display_mode == "foxglove":
        log_foxglove_data(observation=observation, action=action, compress_images=compress_images)
    elif display_mode == "web":
        log_rerun_web_data(observation=observation, action=action, compress_images=compress_images)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def shutdown_visualization(display_mode: str) -> None:
    """Shuts down the backend selected by ``display_mode``."""

    global _web_override
    resolved = _resolve(display_mode)
    _web_override = False
    if resolved == "rerun":
        shutdown_rerun()
    elif resolved == "foxglove":
        shutdown_foxglove()
    elif resolved == "web":
        shutdown_rerun_web()
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")
