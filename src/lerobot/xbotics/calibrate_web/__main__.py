"""``lerobot-calibrate-web``：SO101 主臂 / 从臂的网页交互式标定。

默认监听全部网卡的 8006（板子上的网页服务连号 8001–8007），学员在自己电脑的浏览器里打开
``http://<板子>:8006``，页面里选主臂或从臂、选串口，按 URDF 动画一步步标。标定文件写到与
``lerobot-calibrate`` 相同的位置（``$HF_LEROBOT_CALIBRATION/{robots,teleoperators}/<类型>/<id>.json``），
所以之后录数据、推理时 ``--robot.id`` / ``--teleop.id`` 给同一个 id 就能直接用。

::

    lerobot-calibrate-web                                   # 板子上：串口默认取 /dev/xbotics/{leader,follower}
    lerobot-calibrate-web --follower-port /dev/ttyACM0 --follower-id my_follower --leader-id my_leader
    lerobot-calibrate-web --demo --host 127.0.0.1 --open-browser   # 不接硬件，把流程走一遍
"""

import argparse
from pathlib import Path

from .app import run_server
from .devices import ArmSettings, default_port

DEFAULT_PORT = 8006


def main():
    parser = argparse.ArgumentParser(description="星禾套件网页标定")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0（局域网可访问）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口，默认 {DEFAULT_PORT}")
    parser.add_argument("--leader-port", default=None, help="主臂串口，默认 /dev/xbotics/leader（存在时）")
    parser.add_argument(
        "--leader-id", default="leader", help="主臂 id（= --teleop.id），默认 leader（与板子使用手册一致）"
    )
    parser.add_argument(
        "--follower-port", default=None, help="从臂串口，默认 /dev/xbotics/follower（存在时）"
    )
    parser.add_argument(
        "--follower-id",
        default="follower",
        help="从臂 id（= --robot.id），默认 follower（与板子使用手册一致）",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="标定文件目录（= --robot.calibration_dir），默认与 lerobot-calibrate 相同",
    )
    parser.add_argument(
        "--demo", action="store_true", help="页面默认勾上演示模式：模拟舵机，不碰串口和标定文件"
    )
    parser.add_argument("--open-browser", action="store_true", help="启动后在本机打开浏览器")
    args = parser.parse_args()

    arms = {
        "leader": ArmSettings(
            type="so101_leader",
            port=args.leader_port if args.leader_port is not None else default_port("leader"),
            id=args.leader_id,
            calibration_dir=args.calibration_dir,
        ),
        "follower": ArmSettings(
            type="so101_follower",
            port=args.follower_port if args.follower_port is not None else default_port("follower"),
            id=args.follower_id,
            calibration_dir=args.calibration_dir,
        ),
    }
    run_server(arms, host=args.host, port=args.port, demo=args.demo, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
