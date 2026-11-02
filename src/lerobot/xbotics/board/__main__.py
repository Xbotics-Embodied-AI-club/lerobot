"""``python -m lerobot.xbotics.board install``：把板子装成（或升级成）这一版 LeRobot 对应的样子，见 install.py。"""

import sys

from .install import install


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "install"
    if cmd != "install":
        raise SystemExit(f"不认识的命令：{cmd}（只有 install）")
    install()


if __name__ == "__main__":
    main()
