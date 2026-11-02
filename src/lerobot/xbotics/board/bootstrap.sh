#!/usr/bin/env bash
# 星禾套件端侧板一键安装，适用于新端侧板以及按旧版手册安装过的端侧板（以 root 在端侧板上执行）：
#   curl -fsSL https://raw.githubusercontent.com/Xbotics-Embodied-AI-club/lerobot/xbotics-v0.6.1/src/lerobot/xbotics/board/bootstrap.sh | bash
# 它安装 uv、建立 LeRobot 环境（/root/xbotics），再由环境中的 LeRobot 安装端侧板上的其余内容。
# 之后升级全部只需执行：xbotics-config update
# 设 XBOTICS_REF=<分支、标签或提交> 可从其他版本安装（默认标签 xbotics-v0.6.1）。
set -euo pipefail
REF=${XBOTICS_REF:-xbotics-v0.6.1}
RAW=https://raw.githubusercontent.com/Xbotics-Embodied-AI-club/lerobot/$REF/src/lerobot/xbotics/board
[[ $EUID -eq 0 ]] || { echo "须以 root 运行" >&2; exit 1; }
export HOME=/root PATH=/root/.local/bin:$PATH

if ! command -v uv >/dev/null; then
  echo "· 安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
mkdir -p /root/xbotics && cd /root/xbotics
[[ -f pyproject.toml ]] || curl -fsSL "$RAW/board_env/pyproject.toml" -o pyproject.toml
# 将 pyproject.toml 中 LeRobot 的源指向 $1：标签 xbotics-v* 写作 tag，提交号写作 rev，其余视为分支
set_ref() {
  local kind
  if [[ "$1" =~ ^[0-9a-f]{7,40}$ ]]; then kind=rev; elif [[ "$1" == xbotics-v* ]]; then kind=tag; else kind=branch; fi
  sed -i -E "s#^lerobot = \{ git = \"([^\"]+)\", (tag|branch|rev) = \"[^\"]*\" \}#lerobot = { git = \"\1\", $kind = \"$1\" }#" pyproject.toml
}
[[ -n "${XBOTICS_REF:-}" ]] && set_ref "$REF"
echo "· 建立 LeRobot 环境（首次需下载约 1.5 GB，需要几分钟）"
uv lock --upgrade-package lerobot
uv sync
exec .venv/bin/python -m lerobot.xbotics.board install
