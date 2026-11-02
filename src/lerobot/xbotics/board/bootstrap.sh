#!/usr/bin/env bash
# 星禾套件板子一键安装：新板子、照旧手册装过的板子都用它（以 root 在板子上敲）：
#   curl -fsSL https://raw.githubusercontent.com/Xbotics-Embodied-AI-club/lerobot/xbotics-v0.6.1/src/lerobot/xbotics/board/bootstrap.sh | bash
# 它装 uv、建 LeRobot 环境（/root/xbotics），再由环境里的 LeRobot 把板子上其余的东西装好。
# 之后升级全部只要一句：xbotics-config update
# XBOTICS_REF=<分支或标签> 改从别的版本装（默认标签 xbotics-v0.6.1）。
set -euo pipefail
REF=${XBOTICS_REF:-xbotics-v0.6.1}
RAW=https://raw.githubusercontent.com/Xbotics-Embodied-AI-club/lerobot/$REF/src/lerobot/xbotics/board
[[ $EUID -eq 0 ]] || { echo "要用 root 运行" >&2; exit 1; }
export HOME=/root PATH=/root/.local/bin:$PATH

if ! command -v uv >/dev/null; then
  echo "· 装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
mkdir -p /root/xbotics && cd /root/xbotics
[[ -f pyproject.toml ]] || curl -fsSL "$RAW/board_env/pyproject.toml" -o pyproject.toml
# 把 pyproject.toml 里 LeRobot 的源指到 $1：标签 xbotics-v* 写 tag，提交号写 rev，其余当分支
set_ref() {
  local kind
  if [[ "$1" =~ ^[0-9a-f]{7,40}$ ]]; then kind=rev; elif [[ "$1" == xbotics-v* ]]; then kind=tag; else kind=branch; fi
  sed -i -E "s#^lerobot = \{ git = \"([^\"]+)\", (tag|branch|rev) = \"[^\"]*\" \}#lerobot = { git = \"\1\", $kind = \"$1\" }#" pyproject.toml
}
[[ -n "${XBOTICS_REF:-}" ]] && set_ref "$REF"
echo "· 建 LeRobot 环境（第一次要下载约 1.5 GB，几分钟）"
uv lock --upgrade-package lerobot
uv sync
exec .venv/bin/python -m lerobot.xbotics.board install
