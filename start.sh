#!/usr/bin/env bash
# 推理质量测试 启动脚本 —— Linux / macOS / 其他系统
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "   Inference Quality Test (Python, standalone)"
echo "============================================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Please install Python 3.9+ and retry."
  exit 1
fi

echo "[1/3] Preparing virtual environment + dependencies (first run may take a while)..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python3 -m pip install -q -r requirements.txt

echo "[2/3] Starting server -> http://127.0.0.1:17889/  (Ctrl+C to stop)"
( sleep 1.5; xdg-open http://127.0.0.1:17889/ >/dev/null 2>&1 || open http://127.0.0.1:17889/ >/dev/null 2>&1 || true ) &

echo "[3/3] Server is starting, browser will open automatically (if available)..."
exec python3 server.py
