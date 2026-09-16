#!/usr/bin/env bash
# 推理质量测试 启动脚本 —— Linux / macOS / 其他系统
cd "$(dirname "$0")" || exit 1

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

# ---------------------------------------------------------------
# [2/3] 端口自愈: 17889 被占用(常见于上一次没退干净的服务)时,
#       强制结束占用进程, 复查确认空闲后再启动, 避免
#       "address already in use" 导致启动失败。
#       端口可用 QTEST_PORT 覆盖, 与 server.py 保持一致。
# ---------------------------------------------------------------
PORT="${QTEST_PORT:-17889}"

# 列出正在监听 PORT 的进程号(按可用工具依次回退)
listening_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnpH "sport = :${PORT}" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2
  elif command -v fuser >/dev/null 2>&1; then
    fuser "${PORT}/tcp" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$'
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p"$" {print $7}' | cut -d/ -f1
  fi
}

echo "[2/3] Checking whether port ${PORT} is free..."
tries=0
stuck=0
last_pids=""
while :; do
  pids="$(listening_pids | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')"
  pids="${pids%% }"
  if [ -z "${pids}" ]; then
    echo "   [port] port ${PORT} is free. (checked ${tries} time(s))"
    break
  fi

  tries=$((tries + 1))
  if [ "${pids}" = "${last_pids}" ]; then
    stuck=$((stuck + 1))
  else
    stuck=0
  fi
  last_pids="${pids}"

  if [ "${tries}" -gt 15 ] || [ "${stuck}" -ge 5 ]; then
    echo
    echo "   [port] ERROR: port ${PORT} is still in use by PID(s): ${pids}"
    echo "          Please stop them manually (may need sudo), then re-run start.sh:"
    echo "            lsof -i tcp:${PORT}      # or: ss -ltnp | grep :${PORT}"
    echo "            kill -9 <PID>"
    echo "          Or start on another port:  QTEST_PORT=18000 bash start.sh"
    echo
    exit 1
  fi

  for pid in ${pids}; do
    if [ "${pid}" = "0" ]; then
      echo "   [port] WARNING: port ${PORT} is held by System PID 0, cannot terminate."
      continue
    fi
    name="$(ps -p "${pid}" -o comm= 2>/dev/null)"
    echo "   [port] port ${PORT} is in use by PID ${pid} (${name:-unknown}) - terminating it..."
    if ! kill -9 "${pid}" 2>/dev/null; then
      echo "   [port] WARNING: failed to terminate PID ${pid} (try sudo)."
    fi
  done
  sleep 0.5
done

echo "[3/3] Server is starting, browser will open automatically (if available)..."
( sleep 1.5; xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || true ) &
exec python3 server.py
