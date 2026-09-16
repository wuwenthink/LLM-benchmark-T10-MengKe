#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理质量测试工具 —— 纯 Python 启动器
=====================================================
在任意装有 Python 3.9+ 的机器上运行本目录即可:
  Windows : 双击 start.bat
  Linux/Mac/其他 : bash start.sh (或 python3 server.py)

环境变量(可选):
  QTEST_HOST  监听地址, 默认 127.0.0.1(仅本机)。局域网其他设备访问可设 0.0.0.0
  QTEST_PORT  端口, 默认 17889
  QUALITY_DATA_DIR  运行数据目录, 默认本目录 data/
  QTEST_SQL_TABLES  SQL 参考表目录, 默认本目录 tables/
"""
import os
import sys
import time
import socket
import threading
import webbrowser
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
TABLES_DIR = BASE / "tables"
WEB_DIR = BASE / "web"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 默认路径: 题库/结果/历史都在本目录 data/, SQL 参考表在 tables/
os.environ.setdefault("QUALITY_DATA_DIR", str(DATA_DIR))
os.environ.setdefault("QTEST_SQL_TABLES", str(TABLES_DIR))
# 让同目录的 quality_api/quality_engine 可直接 import
sys.path.insert(0, str(BASE))

HOST = os.environ.get("QTEST_HOST", "127.0.0.1")
PORT = int(os.environ.get("QTEST_PORT", "17889"))

# 依赖: fastapi uvicorn httpx (见 requirements.txt)
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from quality_api import router as quality_router

app = FastAPI(title="梦客AI工具箱-推理质量测试", version="1.0.0")
app.include_router(quality_router)


@app.get("/api/snapshot")
async def snapshot():
    """推理服务清单(可选扩展点)。

    本工具默认不扫描任何集群, 因此返回空清单 —— 请在页面端点列表手动填写
    OpenAI 兼容服务地址(如 http://127.0.0.1:8000)。
    若你的部署环境能提供推理实例清单, 按下面的结构返回, 页面上的
    「自动获取运行中推理」按钮即可一键生成端点:

        {"ok": true,
         "hosts": {"<host_id>": {"display_ip": "10.0.0.1"}},
         "inferences": [{"state": "running",
                         "instances": [{"base_url": "http://10.0.0.1:8000",
                                        "model": "my-model"}]}]}
    """
    return JSONResponse({"ok": True, "hosts": {}, "inferences": []})


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/quality")
async def quality_page():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/quality", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def _open_browser():
    url = f"http://127.0.0.1:{PORT}/"
    for _ in range(20):
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
                break
        except OSError:
            time.sleep(0.3)
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    print(f"推理质量测试已启动 -> http://{HOST}:{PORT}/  (Ctrl+C 停止)")
    uvicorn.run(app, host=HOST, port=PORT)
