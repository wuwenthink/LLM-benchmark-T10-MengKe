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
async def snapshot(box: str | None = None):
    """推理服务清单(扩展点): 本工具自己不扫描集群, 但可以从「梦客工具箱」代取。

    「自动获取运行中推理」按钮会依次尝试:
      1) 本接口不带参数 → 用已知的工具箱地址(环境变量/上次成功用过的/127.0.0.1:17888)代取;
      2) 带 ?box=http://<工具箱>:17888 → 直接代取该地址的 /api/snapshot
         (浏览器直连工具箱会被 CORS 拦住, 所以由本地服务端代理转发)。
    都拿不到时返回空清单, 页面会提示手动填写(任意 OpenAI 兼容服务都支持)。

    工具箱快照结构(原样透传):
        {"hosts": {"<host_id>": {"display_ip": "10.0.0.1"}},
         "inferences": [{"state": "running",
                         "instances": [{"host_id": "...", "api_port": 8000,
                                        "served_model_name": "my-model"}]}]}
    """
    import httpx

    def _candidates():
        out = []
        for k in ("MENGKE_BOX_URL", "QTEST_BOX_URL"):
            v = os.environ.get(k)
            if v:
                out.append(v)
        saved = _load_box_url()
        if saved:
            out.append(saved)
        out.append("http://127.0.0.1:17888")
        if box:
            out.insert(0, box)
        # 去重 + 补协议 + 去掉末尾斜杠
        seen, uniq = set(), []
        for u in out:
            u = (u or "").strip().rstrip("/")
            if not u:
                continue
            if not u.startswith(("http://", "https://")):
                u = "http://" + u
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq

    errors = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0)) as client:
        for base in _candidates():
            try:
                r = await client.get(base + "/api/snapshot")
                if r.status_code != 200:
                    errors.append(f"{base} -> HTTP {r.status_code}")
                    continue
                data = r.json()
                if not isinstance(data, dict):
                    errors.append(f"{base} -> 返回格式不是对象")
                    continue
                _save_box_url(base)          # 记下可用的工具箱地址, 下次直接用
                data["box_url"] = base
                return JSONResponse(data)
            except Exception as e:                      # noqa: BLE001
                errors.append(f"{base} -> {type(e).__name__}: {str(e)[:80]}")
    return JSONResponse({"ok": False, "hosts": {}, "inferences": [],
                         "error": "; ".join(errors[:4]) or "no toolbox url",
                         "hint": "本按钮只能通过梦客工具箱获取API，当前无工具箱启动，请手动填写。"})


# 上次成功用过的梦客工具箱地址(存在 data/box_url.json, 不参与版本控制)
BOX_URL_FILE = DATA_DIR / "box_url.json"


def _load_box_url():
    try:
        import json
        v = json.loads(BOX_URL_FILE.read_text(encoding="utf-8")).get("box_url")
        return v if isinstance(v, str) else None
    except Exception:
        return None


def _save_box_url(url: str):
    try:
        import json
        if _load_box_url() == url:
            return
        BOX_URL_FILE.write_text(json.dumps({"box_url": url}, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


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
