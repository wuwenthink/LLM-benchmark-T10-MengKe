#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推理质量测试 - 后端 API
GET  /api/quality/meta        题集/档位/领域-类别元数据
GET  /api/quality/questions?tier=T1|domain=xxx&benchmark=yyy  题目清单
GET  /api/quality/question?qid=xxx   单题完整内容 + 所属档位 (tiers)
POST /api/quality/run         启动评测 {tier, endpoints:[{name,base_url,model,thinking}], concurrency, subjective_judge:{enabled, base_url, model}}
GET  /api/quality/status      当前运行状态/进度/每端点进度
GET  /api/quality/results?run_id=xxx   运行结果 (含 partial 标志)
POST /api/quality/cancel      取消当前运行（部分结果保留落盘）
GET  /api/quality/history     历史运行列表（含状态/已完成数）
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import threading
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import quality_engine as qe
from benchmark_guard import set_active as guard_set_active

router = APIRouter(prefix="/api/quality", tags=["quality"])

# 默认数据目录 = 本目录 data/ (server.py 会用 QUALITY_DATA_DIR 显式指定; 环境变量优先)
DATA_DIR = Path(os.environ.get("QUALITY_DATA_DIR") or (Path(__file__).resolve().parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
QUESTION_PATH = DATA_DIR / "questions.jsonl"
# SQL 单独测试档位题集(25 题, 随代码发布, 与题集合并)
SQL_Q_PATH = Path(__file__).resolve().parent / "sql_questions.jsonl"

# 运行时状态 (单进程内存)
_state = {"running": False, "cancel": None, "progress": {}, "per_ep": {}, "current": None,
          "results": {}, "run_id": 0, "last_error": None,
          # 实时进行记录(内存, 最近 500 条精简字段), 供 /api/quality/live 轮询
          "live": deque(maxlen=500),
          # 实时热力图用: 在途 {"qid|round|ep": 起始ts}, 端点名
          # 2026-09-14: 骨架改为「全量 + 一次性下发」(见 skel/skeleton 路由), 不再随 /live 反复传,
          # 也不再限 800 题(超过 800 题的档位此前拿不到骨架 → 格子数与题目数对不上)。
          # 另加 cells/cell_seq: 已判定格子的紧凑增量表 [qid, ep, round, code, score]，
          # 供页面中途刷新/后开也能补齐全部格子(旧版只有 live deque 500 条 → 刷新后格子凭空少一大截)。
          # code: 1 通过 / 2 失败(0 分) / 3 错误 / 4 已答未评分 / 5 部分得分(0<score<1)。
          "live_run": {}, "items_lite": None, "endpoints_lite": [],
          "rounds": 1, "skel": None, "skel_total": 0,
          "cells": [], "cell_seq": 0}
_lock = threading.Lock()

# 档位成员索引缓存：{tier: set(qid)}，按题集 mtime 失效
_tier_index: Optional[dict[str, set[str]]] = None
_tier_index_mtime: Optional[int] = None


_QS_CACHE: dict = {"key": None, "qs": None}
_META_CACHE: dict = {"key": None, "val": None}


def _load_questions() -> list[dict]:
    """mtime+size 缓存: 题库解析(整包 JSONL, NAS 弱 CPU 上可达秒级)只做一次,
    否则每次 /meta 都全量 json.loads 并阻塞事件循环 → /live /status 全被拖住,
    实时热力图卡死数秒后跳变(2026-09-13 根因修复)。"""
    key = []
    for p in (QUESTION_PATH, SQL_Q_PATH):
        try:
            st = p.stat()
            key += (st.st_mtime_ns, st.st_size)
        except OSError:
            key += (0, 0)
    key = tuple(key)
    if _QS_CACHE["key"] == key and _QS_CACHE["qs"] is not None:
        return _QS_CACHE["qs"]
    qs = _load_questions_sync()
    _QS_CACHE["key"] = key
    _QS_CACHE["qs"] = qs
    return qs


def _load_questions_sync() -> list[dict]:
    qs = []
    if QUESTION_PATH.exists():
        with open(QUESTION_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    qs.append(json.loads(line))
    # 合入 SQL 单独测试题集(按 id 去重, SQL 题优先稳定在后)
    if SQL_Q_PATH.exists():
        have = {q.get("id") for q in qs}
        with open(SQL_Q_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if o.get("id") not in have:
                    qs.append(o)
    return qs


def _summary(q: dict) -> dict:
    return {"id": q["id"], "benchmark": q["benchmark"], "domain": q["domain"],
            "difficulty": q["difficulty"], "lang": q.get("lang", "en"),
            "judge": q["judge"]["type"]}


def _tier_index_lazy(qs: list[dict]) -> dict[str, set[str]]:
    """全量档位成员索引（懒构建 + 按题集 mtime 失效）。10 档 × 11k 题 O(n) 可接受。"""
    global _tier_index, _tier_index_mtime
    try:
        mtime = QUESTION_PATH.stat().st_mtime_ns
    except OSError:
        mtime = None
    if _tier_index is not None and _tier_index_mtime == mtime:
        return _tier_index
    idx: dict[str, set[str]] = {}
    for name, budget in qe.TIERS:
        t = qe.build_tier(qs, name, budget, per_bm_cap=5000)
        idx[name] = {q["id"] for q in t["items"]}
    _tier_index = idx
    _tier_index_mtime = mtime
    return idx


def _tiers_of(qs: list[dict], qid: str) -> list[str]:
    idx = _tier_index_lazy(qs)
    return [t for t, ids in idx.items() if qid in ids]


# ---------- 请求模型 ----------
class EndpointIn(BaseModel):
    name: str
    base_url: str
    model: str
    thinking: bool = False
    api_key: Optional[str] = None  # 可选: OpenAI 兼容端点需要鉴权时使用


class RunRequest(BaseModel):
    tier: str
    endpoints: list[EndpointIn]
    # 并发不设上限(ge=1 仅为避免除零/无 worker 的空跑), 多大都能生效
    concurrency: int = Field(default=2, ge=1)
    subjective_judge: Optional[dict] = None  # {enabled, base_url, model}
    sample_n: Optional[int] = Field(default=None, ge=0)  # 可选: 每题限制/全量调试
    # ----- 可调参数(不设上下限, 由使用者自行权衡) -----
    rounds: int = Field(default=1, ge=1)  # 测试轮数(同一题集重复评测, 热力图行=轮次)
    max_tokens: int = Field(default=2048, ge=1)  # 生成长度
    temperature: float = Field(default=0.1)  # 温度
    timeout: float = Field(default=300.0, ge=1.0)  # 单个测试总超时(秒)
    retries: int = Field(default=2, ge=0)  # 单题失败最大重试次数
    difficulty_filter: Optional[list[str]] = None  # None=全部; 否则只含这些难度 (easy/medium/hard)
    resume_run_id: Optional[int] = None  # 断点续测: 继续指定历史运行, 自动跳过已完成题轮


# ---------- 工具 ----------
def _ep_meta(e: "EndpointIn") -> dict:
    """run 元数据里的端点描述: 不落盘 api_key(防止 history 泄露), 仅记 has_key。"""
    d = e.model_dump()
    d.pop("api_key", None)
    d["has_key"] = bool(e.api_key)
    return d


def _dedup_results(lst: list) -> list:
    """按 (qid, round, endpoint) 去重, 后写覆盖先写, 顺序稳定(续跑合并用)。"""
    idx: dict = {}
    out: list = []
    for r in lst:
        k = (r.get("qid"), int(r.get("round", 1) or 1), r.get("endpoint"))
        if k in idx:
            out[idx[k]] = r
        else:
            idx[k] = len(out)
            out.append(r)
    return out


# ---------- 路由 ----------
@router.get("/meta")
async def meta():
    val = await asyncio.to_thread(_meta_compute)
    if val is None:
        return {"ok": False, "error": "题集未就绪", "question_file": str(QUESTION_PATH)}
    return val


def _meta_compute():
    """11 个档位 build_tier 合计 ~7s(万题), 必须整块线程化 + 按题集指纹缓存,
    否则每次开质量页都阻塞事件循环 7 秒 → /live /status 全部卡住, 热力图冻结跳变。"""
    qs = _load_questions()
    if not qs:
        return None
    key = _QS_CACHE.get("key")
    if _META_CACHE["key"] == key and _META_CACHE["val"] is not None:
        return _META_CACHE["val"]
    val = _meta_build(qs)
    _META_CACHE["key"] = key
    _META_CACHE["val"] = val
    return val


def _meta_build(qs):
    tiers = []
    for name, budget in qe.TIERS:
        t = qe.build_tier(qs, name, budget, per_bm_cap=5000)
        tiers.append({"tier": name, "budget_min": budget, "n": t["n"],
                      "est_wall_min": round(t["est_wall_min"], 1),
                      "diff": t["diff_count"]})
    # 领域 → 类别(benchmark) 树
    domains: dict[str, dict[str, dict]] = {}
    for q in qs:
        d = domains.setdefault(q["domain"], {})
        b = d.setdefault(q["benchmark"], {"count": 0, "by_diff": Counter()})
        b["count"] += 1
        b["by_diff"][q.get("difficulty", "medium")] += 1
    benchmarks = {
        d: [{"benchmark": k, "count": v["count"], "by_diff": dict(v["by_diff"])}
            for k, v in sorted(items.items(), key=lambda kv: (-kv[1]["count"], kv[0]))]
        for d, items in sorted(domains.items())
    }
    return {"ok": True, "total": len(qs),
            "domains": dict(Counter(q["domain"] for q in qs)),
            "benchmarks": benchmarks,
            "judges": dict(Counter(q["judge"]["type"] for q in qs)),
            "tiers": tiers}


@router.get("/questions")
async def questions(tier: str = "T1", domain: str | None = None, benchmark: str | None = None,
                    with_tiers: bool = False, limit: int = Query(0, ge=0, le=20000),
                    offset: int = Query(0, ge=0)):
    """题目清单。两种模式：
    - ?tier=T1 ：档位题目摘要（向后兼容）
    - ?domain=xxx[&benchmark=yyy] ：领域/类别下全部题目（题库浏览），可选 with_tiers 带档位
    """
    qs = await asyncio.to_thread(_load_questions)
    if not qs:
        raise HTTPException(404, "题集未就绪")
    if domain:
        items = [q for q in qs if q.get("domain") == domain]
        if benchmark:
            items = [q for q in items if q.get("benchmark") == benchmark]
        summaries = [_summary(q) for q in items]
        if with_tiers:
            idx = _tier_index_lazy(qs)
            tiers_by_id = {tier_name: ids for tier_name, ids in idx.items()}
            for s in summaries:
                s["tiers"] = [t for t, ids in tiers_by_id.items() if s["id"] in ids]
        total = len(summaries)
        if offset:
            summaries = summaries[offset:]
        if limit:
            summaries = summaries[:limit]
        return {"ok": True, "domain": domain, "benchmark": benchmark,
                "n": total, "items": summaries}
    t = await asyncio.to_thread(qe.build_tier, qs, tier, dict(qe.TIERS).get(tier, 10), per_bm_cap=5000)
    items = [_summary(q) for q in t["items"]]
    if limit and limit > 0:
        items = items[offset: offset + limit] if offset else items[:limit]
    return {"ok": True, "tier": tier, "n": t["n"], "items": items}


@router.get("/question")
async def question(qid: str):
    """按 id 取完整题目 (prompt/answer/judge) + 所属档位 tiers, 供 inspector 展示"""
    qs = await asyncio.to_thread(_load_questions)
    for q in qs:
        if q.get("id") == qid:
            out = dict(q)
            try:
                out["tiers"] = _tiers_of(qs, qid)
            except Exception:
                out["tiers"] = []
            return {"ok": True, "question": out}
    raise HTTPException(404, "题目不存在")


@router.post("/run")
async def start_run(req: RunRequest):
    qs = await asyncio.to_thread(_load_questions)
    if not qs:
        raise HTTPException(400, "题集未就绪")
    # 端点数量上限: 防止历史遗留的超多端点配置同时压打推理端点
    # (40 端点 × 每端点并发 = 请求风暴, 推理持续排队无速度)
    if len(req.endpoints) < 1:
        raise HTTPException(400, "至少需要一个评测端点")
    if len(req.endpoints) > 8:
        raise HTTPException(400, f"端点数量上限 8 个（当前 {len(req.endpoints)} 个），请精简后再启动")
    # 测试名称默认 = 模型名称: 前端留空时会带上模型名, 但直接调 API/旧配置可能传空名,
    # 这里兜底, 避免热力图行标签、历史、结果里出现空字符串名称。
    for ep in req.endpoints:
        if not (ep.name or "").strip():
            ep.name = (ep.model or "").strip() or ep.base_url.strip() or "endpoint"
    # 同名端点区分: 多端点同名(常见于同一模型跑多机, 名称都默认成模型名) -> 追加 @主机:端口,
    # 否则热力图行标签/统计/历史会混成一条, 分不清是哪台机器。
    _names = [e.name.strip() for e in req.endpoints]
    _dup = {n for n in _names if _names.count(n) > 1}
    if _dup:
        for e in req.endpoints:
            if e.name.strip() in _dup:
                _host = e.base_url.split("//")[-1].split("/")[0].strip()
                if _host:
                    e.name = f"{e.name.strip()}@{_host}"
    seen = set()
    for ep in req.endpoints:
        key = ep.base_url.strip().rstrip("/")
        if key in seen:
            raise HTTPException(400, f"端点重复（同一推理实例 {key} 只能测一个条目）: {ep.name}")
        seen.add(key)
    # ----- 断点续测: 校验源运行并强制沿用其采样口径(tier/轮数/难度), 保证题集与原来完全一致 -----
    resume_meta = None
    if req.resume_run_id:
        src_p = DATA_DIR / f"run_{req.resume_run_id}.json"
        if not src_p.exists():
            raise HTTPException(404, f"找不到要续跑的运行 #{req.resume_run_id}")
        try:
            resume_meta = json.loads(src_p.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(500, "续跑运行的元数据损坏")
        if resume_meta.get("status") == "running" and _state["running"]:
            raise HTTPException(409, "评测正在运行中, 无法续跑")
        # status=running 但后台没在跑 = 崩溃残留, 允许续跑
        req.tier = resume_meta.get("tier") or req.tier
        req.rounds = int(resume_meta.get("rounds") or 1)
        req.difficulty_filter = resume_meta.get("difficulty_filter")
    budget = dict(qe.TIERS).get(req.tier, 10)
    if req.tier == "T10":
        budget = None
    # 纯校验放在占位之前: 若放进锁内(running 已置 True), raise 会把整个评测模块永久锁死
    # 2026-09-14: 取消「SQL 档必须启用评委」硬门槛。SQL 判分 = 真实执行
    # SQL 与题面 row_count/columns/first_row 比对, 全程无 LLM 评委(sqlite 引擎),
    # 评委对 SQL 只是可选的第二意见; 极端情况下(参考表缺失)未启用评委的 SQL 题会落
    # 「已答未评分」琥珀格, 属正常显示, 不该拦用户起跑。
    if req.tier == "SQL" and qe.sql_tables_dir() is None:
        # 不拦, 只告知(前端把这句显示在档位行)
        _state["sql_note"] = ("未找到 SQL 参考数据表(7 张 CSV, 应在 tables/ 目录), "
                              "本机执行判分不可用: 未启用评委时 SQL 题将显示「已答未评分」。")
    else:
        _state["sql_note"] = None
    with _lock:
        if _state["running"]:
            raise HTTPException(409, "已有评测运行中")
        _state["running"] = True
        # 2026-09-16 修复(停止后立刻续跑/新测 → 上一轮任务复活并回写结果文件):
        # 旧实现所有运行复用同一个 cancel Event, 新运行里的 clear() 会把「刚被停止的上一轮」
        # 重新激活 —— 上一轮的后台任务从 asyncio.wait 醒来后继续跑完整档位, 并与新任务
        # 并发写同一份 run_N_results.json / run_N.json
        # (实测: 停止 → 立刻续跑 → 续跑写好的 25 条被旧任务回写成 3~5 条, 历史永久卡在 running)。
        # 现在每次运行各自持有独立 Event, 任何运行都清不掉别人的停止标志。
        run_cancel = asyncio.Event()
        _state["cancel"] = run_cancel
        if resume_meta:
            run_id = req.resume_run_id
            _state["run_id"] = run_id
        else:
            _state["run_id"] += 1
            # 与磁盘已有结果对齐 (避免重启后从 1 重来覆盖旧结果)
            try:
                max_disk = max(int(p.name.split("_")[1]) for p in DATA_DIR.glob("run_*_results.json"))
            except ValueError:
                max_disk = 0
            _state["run_id"] = max(_state["run_id"], max_disk + 1)
            run_id = _state["run_id"]
        _state["progress"] = {"done": 0, "total": 0}
        _state["per_ep"] = {}
        _state["live"].clear()
        _state["last_error"] = None
        # 档位构建 & 落盘 run 描述
        tier_cfg = await asyncio.to_thread(qe.build_tier, qs, req.tier, budget, per_bm_cap=5000)
        if req.difficulty_filter:
            df = set(req.difficulty_filter)
            tier_cfg["items"] = [i for i in tier_cfg["items"] if i.get("difficulty") in df]
            tier_cfg["n"] = len(tier_cfg["items"])
        if req.sample_n and req.sample_n > 0:
            tier_cfg["items"] = tier_cfg["items"][: req.sample_n]
            tier_cfg["n"] = len(tier_cfg["items"])
        if not tier_cfg["items"]:
            _state["running"] = False
            raise HTTPException(400, "当前档位+难度过滤后没有可测题目")
        # ----- 续跑: 计算已完成 (qid,round), 只把未完成题轮交给引擎 -----
        old_res: list = []
        done_start = 0
        total_all = tier_cfg["n"] * req.rounds
        est_min = tier_cfg["est_wall_min"]
        items_all = tier_cfg["items"]
        if resume_meta:
            want_n = int(resume_meta.get("n") or 0)
            if want_n and len(tier_cfg["items"]) < want_n:
                _state["running"] = False
                raise HTTPException(400, "题库/档位构成已变化, 原题目集无法复现, 不能续跑")
            if want_n:
                tier_cfg["items"] = tier_cfg["items"][:want_n]
                tier_cfg["n"] = want_n
            items_all = tier_cfg["items"]
            total_all = tier_cfg["n"] * req.rounds
            ep_names = {e.name for e in req.endpoints}
            if ep_names != {x.get("name") for x in resume_meta.get("endpoints", [])}:
                _state["running"] = False
                raise HTTPException(400, "续跑端点必须与原运行一致(按名称匹配), 请调整端点后重试")
            rp = DATA_DIR / f"run_{run_id}_results.json"
            if rp.exists():
                try:
                    old_res = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    old_res = []
            rounds_done: dict = {}
            for r in old_res:
                k = (r.get("qid"), int(r.get("round", 1) or 1))
                rounds_done.setdefault(k, set()).add(r.get("endpoint"))
            full = {k for k, v in rounds_done.items() if ep_names <= v}
            done_start = sum(1 for it in items_all for rd in range(1, req.rounds + 1)
                             if (it["id"], rd) in full)
            tier_cfg["items"] = [it for it in items_all
                                 if any((it["id"], rd) not in full for rd in range(1, req.rounds + 1))]
            if not tier_cfg["items"]:
                _state["running"] = False
                raise HTTPException(400, f"运行 #{run_id} 已全部完成({done_start} 题轮), 无需续跑")
            est_min = tier_cfg["est_wall_min"] * (len(tier_cfg["items"]) * req.rounds) / max(1, total_all)
        # 实时热力图骨架: 一次性全量存内存 + 由 /skeleton 下发(核心不变式: 格子数量只由骨架
        # 决定，永不由结果决定)。旧版只在 ≤800 题时随 /live 传骨架 → 大档位拿不到骨架,
        # 前端只能"结果来一格补一格", 格子数永远比题目数少(用户报的数量不对)。
        _state["live_run"] = {}
        _state["cells"] = []
        _state["cell_seq"] = 0
        _sk_q = [{"qid": i["id"], "benchmark": i["benchmark"],
                  "domain": i.get("domain", ""), "difficulty": i.get("difficulty", "medium")}
                 for i in items_all]
        _state["items_lite"] = _sk_q if len(_sk_q) <= 800 else None   # 兼容旧页面缓存
        _state["endpoints_lite"] = [e.name for e in req.endpoints]
        _state["rounds"] = max(1, int(req.rounds or 1))
        _state["skel"] = {"run_id": run_id, "questions": _sk_q,
                          "endpoints": _state["endpoints_lite"], "rounds": _state["rounds"],
                          "concurrency": int(req.concurrency or 1),
                          "total_cells": len(_sk_q) * len(req.endpoints) * _state["rounds"],
                          "tier": req.tier}
        _state["skel_total"] = _state["skel"]["total_cells"]
        _state["concurrency"] = int(req.concurrency or 1)
        # 骨架落盘: 历史回看/续跑的格子数与运行中完全一致(修复前不落盘 → 回看只能从结果猜)。
        try:
            (DATA_DIR / f"run_{run_id}_skeleton.json").write_text(
                json.dumps(_state["skel"], ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        _state["current"] = {"run_id": run_id, "tier": req.tier, "started": time.time(),
                             "est_wall_min": round(est_min, 1),
                             "n": tier_cfg["n"]}
        if resume_meta:
            run_rec = resume_meta
            run_rec["status"] = "running"
            run_rec["resumed_at"] = time.time()
            run_rec["resume_count"] = int(resume_meta.get("resume_count") or 0) + 1
            run_rec["endpoints"] = [_ep_meta(e) for e in req.endpoints]
            run_rec["concurrency"] = req.concurrency
            run_rec["subjective_judge"] = req.subjective_judge
            run_rec["max_tokens"] = req.max_tokens
            run_rec["temperature"] = req.temperature
            run_rec["timeout"] = req.timeout
            run_rec["retries"] = req.retries
            run_rec["n_done"] = len(old_res)
            run_rec.pop("finished", None)
        else:
            run_rec = {"run_id": run_id, "tier": req.tier, "started": time.time(),
                       "endpoints": [_ep_meta(e) for e in req.endpoints],
                       "concurrency": req.concurrency, "subjective_judge": req.subjective_judge,
                       "sample_n": req.sample_n,
                       "rounds": req.rounds, "max_tokens": req.max_tokens,
                       "temperature": req.temperature, "timeout": req.timeout, "retries": req.retries,
                       "difficulty_filter": req.difficulty_filter,
                       "status": "running", "n_done": 0}
            run_rec["n"] = tier_cfg["n"]
        (DATA_DIR / f"run_{run_id}.json").write_text(
            json.dumps(run_rec, ensure_ascii=False, indent=1), encoding="utf-8")

    endpoints = [{"name": e.name, "base_url": e.base_url, "model": e.model,
                  "thinking": e.thinking, "api_key": e.api_key} for e in req.endpoints]
    judge_ep = None
    if req.subjective_judge and req.subjective_judge.get("enabled"):
        judge_ep = {"base_url": req.subjective_judge["base_url"],
                    "model": req.subjective_judge["model"],
                    # 公共 API 评委(DeepSeek/OpenAI 等)需要 Bearer Key, 与端点行一样支持
                    "api_key": req.subjective_judge.get("api_key")}

    def progress_cb(done, total):
        if resume_meta:
            _state["progress"] = {"done": min(done_start + done, total_all), "total": total_all}
        else:
            _state["progress"] = {"done": done, "total": total}

    per_ep_init: dict = {}
    if resume_meta:
        for r in old_res:
            per_ep_init[r.get("endpoint")] = per_ep_init.get(r.get("endpoint"), 0) + 1

    def per_ep_cb(name, done, total):
        if resume_meta:
            _state["per_ep"][name] = {"done": min(per_ep_init.get(name, 0) + done, total_all),
                                      "total": total_all}
        else:
            _state["per_ep"][name] = {"done": done, "total": total}

    def _live_item(q: dict) -> dict:
        """实时面板用精简条目(不含 output/reasoning, 避免轮询流量爆炸)。"""
        return {"qid": q.get("qid"), "benchmark": q.get("benchmark"),
                "domain": q.get("domain"), "difficulty": q.get("difficulty"),
                "endpoint": q.get("endpoint"), "score": q.get("score"),
                "latency": q.get("latency"), "error": q.get("error"),
                "finished": q.get("output") is not None,
                "round": q.get("round", 1),
                "ts": round(time.time(), 3)}

    def _cell_code(q: dict) -> int:
        """格子终态编码(照搬参考页 5+1 态中已判定的部分): 1 pass / 2 fail / 3 error / 4 已答未评分 /
        5 部分得分(0<score<1, 主观题/多轮工具题的 0.1~0.9 不再被压成 0 或 1)。
        返回 0 = 不算终态(cancelled 等 → 骨架里保持「未完成」)。"""
        err = q.get("error")
        if err:
            return 0 if str(err) == "cancelled" else 3
        s = q.get("score")
        if s is None:
            return 4 if q.get("output") is not None else 0
        try:
            v = float(s)
        except (TypeError, ValueError):
            return 4 if q.get("output") is not None else 0
        if v >= 1.0:
            return 1
        return 5 if v > 0 else 2

    def _cells_extend(qres: list) -> None:
        """把一批已完成结果压成紧凑格子记录(供 /live?since= 增量下发)。
        元组: [qid, endpoint, round, code, score] —— score 供前端把 0.1~0.9 直接写在格子里。"""
        buf = _state["cells"]
        for q in qres:
            code = _cell_code(q)
            if not code:
                continue
            buf.append([q.get("qid"), q.get("endpoint"), int(q.get("round") or 1), code,
                        q.get("score")])
        _state["cell_seq"] = len(buf)

    results_ref: dict = {}

    def _write_results_disk(rid: int, results: list) -> None:
        path = DATA_DIR / f"run_{rid}_results.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        os.replace(tmp, path)

    def _write_meta(rid: int, status: str, n_done: int) -> None:
        p = DATA_DIR / f"run_{rid}.json"
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            rec = {"run_id": rid}
        rec["status"] = status
        rec["n_done"] = n_done
        if status in ("done", "cancelled", "error"):
            rec.setdefault("finished", time.time())
        else:
            # 状态回到 running(进度落盘)时必须清掉旧的 finished, 否则历史卡片自相矛盾
            rec.pop("finished", None)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    async def _bg():
        status = "error"
        try:
            persist = {"last": 0.0, "dirty": 0, "buf": list(old_res)}  # 全量累积, 落盘=去重后的全集

            async def on_partial(qres):
                """进度回调里节流落盘：停止/重启后部分结果不丢。"""
                _state["live"].extend(_live_item(q) for q in qres)
                _cells_extend(qres)
                # 该题(该轮)全部端点完成 -> 兜底清掉该题轮所有端点的「进行中」标记
                if qres:
                    for _epn in (_state.get("endpoints_lite") or []):
                        _state["live_run"].pop(f"{qres[0].get('qid')}|{qres[0].get('round', 1)}|{_epn}", None)
                persist["buf"].extend(qres)
                persist["dirty"] += len(qres)
                now = time.time()
                # 2026-09-13 修复: 旧实现 flush 后清空 buf → 结果文件被覆盖成"最近 15s
                # 窗口"(崩溃即丢历史部分结果; 历史卡 n_done 也只有窗口条数, 永远显示 1-2)。
                # 正确做法: buf 永久全量累积, 每次落盘写去重后的全集。
                if now - persist["last"] >= 15 or persist["dirty"] >= 300:
                    persist["last"] = now
                    persist["dirty"] = 0
                    dd = _dedup_results(list(persist["buf"]))
                    await asyncio.to_thread(_write_results_disk, run_id, dd)
                    _write_meta(run_id, "running", len(dd))

            def on_start_cb(qid, rnd, ep_name):
                """engine 单端点开始回调: 记录该题该轮该端点正在跑(热力图「进行中」标记)。
                值=起始时间戳, /live 按它排序 → 前端每行只把「最早在途」那一格涂蓝(照搬参考页
                「一行恒 1 个运行中」), 其余并发在途用文字表达, 不再出现多格同时闪。"""
                _state["live_run"][f"{qid}|{rnd}|{ep_name}"] = time.time()

            def on_ep_done_cb(qid, rnd, ep_name):
                _state["live_run"].pop(f"{qid}|{rnd}|{ep_name}", None)

            _trip: dict = {}
            res = await qe.run_tier(qs, tier_cfg, endpoints,
                                    concurrency=req.concurrency,
                                    judge_endpoint=judge_ep,
                                    on_progress=progress_cb,
                                    cancel_flag=run_cancel,
                                    results_cb=on_partial,
                                    on_per_ep=per_ep_cb,
                                    on_start=on_start_cb,
                                    on_ep_done=on_ep_done_cb,
                                    trip_info=_trip,
                                    rounds=req.rounds,
                                    max_tokens=req.max_tokens,
                                    temperature=req.temperature,
                                    timeout=req.timeout,
                                    retries=req.retries)
            res_full = _dedup_results(list(old_res) + list(res))
            results_ref["res"] = res_full
            await asyncio.to_thread(_write_results_disk, run_id, res_full)
            if _trip.get("tripped"):
                status = "error"
                _state["last_error"] = ("所有评测端点连续失败被熔断, 评测中止: "
                                        + ", ".join(_trip["tripped"])
                                        + "（端点可能已停止/地址失效, 请点「自动获取运行中推理」或检查推理实例）")
            elif run_cancel.is_set():
                status = "cancelled"
            else:
                status = "done"
        except Exception as e:
            _state["last_error"] = f"{type(e).__name__}: {e}"
            import traceback
            _state["last_error"] += "\n" + traceback.format_exc()[-2000:]
            status = "error"
        finally:
            n_done = len(results_ref.get("res") or [])
            _write_meta(run_id, status, n_done)
            _state["results"][run_id] = results_ref.get("res") or []
            # 仅当共享状态仍属于本次运行时才复位: 万一有新运行顶上来, 旧任务不得抹掉它的状态
            # (2026-09-16: 配合「每次运行独立 cancel Event」彻底消除旧任务复活写文件的可能)
            if _state.get("cancel") is run_cancel:
                _state["running"] = False
                _state["per_ep"] = {}
                _state["live_run"] = {}
                run_cancel.clear()
            # 评测结束 -> 解除静默闸门(见 benchmark_guard.py)
            guard_set_active(False)

    # 评测运行期间开启静默闸门(见 benchmark_guard.py): 单机运行是空实现,
    # 在带后台监控/轮询的部署里可借此避免抢带宽, 保证被测端点性能数据干净。
    guard_set_active(True, owner=f"quality run_id={run_id}")
    asyncio.create_task(_bg())
    return {"ok": True, "run_id": run_id, "n": tier_cfg["n"],
            "est_wall_min": round(est_min, 1),
            "resumed": bool(resume_meta), "remaining_rounds": len(tier_cfg["items"]) * req.rounds,
            "total_rounds": total_all}


@router.get("/status")
async def status():
    p = _state["progress"]
    cancelling = bool(_state["cancel"] is not None and _state["cancel"].is_set())
    return {"running": _state["running"], "current": _state["current"],
            "progress": p, "per_ep": dict(_state["per_ep"]),
            "last_error": _state["last_error"], "cancelling": cancelling,
            "done": p.get("done", 0), "total": p.get("total", 0)}


class _PreflightIn(BaseModel):
    bases: list[str]
    # 通用 OpenAI 端点支持: 可只传 bases(旧行为), 也可传完整 endpoints(含 api_key/model)
    endpoints: Optional[list[dict]] = None


class _TestEpItem(BaseModel):
    name: Optional[str] = None
    base_url: str = ""
    model: Optional[str] = None
    api_key: Optional[str] = None
    thinking: Optional[bool] = None


class _TestEpIn(BaseModel):
    endpoints: list[_TestEpItem] = []
    # 评委端点(可选, 与评测里的 subjective_judge 同结构)
    judge: Optional[dict] = None


async def _probe_openai_ep(base_url: str, model: str = "", api_key: str = "",
                           model_hint: bool = True) -> dict:
    """通用 OpenAI 兼容端点探测(与评测同网络路径, 避开浏览器 CORS)。

    两级判定, 避免"没有 /v1/models 的好端点被误判不可达":
      1) GET  {base}/v1/models  —— 能列出模型最好, 顺便校验模型名是否存在
      2) 否则 POST {base}/v1/chat/completions(max_tokens=1) —— 真发一次最小请求
    两级都会自动尝试 base_url 的两种写法(带/不带 /v1; 允许直接粘贴完整 URL),
    公共 API(DeepSeek/OpenAI 等)带 Bearer API Key。
    返回 {ok, url, mode, models, model_found, latency_ms, error, hint}
    """
    import httpx

    base_url = (base_url or "").strip()
    model = (model or "").strip()
    if not base_url:
        return {"ok": False, "url": "", "mode": None, "models": [], "model_found": None,
                "latency_ms": 0, "error": "base_url 为空", "hint": "请填写 OpenAI 兼容服务地址"}
    headers = {"Authorization": "Bearer " + api_key} if api_key else None
    t0 = time.time()
    last_err = ""
    last_status = 0
    net_down = False   # 连接层就失败(地址不通) -> 同一主机的其它写法/阶段不必再试, 立即返回

    # connect=2.5s: 地址不通时快速失败(Windows 上被防火墙丢弃的连接会挂到总超时, 体验很差)
    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=2.5), follow_redirects=True) as c:
        # 1) /v1/models
        for url in qe.openai_url_candidates(base_url, "/models"):
            try:
                r = await c.get(url, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_err = f"{type(e).__name__}: {str(e)[:90]}"
                net_down = True
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:90]}"
                continue
            last_status = r.status_code
            if r.status_code == 200:
                ids = []
                try:
                    ids = [str(m.get("id", "")) for m in (r.json().get("data") or []) if m.get("id")]
                except Exception:
                    ids = []
                found = None
                if model and ids:
                    found = model in ids
                    if not found:  # 容忍 "deepseek-chat" vs "deepseek-chat-0324" 这类前缀差异
                        found = any(i.startswith(model) or model.startswith(i) for i in ids)
                return {"ok": True, "url": url, "mode": "models", "models": ids[:60],
                        "model_found": found, "latency_ms": int((time.time() - t0) * 1000),
                        "error": None, "hint": ""}
            last_err = f"HTTP {r.status_code}"
        # 2) 最小 chat 请求(有些服务不实现 /models)
        if not net_down:
            for url in qe.openai_url_candidates(base_url, "/chat/completions"):
                body = {"model": model or "default", "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1, "temperature": 0, "stream": False}
                try:
                    r = await c.post(url, json=body, headers=headers)
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    last_err = f"{type(e).__name__}: {str(e)[:90]}"
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {str(e)[:90]}"
                    continue
                last_status = r.status_code
                if 200 <= r.status_code < 300:
                    return {"ok": True, "url": url, "mode": "chat", "models": [],
                            "model_found": None, "latency_ms": int((time.time() - t0) * 1000),
                            "error": None,
                            "hint": "" if model else "该服务不提供 /v1/models, 请手动填写模型名"}
                snippet = ""
                try:
                    snippet = r.text[:160]
                except Exception:
                    pass
                last_err = f"HTTP {r.status_code}" + (f": {snippet}" if snippet else "")

    hint = qe._endpoint_hint(last_status, model or "<模型名>", base_url)
    if net_down and not hint:
        hint = " 提示: 地址连不上 —— 请确认服务已启动、端口正确、本机可访问(远程地址注意防火墙)"
    return {"ok": False, "url": (qe.openai_url_candidates(base_url, "/chat/completions") or [""])[0],
            "mode": None, "models": [], "model_found": None,
            "latency_ms": int((time.time() - t0) * 1000),
            "error": last_err or "连接失败", "hint": hint.strip()}


@router.post("/test_endpoint")
async def test_endpoint(req: _TestEpIn):
    """「测试连接」: 逐个探测端点(含评委), 返回是否连通 + 模型列表 + 延迟 + 失败原因。

    走服务端探测(与评测同一网络路径), 因此浏览器 CORS / HTTPS 混合内容都不影响结果;
    DeepSeek 等公共 API 只要在端点行填了 API Key 就能连通。
    """
    items = []
    for e in (req.endpoints or [])[:10]:
        items.append(e.model_dump() if hasattr(e, "model_dump") else dict(e))
    if req.judge and (req.judge.get("base_url") or "").strip():
        items.append({"name": (req.judge.get("name") or "评委端点"),
                      "base_url": req.judge.get("base_url"), "model": req.judge.get("model"),
                      "api_key": req.judge.get("api_key"), "is_judge": True})
    if not items:
        return {"ok": True, "results": []}
    results = await asyncio.gather(*[
        _probe_openai_ep(it.get("base_url") or "", it.get("model") or "", it.get("api_key") or "")
        for it in items])
    out = []
    for it, r in zip(items, results):
        r = dict(r)
        r["name"] = it.get("name") or it.get("model") or it.get("base_url")
        r["base_url"] = it.get("base_url")
        r["model"] = it.get("model") or ""
        r["is_judge"] = bool(it.get("is_judge"))
        out.append(r)
    return {"ok": True, "results": out, "n_ok": sum(1 for r in out if r["ok"]), "n": len(out)}


@router.post("/preflight")
async def preflight(req: _PreflightIn):
    """启动前预检: 由服务端探测各端点(与评测同网络路径, 避开浏览器 CORS)。

    2026-09-17 修复「公共 API 被误判不可达而拦住评测」:
      - 旧实现只探 /v1/models 且**不带 API Key** → DeepSeek 等鉴权服务一律 401,
        全部端点"不可达"时直接阻止开跑;
      - 现在改为两级探测(见 _probe_openai_ep): /v1/models 失败就真发一次最小 chat 请求,
        并携带端点自己的 API Key, base_url 写法也自动容错。
    兼容旧调用: 只传 bases 时按无 Key 探测。
    """
    eps = []
    if req.endpoints:
        for e in req.endpoints:
            eps.append((str(e.get("base_url") or ""), str(e.get("model") or ""),
                        str(e.get("api_key") or "")))
    else:
        eps = [(b, "", "") for b in (req.bases or [])]
    eps = [e for e in eps if e[0].strip()][:10]
    results = await asyncio.gather(*[_probe_openai_ep(b, m, k) for b, m, k in eps])
    simple = [{"ok": r["ok"], "models": r["models"], "error": r["error"],
               "mode": r["mode"], "url": r["url"], "latency_ms": r["latency_ms"],
               "hint": r.get("hint", ""), "model_found": r.get("model_found")} for r in results]
    return {"ok": True, "results": simple}


@router.get("/live")
async def live(since: int = Query(0, ge=0)):
    """实时测试表现: 最近完成的条目(精简) + 进度/每端点 + 实时热力图三件套。
    显示方案的两处关键设计:
    - cells: 已判定格子的紧凑增量表 [qid, ep, round, code, score]
      (code 1 通过/2 失败/3 错误/4 已答未评分/5 部分得分 0<score<1), 带 seq 游标。
      页面中途刷新/晚开也能一次拉全 → 格子数不再「凭空少一截」; score 让 0.1~0.9 直接显示在格子里。
    - running_now: 按开始时间升序(在途任务原本无序 → 每轮轮询选到不同题 = 用户看到的「乱闪」)。
      前端不再按这里的顺序涂蓝, 而是取「该 端点×轮次 泳道里第一个未完成格」涂蓝,
      这样蓝格恒为已完成格子的下一格 → 从左到右顺序推进, 其余在途数用文字 `+N 在途`。
    """
    p = _state["progress"]
    cancelling = bool(_state["cancel"] is not None and _state["cancel"].is_set())
    cells = _state["cells"]
    run_now = [{"qid": k.split("|")[0], "round": int(k.split("|")[1] or 1),
                "ep": "|".join(k.split("|")[2:]), "ts": ts}
               for k, ts in sorted(_state.get("live_run", {}).items(), key=lambda kv: kv[1])]
    return {"ok": True, "run_id": _state["run_id"], "running": _state["running"],
            "cancelling": cancelling, "progress": p, "per_ep": dict(_state["per_ep"]),
            "items": list(_state["live"]),
            "questions": _state.get("items_lite"), "endpoints": _state.get("endpoints_lite") or [],
            "concurrency": _state.get("concurrency") or 0,
            "rounds": _state.get("rounds") or 1,
            "skeleton": ({"run_id": _state["skel"]["run_id"], "n": len(_state["skel"]["questions"]),
                          "endpoints": _state["skel"]["endpoints"], "rounds": _state["skel"]["rounds"],
                          "total_cells": _state["skel"]["total_cells"]}
                         if _state.get("skel") else None),
            "cells": cells[since:] if since < len(cells) else [],
            "seq": len(cells),
            "running_now": run_now}


@router.get("/skeleton")
async def skeleton(run_id: Optional[int] = None):
    """热力图骨架(题目全集 × 端点 × 轮次)。照搬参考页: 格子数量只由骨架决定, 永不由结果决定。
    当前运行走内存, 历史运行读 run_N_skeleton.json; 都没有(修复前的旧 run)时回落到结果文件
    去重集合并标记 partial_skeleton=true(前端据此显示「骨架未知」提示, 不再猜轮次)。"""
    import json as _json
    rid = run_id if run_id is not None else _state["run_id"]
    sk = _state.get("skel")
    if sk and sk["run_id"] == rid and rid != 0:
        return {"ok": True, "run_id": rid, **sk}
    p = DATA_DIR / f"run_{rid}_skeleton.json"
    if p.exists():
        try:
            return {"ok": True, "run_id": rid, **_json.loads(p.read_text(encoding="utf-8"))}
        except Exception:
            pass
    rp = DATA_DIR / f"run_{rid}_results.json"
    if not rp.exists():
        raise HTTPException(404, "该运行没有骨架记录(修复前产生的历史 run)")
    res = _json.loads(rp.read_text(encoding="utf-8"))
    seen: dict = {}
    eps: dict = {}
    rnds: set = set()
    for r in res:
        seen.setdefault(r.get("qid"), {"qid": r.get("qid"), "benchmark": r.get("benchmark", ""),
                                       "domain": r.get("domain", ""),
                                       "difficulty": r.get("difficulty", "medium")})
        if r.get("endpoint"):
            eps[r["endpoint"]] = True
        rnds.add(int(r.get("round") or 1))
    return {"ok": True, "run_id": rid, "questions": list(seen.values()),
            "endpoints": list(eps.keys()), "rounds": max(rnds or {1}),
            "total_cells": len(seen) * len(eps) * max(rnds or {1}),
            "partial_skeleton": True}



@router.get("/results")
async def results(run_id: Optional[int] = None, limit: int = Query(0, ge=0), offset: int = Query(0, ge=0)):
    if run_id is None:
        run_id = _state["run_id"]
        if run_id == 0:
            raise HTTPException(404, "尚未运行过评测")
    path = DATA_DIR / f"run_{run_id}_results.json"
    if not path.exists():
        raise HTTPException(404, "结果不存在")
    res = json.loads(path.read_text(encoding="utf-8"))
    total = len(res)
    status = "done"
    meta_path = DATA_DIR / f"run_{run_id}.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status = meta.get("status", "done")
        except Exception:
            pass
    limit = limit or 0
    offset = offset or 0
    if offset or limit:
        res = res[offset: offset + limit] if limit else res[offset:]
    _rounds = 1
    if meta_path.exists():
        try:
            _rounds = int(json.loads(meta_path.read_text(encoding="utf-8")).get("rounds") or 1)
        except Exception:
            pass
    return {"ok": True, "run_id": run_id, "total": total, "status": status, "rounds": _rounds,
            "partial": status in ("running", "cancelled", "error"), "results": res}


@router.post("/cancel")
async def cancel():
    """停止当前运行: 只置停止标志, 由后台任务自己收尾(落盘部分结果 → status=cancelled)。

    2026-09-16 修复: 旧实现顺带把 running 立即置 False —— 此时后台任务其实还在收尾,
    用户「停止 → 立刻点新测试/续跑」会让两个任务并发写同一份结果文件(旧任务还会被
    clear() 复活)。现在 running 由后台任务的 finally 负责复位, 收尾完成前新运行会被
    409 挡住, 前端 /status 的 cancelling 标志照旧显示「停止中…」。
    """
    ev = _state.get("cancel")
    if ev is not None:
        ev.set()
    return {"ok": True}


@router.get("/history")
async def history():
    runs = []
    for p in sorted(DATA_DIR.glob("run_*_results.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        rid = p.name.split("_")[1]
        meta_path = DATA_DIR / f"run_{rid}.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        runs.append({"run_id": int(rid), "meta": meta,
                     "status": meta.get("status", "unknown"),
                     "n_done": meta.get("n_done", 0)})
    return {"ok": True, "runs": runs}


@router.delete("/history-all")
async def delete_history_all():
    """删除全部历史运行(所有 meta+results 文件)并清空内存结果缓存。
    若某 run 正在运行中, 跳过它(需先停止)。"""
    cur = None
    if _state["running"] and _state.get("current"):
        cur = _state["current"].get("run_id")
    removed, kept = 0, []
    # run_*.json 已包含 run_*_results.json，一次 glob 去重即可（二次 unlink 会 FileNotFoundError）
    for p in sorted(set(DATA_DIR.glob("run_*.json"))):
        parts = p.name.replace(".json", "").split("_")
        try:
            rid = int(parts[1])
        except (IndexError, ValueError):
            continue
        if cur is not None and rid == cur:
            kept.append(rid)
            continue
        try:
            if p.exists():
                p.unlink()
                removed += 1
        except OSError as e:
            raise HTTPException(500, f"删除失败 {p.name}: {e}")
    _state["results"].clear()
    _state["live"].clear()
    _state["live_run"] = {}
    _state["cells"] = []
    _state["cell_seq"] = 0
    return {"ok": True, "removed_files": removed, "kept_running": kept}


@router.delete("/history/{run_id}")
async def delete_history(run_id: int):
    """删除一条历史运行(元数据 + 结果文件); 正在运行/进行中的 run 禁止删除。"""
    if _state["running"] and _state.get("current") and _state["current"].get("run_id") == run_id:
        raise HTTPException(400, "该运行正在进行中, 请先停止再删除")
    removed = []
    for p in (DATA_DIR / f"run_{run_id}.json", DATA_DIR / f"run_{run_id}_results.json",
              DATA_DIR / f"run_{run_id}_skeleton.json"):
        if p.exists():
            try:
                p.unlink()
                removed.append(p.name)
            except OSError as e:
                raise HTTPException(500, f"删除文件失败 {p.name}: {e}")
    if run_id in _state.get("results", {}):
        _state["results"].pop(run_id, None)
    return {"ok": True, "run_id": run_id, "removed": removed}


@router.on_event("startup")
async def _quality_warm():
    asyncio.get_event_loop().run_in_executor(None, _meta_compute)
    # 2026-09-14: SQL 执行判分引擎(内存 SQLite 建库 ~2s)后台预热, 避免第一道 SQL 题现场建库
    if qe.sql_tables_dir():
        asyncio.get_event_loop().run_in_executor(None, qe.warm_sql_exec)
