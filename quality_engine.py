#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理质量测试 - 评测引擎
- 档位生成: 按时间预算从题集抽题 (每档覆盖全部领域, 难度配比易25/中40/难35偏难)
- 执行: asyncio + httpx 并发调 OpenAI 兼容端点
- 评分: exact/float_exact/option/regex/contains/code/bfcl/ifeval/subjective
- 可 CLI 运行, 也可被 FastAPI 服务(见 server.py)调用
"""
import asyncio, csv, json, os, re, ast, time, random, hashlib, sqlite3, threading, traceback
from collections import defaultdict
try:
    import httpx
except ImportError:
    httpx = None

# 可选的调用追踪钩子: 若同目录存在 call_tracker.py 且提供 begin()/end(),
# 每次调用推理端点前后会被记录(便于排查/审计); 没有该文件时自动跳过, 不影响评测。
try:
    import call_tracker as _call_tracker  # type: ignore
except Exception:  # pragma: no cover
    _call_tracker = None

PREFILL_TPS = 1500.0
DECODE_TPS = 35.0
CONCURRENCY = 2          # 每端点并发 (默认低并发: 避免大量同时访问压垮远端)
EFF = 1.3                # 实际/理想裕量

# 请求健壮性 (2026-09-12 加固):
IDLE_TIMEOUT = 90.0      # 流式响应「无新字节」看门狗秒数: 卡住不再生成内容 -> 立即报错并释放并发槽
CONNECT_TIMEOUT = 10.0   # 连接超时
WRITE_TIMEOUT = 30.0     # 发送请求体超时
EP_FAIL_LIMIT = 3        # 某端点连续失败 N 次 -> 熔断跳过该端点 (给它喘息, 不继续堆积请求)

# ============ 档位定义 (墙钟分钟) ============
TIERS = [
    ("T1", 10), ("T2", 30), ("T3", 60), ("T4", 120), ("T5", 180),
    ("T6", 240), ("T7", 300), ("T8", 360), ("T9", 720), ("T10", None),
    ("SQL", None),  # SQL 单独测试: 全部 sql_benchmark 参考题(25), LLM 评委判 SQL 等价性
]
# 难度配比: 偏难 (易25 中40 难35)
DIFF_WEIGHTS = {"easy": 0.25, "medium": 0.40, "hard": 0.35}

# ============ 五档难度 (2026-09-17 按用户要求: 小白 / 简单 / 中等 / 困难 / 极难) ============
# 题库原始 difficulty 只有 easy / medium / hard 三档, 这里在 easy 与 hard 档**内部**再按题面
# 规模(估 prompt+生成 token)各切一刀, 得到 5 档 —— 阈值取全库统计(easy 档中位数 207,
# hard 档 p70 = 657), 规则确定、可复现, 且不会与题库原始标注冲突:
#   小白 = easy 且规模 ≤207 ; 简单 = 其余 easy ; 中等 = medium ;
#   困难 = hard 且规模 ≤657 ; 极难 = 其余 hard(长题面/长输出/多轮工具调用等重活)。
DIFF_TIERS = ["小白", "简单", "中等", "困难", "极难"]
DIFF_RANK = {t: i for i, t in enumerate(DIFF_TIERS)}
DIFF_EASY_SPLIT = 207    # easy 档 (est_prompt_tok+est_gen_tok) 中位数
DIFF_HARD_SPLIT = 657    # hard 档 (est_prompt_tok+est_gen_tok) p70


def diff_tier(q) -> str:
    """五档难度标签(小白/简单/中等/困难/极难)。"""
    d = (q or {}).get("difficulty") or "medium"
    size = ((q or {}).get("est_prompt_tok") or 0) + ((q or {}).get("est_gen_tok") or 0)
    if d == "easy":
        return "小白" if size <= DIFF_EASY_SPLIT else "简单"
    if d == "hard":
        return "极难" if size > DIFF_HARD_SPLIT else "困难"
    return "中等"


def sort_by_difficulty(items):
    """派题/显示顺序 = 难度序(小白→简单→中等→困难→极难), 同级按题号自然序。
    2026-09-17 按用户要求: 抽题仍按 benchmark 轮转分层(保证每个基准都覆盖),
    但**执行顺序**改成从易到难, 于是测试是「越做越难」地推进, 热力图也自然按难度分带。"""
    return sorted(items, key=lambda q: (DIFF_RANK.get(diff_tier(q), 2), _nat_key(q)))

# ============ 加载题集 ============
def load_questions(path):
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            o["est_prompt_tok"] = o.get("est_prompt_tok", est_tok(str(o.get("prompt", ""))))
            o["est_gen_tok"] = o.get("est_gen_tok", 300)
            qs.append(o)
    return qs

def est_tok(s):
    zh = sum(1 for c in s if '\u4e00' <= c <= '\u9fff')
    other = len(s) - zh
    return max(1, int(zh / 1.6 + other / 4.0))

# thinking 模型每 benchmark 典型输出 token (decode 35 tok/s 口径, 与用户档位估算一致)
# 口径: 保守取 thinking 端点中位输出 (reasoning+content)。非 thinking 端点更快, 以 thinking 为瓶颈。
THINK_OUT = {
    "gsm8k": 1000, "mgsm_en": 400, "mgsm_zh": 400, "aime2024": 2000, "aime2025": 1800,
    "humaneval": 600, "mbpp": 700,
    "ifeval": 800,
    "arc_challenge": 400, "mmlu_pro": 500, "supergpqa": 400,
    "drop": 300,
    "bbh_boolean_expressions": 550, "bbh_date_understanding": 550, "bbh_geometric_shapes": 550,
    "bfcl_simple": 200, "bfcl_parallel": 250, "bfcl_multiple": 250,
    "bfcl_parallel_multiple": 250, "bfcl_rest": 250,
    "bfcl_multi_turn_base": 800, "bfcl_multi_turn_composite": 1000,
    "mtbench": 1000, "alpaca_eval": 1000,
    "ib_passkey": 80, "ib_math_find": 200, "ib_number_string": 100,
    "ib_kv_retrieval": 60, "ib_longbook_qa_chn": 300,
}
DEFAULT_THINK_OUT = 500

def q_est_sec(q, thinking=True):
    """单请求单流估算秒 (thinking 口径)"""
    p = q.get("est_prompt_tok", est_tok(str(q.get("prompt", ""))))
    g = q.get("est_gen_tok", 300)
    if thinking:
        g = THINK_OUT.get(q.get("benchmark", ""), DEFAULT_THINK_OUT)
    return max(1.0, p / PREFILL_TPS + g / DECODE_TPS)

# ============ 档位抽题 ============
def _nat_key(q):
    """题号自然排序键: 数字后缀优先(如 sql_benchmark-01..25), 无数字按字符串。
    排序后 -> 执行序 = 骨架序 = 矩阵显示序, 题库从左到右按题号推进,
    进行中的格子始终处在序列中正确位置(照搬参考页 questionsList 顺序)。"""
    qid = q.get("id", "") if isinstance(q, dict) else str(q)
    m = re.search(r"(\d+)$", qid)
    return (int(m.group(1)), qid) if m else (1 << 60, qid)


def build_tier(qs, name, budget_min, per_bm_cap=2000, seed=42, endpoints_n=3, concurrency=None):
    """按时间预算抽题: 分层难度配比偏难(easy25/medium40/hard35), 轮转扫描保证每 benchmark 均衡覆盖。
    budget_min=None => 无限档(取全部题, 受 per_bm_cap 限制)。
    墙钟估计 = 单流时间 / (并发*EFF*端点数), 目标 wall ~ budget_min。
    返回 {tier, items, est_sec, est_wall_min, n, diff_count}
    """
    concurrency = concurrency or CONCURRENCY
    if name == "SQL":
        # SQL 单独测试: 全量 sql_benchmark 题(不抽样, 严格按题号 01..25 排序, 与参考页 questionsList 一致)
        sel = [q for q in qs if q.get("benchmark") == "sql_benchmark"]
        sel.sort(key=_nat_key)
        tot = float(sum(q_est_sec(q) for q in sel))
        return finalize(sel, name, budget_min, tot, concurrency, endpoints_n)
    random.Random(seed).shuffle(qs)
    by_bm = defaultdict(lambda: {"easy": [], "medium": [], "hard": []})
    for q in qs:
        d = q.get("difficulty", "medium")
        if d not in ("easy", "medium", "hard"): d = "medium"
        by_bm[q["benchmark"]][d].append(q)
    names = sorted(by_bm.keys())
    for bm in names:
        for d in by_bm[bm]:
            random.Random(seed + hash(bm) % 10000 + hash(d) % 1000).shuffle(by_bm[bm][d])
    # 预算(单流秒): 墙钟分钟 * 60 * EFF * 并发 (3端点并行各跑一套, 墙钟由单端点决定)
    if budget_min is None:
        budget_sec = float("inf")
    else:
        budget_sec = budget_min * 60 * EFF * concurrency + 1e-9
    # 手工 20 位难度序列: 约 25% easy / 40% medium / 35% hard (偏难)
    diff_seq = ["easy", "easy", "medium", "medium", "medium", "hard", "hard", "hard",
                "medium", "hard", "medium", "medium", "hard", "hard", "hard", "medium",
                "hard", "hard", "hard", "medium"]
    selected = []
    total_sec = 0.0
    counts = {bm: 0 for bm in names}
    total_pool = {bm: sum(len(v) for v in by_bm[bm].values()) for bm in names}
    # 轮转扫描: 每轮每个 benchmark 取 1 题, 难度按 diff_seq 循环
    while True:
        advanced = False
        for bm in names:
            if counts[bm] >= per_bm_cap or counts[bm] >= total_pool[bm]:
                continue
            d = diff_seq[counts[bm] % len(diff_seq)]
            pool = by_bm[bm][d]
            if not pool:
                for alt in ("medium", "hard", "easy"):
                    if by_bm[bm][alt]:
                        d = alt; pool = by_bm[bm][alt]; break
                else:
                    continue
            q = pool.pop(0)
            selected.append(q)
            counts[bm] += 1
            total_sec += q_est_sec(q)
            advanced = True
            if total_sec >= budget_sec:
                return finalize(sorted(selected, key=_nat_key), name, budget_min, total_sec, concurrency, endpoints_n)
        if not advanced:
            break
    # 执行序 = 题号序(从第一道到最后一道), 保证进行中格子在矩阵中按序推进
    return finalize(sorted(selected, key=_nat_key), name, budget_min, total_sec, concurrency, endpoints_n)

def finalize(selected, name, budget_min, total_sec, concurrency=CONCURRENCY, endpoints_n=3):
    diff_count = defaultdict(int)
    diff_count3 = defaultdict(int)
    for q in selected:
        diff_count[diff_tier(q)] += 1
        diff_count3[q.get("difficulty", "medium")] += 1
    return {
        "tier": name,
        "budget_min": budget_min,
        "est_sec": total_sec,
        "est_wall_min": total_sec / (concurrency * EFF) / 60,
        "items": selected,
        "n": len(selected),
        "diff_count": {t: diff_count.get(t, 0) for t in DIFF_TIERS},   # 五档(小白…极难)
        "diff_count_raw": dict(diff_count3),                            # 题库原始三档
    }

# ============ 评分器 ============
def extract_answer_num(s):
    """从输出提取数值答案: 优先 'answer/答案是/result/=' 后跟数字, 否则取最后一个数字"""
    s = str(s)
    pats = [
        r"(?:answer|答案|result|output|equals?|is)\s*(?:is|:|=|是|为|：)?\s*[\$¥]?\s*(-?\d+(?:[.,]\d+)?)",
        r"[\$¥]\s*(-?\d+(?:[.,]\d+)?)",
    ]
    for p in pats:
        ms = re.findall(p, s, re.I)
        if ms:
            return ms[-1].replace(",", "").replace("，", "")
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", s.replace(",", ""))
    return nums[-1] if nums else s

def norm_num(s):
    s = str(s).strip()
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", s.replace(",", ""))
    return m.group(0).replace(",", ".") if m else s

def score_exact(out, ans, **kw):
    return 1.0 if str(out).strip() == str(ans).strip() else 0.0

def score_float_exact(out, ans, **kw):
    o = extract_answer_num(out); a = norm_num(ans)
    try:
        return 1.0 if abs(float(o) - float(a)) < 1e-6 else 0.0
    except Exception:
        return 1.0 if o == a else 0.0

def score_option(out, ans, letters=None, **kw):
    # 提取输出中最后的字母
    s = str(out)
    m = re.findall(r"\b([A-Za-z])\b", s)
    if not m:
        # 找 "answer is X" 或 last letter
        mm = re.findall(r"(?:answer|letter|option|是|选)\s*[:\s]*([A-Za-z])", s, re.I)
        if mm: m = mm
    if not m:
        return 0.0
    cand = m[-1].upper()
    return 1.0 if cand == str(ans).strip().upper() else 0.0

def score_regex(out, ans, pattern=None, **kw):
    return 1.0 if re.search(pattern or str(ans), str(out), re.I | re.S) else 0.0

def score_contains(out, ans, ignore_case=True, min_len=1, **kw):
    o, a = str(out), str(ans)
    if ignore_case: o, a = o.lower(), a.lower()
    if len(a) < min_len: return 0.0
    return 1.0 if a in o else 0.0

def score_code(out, ans, lang="python", tests=None, prompt="", timeout=5, **kw):
    """提取 ```python 代码/函数实现并执行验收测试
    tests: list[str] (mbpp: assert 语句) 或 str (humaneval: check(candidate) 脚本)
    ans 可为 tests 或测试脚本 (若 tests 未提供)
    """
    code = extract_code(str(out), lang)
    if not code: return 0.0
    if tests is None:
        if isinstance(ans, list):
            tests = ans
        elif isinstance(ans, str):
            try:
                tests = ast.literal_eval(ans)
            except Exception:
                tests = ans
    return run_py(code, tests, prompt, timeout)

def extract_code(out, lang="python"):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", out, re.S)
    if m: return m.group(1)
    # 直接输出代码
    return out.strip()

def _first_def_name(code):
    m = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    return m.group(1) if m else None

def run_py(code, tests, prompt="", timeout=5):
    """执行模型输出代码并跑测试。
    tests: list[str] (mbpp: assert 语句引用 candidate) 或 str (humaneval: check(candidate) 测试脚本)
    绑定规则: 从 code 中第一个 def 取函数名, 注入 ns['candidate'] (mbpp 语义) 并执行测试脚本。
    """
    code = code.strip()
    if not code: return 0.0
    PRELOAD = ("from typing import List, Dict, Tuple, Optional, Set, Any, Union, Callable, Iterable\n"
               "import math, re, json, os, sys, itertools, collections, functools, string, heapq, random\n"
               "from collections import defaultdict, Counter, deque\n")
    ns = {"__builtins__": __builtins__}
    try:
        exec(PRELOAD, ns)
    except Exception:
        pass
    fn_name = _first_def_name(code)
    try:
        exec(code, ns)
    except Exception:
        return 0.0
    if fn_name and fn_name in ns:
        ns["candidate"] = ns[fn_name]
    # 从 prompt 提取函数名兜底
    if not fn_name:
        m = re.search(r"```(?:python|py)?\s*\n\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt or "")
        if m and m.group(1) in ns:
            ns["candidate"] = ns[m.group(1)]
    # 若 code 未含 def 定义 (模型只给函数体), 尝试按 prompt 签名包装
    if not fn_name and prompt:
        m = re.search(r"```(?:python|py)?\s*\n(.*?)```", prompt, re.S)
        sig = m.group(1).strip() if m else ""
        pm = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?:->[^:]*)?:", sig)
        if pm:
            fname = pm.group(1)
            body = "\n".join("    " + l if l.strip() else l for l in code.splitlines())
            try:
                exec(f"{sig.splitlines()[0]}\n{body}", ns)
                ns["candidate"] = ns[fname]
            except Exception:
                return 0.0
    if isinstance(tests, str):
        t = tests.strip()
        try:
            exec(t, ns)
        except Exception:
            return 0.0
        # 2026-09-16 修复(humaneval 恒 1.0): humaneval 的 answer 是「只定义 check(candidate)
        # 的测试脚本」—— 旧实现 exec 完就直接 return 1.0, 等于**从未执行断言**,
        # 空实现/抛异常的实现都拿满分(实测 def candidate(*a,**k): raise 也是 1.0)。
        # 正确口径照官方评测: 取出 check, 用模型实现(或首个 def/候选函数)真正调用它。
        chk = ns.get("check")
        if callable(chk):
            cand = ns.get("candidate")
            if cand is None:
                for k, v in ns.items():
                    if k.startswith("__") or k == "check" or not callable(v):
                        continue
                    cand = v
                    break
            if cand is None:
                return 0.0
            try:
                chk(cand)
                return 1.0
            except Exception:
                return 0.0
        # 纯 assert 语句脚本: exec 成功即通过
        return 1.0
    if not isinstance(tests, list): tests = []
    if not tests: return 0.0
    passed = 0
    for t in tests:
        try:
            if isinstance(t, str):
                exec(t, ns)
            else:
                # 形如 (args, expected) 的基础调用检查
                fn = ns.get("candidate") or (list(ns.values())[0] if ns else None)
                if fn is None: raise RuntimeError("no fn")
                got = fn(*t[0]) if isinstance(t[0], tuple) else fn(t[0])
                eq = abs(float(got) - float(t[1])) < 1e-6 if isinstance(got, (int, float)) else got == t[1]
                if not eq: raise AssertionError
            passed += 1
        except Exception:
            pass
    return passed / len(tests) if tests else 0.0

def score_bfcl(out, ans, expected=None, irrelevance=False, **kw):
    """BFCL: 解析 model tool_calls (来自response), 与可能答案比较
    out = assistant message dict 含 tool_calls; expected = ground_truth 列表 [{fn: {arg: [values]}}]

    2026-09-16 修复: 期望列表为空 = 正确答案就是「不调用任何工具」。
    旧实现只在 irrelevance=true 时认这条, 而题集里 50 道 bfcl_rest 是 expected=[] +
    irrelevance=false(ground truth 就是 "[]") —— 于是不调用记 0 分(判反),
    乱调用反而走 match_bfcl(got, []) 恒返回 1.0(空期望循环不执行) → 蒙对。
    """
    try:
        calls = out.get("tool_calls") or []
        exp = expected or []
        if irrelevance or not exp:
            return 1.0 if len(calls) == 0 else 0.0
        if not calls: return 0.0
        got = []
        for c in calls:
            fn = c.get("function") or c      # 兼容嵌套(标准)与扁平(旧)两种 tool_call 形态
            name, args = fn.get("name"), fn.get("arguments", "")
            try: args = json.loads(args) if isinstance(args, str) else args
            except Exception: args = {}
            got.append({name: args})
        # 匹配: 每个 expected 元素需与某个 got 大致相等 (忽略额外字段)
        # 简化: BFS 匹配, 允许 got 顺序不同, 参数值 list 包容
        return match_bfcl(got, exp)
    except Exception:
        return 0.0

def match_bfcl(got, exp):
    used = [False] * len(got)
    for e in exp:
        if not isinstance(e, dict): continue
        found = False
        for i, g in enumerate(got):
            if used[i]: continue
            if not isinstance(g, dict) or not g: continue
            gname, gargs = next(iter(g.items()))
            ename, eargs = next(iter(e.items()))
            if gname != ename: continue
            if args_equal(gargs, eargs):
                used[i] = True; found = True; break
        if not found: return 0.0
    # 非 irrelevance: 所有 exp 被匹配即可 (容忍多余调用? 官方严格等; 宽松: 允许)
    return 1.0

def args_equal(g, e):
    if not isinstance(g, dict) or not isinstance(e, dict): return str(g) == str(e)
    for k, ev in e.items():
        if k not in g: return False
        gv = g[k]
        # values 可能是 list (多值允许) 
        if isinstance(ev, list):
            if not isinstance(gv, list): gv = [gv]
            # 任一匹配
            if not any(values_close(x, y) for x, y in zip(gv, ev)) and set(str(x) for x in gv) != set(str(x) for x in ev):
                return False
        else:
            if not values_close(gv, ev): return False
    return True

def values_close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    sa, sb = str(a).strip().lower(), str(b).strip().lower()
    if sa == sb: return True
    # 路径归一化: /document -> document; ./x -> x; 末尾斜杠去除
    for s_ in (sa, sb):
        s2 = s_.strip().strip("'\"")
        s2 = s2.replace("\\", "/")
        s2 = s2.lstrip("./").rstrip("/")
    if sa == sb: return True
    if sa.lstrip("/").rstrip("/") == sb.lstrip("/").rstrip("/"): return True
    if sa.split("/")[-1] == sb.split("/")[-1] and "/" in (sa + sb) and "/" in sa and "/" in sb: return True
    # 数字字符串 vs 数字
    try:
        if sa.replace(",", "").replace(".", "").isdigit() and sb.replace(",", "").replace(".", "").isdigit():
            return float(sa.replace(",", ".")) == float(sb.replace(",", "."))
    except Exception: pass
    return False

def score_ifeval(out, ans, instruction_ids=None, kwargs=None, key="", **kw):
    """IFEval 简化检查器: 只做可机械判定的约束
    instruction types: response_language, keywords, forbidden_keywords, length_constraints, format
    """
    s = str(out)
    if not instruction_ids: return 0.0
    ok = 0; total = 0
    for iid, kwd in zip(instruction_ids or [], kwargs or []):
        if not isinstance(kwd, dict): kwd = {}
        total += 1
        res = check_ifeval_rule(iid, kwd, s)
        if res: ok += 1
    return (ok / total) if total else 0.0

def check_ifeval_rule(iid, kwd, s):
    iid = iid or ""
    if "response_language" in iid and "language" in kwd:
        lang = (kwd.get("language") or "").lower()
        if lang in ("english", "en"):
            return not re.search(r"[\u4e00-\u9fff]", s)
        if lang in ("chinese", "zh"):
            return bool(re.search(r"[\u4e00-\u9fff]", s))
    if "keywords" in iid and "keywords" in kwd:
        kws = kwd.get("keywords") or []
        r = (kwd.get("relation") or "all")
        if r == "one of":
            return any(k in s for k in kws)
        return all(k in s for k in kws)
    if "forbidden_words" in iid and "forbidden_words" in kwd:
        fw = kwd.get("forbidden_words") or []
        return not any(f in s for f in fw)
    if "length" in iid and "num_words" in kwd:
        rela = kwd.get("relation") or "at least"
        n = len(s.split())
        target = kwd.get("num_words", 0)
        if rela in ("at least", "greater than"): return n >= target
        if rela in ("at most", "less than"): return n <= target
    if "punctuation" in iid and "no_comma" in iid:
        return "," not in s
    return None  # 未实现的规则跳过 (不计入)

def score_subjective(out, ans, category="", **kw):
    """主观题评分由 judge 端点完成 (引擎级 hook); 这里返回 None 表示待评"""
    return None

# ============ 执行器 ============
SCORERS = {
    "exact": score_exact,
    "float_exact": score_float_exact,
    "option": score_option,
    "regex": score_regex,
    "contains": score_contains,
    "code": score_code,
    "bfcl": score_bfcl,
    "ifeval": score_ifeval,
    "subjective": score_subjective,
}

def parse_display_call(s):
    """解析 'cd(folder='document')' -> ('cd', {'folder': 'document'})"""
    m = re.match(r"\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", s.strip())
    if not m: return None
    name = m.group(1).split(".")[-1]
    args = {}
    body = m.group(2).strip()
    if body:
        for part in re.split(r",\s*(?=(?:[^'\"()]|'[^']*'|\"[^\"]*\")*$)", body):
            pm = re.match(r"\s*([A-Za-z_][\w]*)\s*=\s*(.*)\s*$", part)
            if pm:
                val = pm.group(2).strip().strip("'\"")
                args[pm.group(1)] = val
    return name, args

def score_bfcl_mt(out, ans, expected=None, **kw):
    """多轮: 该轮 tool_calls (显示格式 -> 结构) 与期望比对 (超集匹配: 期望 ⊆ 实际)"""
    try:
        calls = out.get("tool_calls") or []
        if not calls:
            return 1.0 if not expected else 0.0
        got = []
        for c in calls:
            fn = c.get("function") or c      # 兼容嵌套(标准)与扁平(旧)两种 tool_call 形态
            name, args = fn.get("name"), fn.get("arguments", "")
            try: args = json.loads(args) if isinstance(args, str) else args
            except Exception: args = {}
            got.append({name.split(".")[-1]: args})
        exp = []
        for cs in (expected or []):
            p = parse_display_call(cs) if isinstance(cs, str) else None
            if p:
                exp.append({p[0]: p[1]})
            elif isinstance(cs, dict):
                exp.append(cs)
        used = [False]*len(got)
        fn_ok = True
        for e in exp:
            en, ea = next(iter(e.items()))
            hit = False
            for i, g in enumerate(got):
                if used[i]: continue
                gn, ga = next(iter(g.items()))
                if gn == en and args_equal(ga, ea):
                    used[i] = True; hit = True; break
            if not hit:
                # 参数不匹配但函数名在 -> 部分分
                fn_hit = any((not used[j]) and next(iter(g.items()))[0] == en for j, g in enumerate(got))
                if not fn_hit:
                    return 0.0
                fn_ok = False
        return 1.0 if fn_ok else 0.5
    except Exception:
        return 0.0
SCORERS["bfcl_mt"] = score_bfcl_mt

# ============ BFCL 多轮文件系统模拟器 ============
def parse_fs(initial_config):
    """initial_config (json 或 python-repr) -> {cwd, tree: dict path->content}"""
    cfg = {}
    if isinstance(initial_config, str) and initial_config:
        try:
            cfg = json.loads(initial_config)
        except Exception:
            try:
                cfg = ast.literal_eval(initial_config)
            except Exception:
                cfg = {}
    elif isinstance(initial_config, dict):
        cfg = initial_config
    fs = cfg.get("GorillaFileSystem") or cfg.get("filesystem") or {}
    tree = {}
    def walk(node, path):
        if isinstance(node, dict):
            if node.get("type") == "file":
                tree[path] = node.get("content", "")
            elif "contents" in node:
                for k, v in (node.get("contents") or {}).items():
                    walk(v, path.rstrip("/") + "/" + str(k))
            else:
                # 节点自身是目录容器 (如 root 直接含 workspace)
                for k, v in node.items():
                    if k in ("type", "contents"): continue
                    walk(v, path.rstrip("/") + "/" + str(k))
    root = fs.get("root") or {}
    walk(root, "/")
    return {"tree": tree, "cwd": "/", "extra": {k: v for k, v in cfg.items() if k != "GorillaFileSystem"}}

def sim_tool(state, name, args):
    """模拟执行 BFCL 多轮工具, 返回结果字符串 (tool message content)"""
    tree = state["tree"]
    cwd = state.get("cwd", "/")
    def resolve(p):
        if not p: return cwd
        s = str(p)
        if s.startswith("/"): return s
        return (cwd.rstrip("/") + "/" + s).replace("//", "/")
    try:
        if name == "cd":
            state["cwd"] = resolve(args.get("folder", args.get("path", "/")))
            return json.dumps({"ok": True, "cwd": state["cwd"]})
        if name == "mkdir":
            d = resolve(args.get("dir_name") or args.get("path") or args.get("dir"))
            if d not in tree: tree[d] = ""
            return json.dumps({"ok": True, "created": d})
        if name == "mv":
            src = resolve(args.get("source"))
            dst = resolve(args.get("destination"))
            content = tree.pop(src, "")
            tree[dst] = content
            return json.dumps({"ok": True, "moved": src, "to": dst})
        if name in ("grep", "find", "ls", "search"):
            fname = args.get("file_name") or args.get("name")
            pat = args.get("pattern", "")
            if name == "grep" and fname:
                s = resolve(fname)
                content = tree.get(s, "")
                lines = [l for l in content.splitlines() if pat.lower() in l.lower()] if pat else content.splitlines()
                return json.dumps({"ok": True, "file": s, "lines": lines[:20]})
            base = cwd.rstrip("/")
            hits = [k for k in tree if k.startswith(base) or (base == "/" and True)]
            return json.dumps({"ok": True, "files": sorted(hits)[:30]})
        if name == "sort":
            fname = resolve(args.get("file_name") or args.get("path"))
            content = tree.get(fname, "")
            tree[fname] = "\n".join(sorted(content.splitlines()))
            return json.dumps({"ok": True, "sorted": fname})
        if name == "diff":
            s1 = resolve(args.get("file_name1"))
            s2 = resolve(args.get("file_name2"))
            c1, c2 = tree.get(s1, ""), tree.get(s2, "")
            return json.dumps({"ok": True, "diff_lines": [l for l in c1.splitlines() if l not in c2.splitlines()][:20]})
        if "tweet" in name or "twitter" in name:
            ct = state["extra"].get("tweet_counter", 0)
            return json.dumps({"ok": True, "tweet_id": ct, "posted": str(args.get("content", ""))[:120]})
        return json.dumps({"ok": True, "result": "simulated"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)[:200]})


async def _aiter_lines_with_idle(resp, idle_timeout: float):
    """迭代 SSE 行, 超过 idle_timeout 无行 -> 抛 asyncio.TimeoutError。

    这是「调用 api 输出卡住不再生成内容(或极慢仍在计算)」的检测核心:
    正常生成时字节流持续到达, 不会触发; vLLM 被 preemption 风暴/队列堵死时
    无新内容, 立即中止该请求, 而不是傻等 600 秒。
    """
    it = resp.aiter_lines()
    while True:
        try:
            line = await asyncio.wait_for(anext(it), idle_timeout)
        except StopAsyncIteration:
            return
        yield line


# ---------------------------------------------------------------------------
# 通用 OpenAI 兼容端点工具
#   目标: 任何 OpenAI 格式的服务都能接入 —— 本地 vLLM / llama.cpp / Ollama /
#   LM Studio, 以及 DeepSeek、OpenAI 等公共 API(带 Bearer API Key), 不要求
#   服务是由哪个工具箱启动的。
#   用户填 base_url 的写法五花八门, 这里统一容错(见 openai_url_candidates)。
# ---------------------------------------------------------------------------
# OpenAI 官方支持的请求字段; 其余字段(如 vLLM 的 chat_template_kwargs)属非标准扩展,
# 公共 API 可能直接 400 拒绝, 因此 400 时用"只保留标准字段"的载荷重试一次。
_OPENAI_STD_KEYS = {
    "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "stream",
    "stream_options", "tools", "tool_choice", "stop", "top_p", "n", "seed",
    "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs", "response_format",
    "user", "parallel_tool_calls", "reasoning_effort",
}
# 公共 API 的 max_tokens 上限普遍较低(如 8192), 超限会被 400 拒绝 → 降级时收敛
_PUBLIC_MAX_TOKENS_CAP = 8192


def openai_base(base_url: str) -> str:
    """把用户填的 base_url 归一成 OpenAI 兼容 API 根(结尾带 /v1)。

    兼容这些常见写法:
      http://127.0.0.1:8000                      -> http://127.0.0.1:8000/v1
      http://127.0.0.1:8000/v1                   -> 原样
      https://api.deepseek.com                   -> https://api.deepseek.com/v1
      https://api.deepseek.com/v1/chat/completions -> https://api.deepseek.com/v1
      http://gw/llm/v1                           -> 原样(自定义前缀不破坏)
    """
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return ""
    low = b.lower()
    for suffix in ("/chat/completions", "/completions", "/models"):
        if low.endswith(suffix):
            b = b[: -len(suffix)].rstrip("/")
            low = b.lower()
            break
    if not b:
        return ""
    if low.endswith("/v1") or "/v1/" in low:
        return b
    return b + "/v1"


def openai_url_candidates(base_url: str, path: str) -> list:
    """返回该 base_url 下 path 的候选 URL(去重、保序)。

    第 1 个是归一后的标准写法; 第 2 个是"原样拼接"(服务没有 /v1 前缀时用)。
    调用方在 404/405(路径不对)时按顺序回退, 于是 base_url 填法不再影响可用性。
    """
    raw = (base_url or "").strip().rstrip("/")
    out = []
    b = openai_base(raw)
    if b:
        out.append(b + path)
    # 用户直接粘贴了某个完整接口路径(models/chat/completions)时, 不要再拼第二种写法
    _full = ("/chat/completions", "/completions", "/models")
    if raw and not raw.lower().endswith(path.lower()) and not raw.lower().endswith(_full):
        cand = raw + path
        if cand not in out:
            out.append(cand)
    return out


def sanitize_openai_payload(payload: dict) -> dict:
    """只保留 OpenAI 标准字段, 并把过大的 max_tokens 收敛到公共 API 常见上限。"""
    q = {k: v for k, v in payload.items() if k in _OPENAI_STD_KEYS}
    if isinstance(q.get("max_tokens"), int) and q["max_tokens"] > _PUBLIC_MAX_TOKENS_CAP:
        q["max_tokens"] = _PUBLIC_MAX_TOKENS_CAP
    return q


def _endpoint_hint(status: int, model: str, base_url: str) -> str:
    """按 HTTP 状态给一句可操作的提示(公共 API 接入最常见的四类错误)。"""
    if status in (401, 403):
        return " 提示: 需要鉴权 —— 请在端点行的「API Key」里填该服务的密钥(DeepSeek/OpenAI 等公共 API 必填)"
    if status == 404:
        return f" 提示: 路径或模型名不对 —— 请确认 base_url(可只填 https://api.deepseek.com)与模型名「{model}」"
    if status == 429:
        return " 提示: 触发限流/欠费 —— 降低并发数或稍后重试"
    if status == 400:
        return " 提示: 请求被服务拒绝 —— 公共 API 可能不支持 tools/思考参数, 可减少题量或关闭「思考」"
    return ""


async def call_endpoint(client, base_url, model, messages, tools=None, max_tokens=1024,
                        temperature=0.0, timeout=600, idle_timeout=IDLE_TIMEOUT, cancel_flag=None,
                        api_key=None, total_timeout=None, payload_extra: dict = None):
    """OpenAI 兼容 chat/completions 流式调用(通用端点, 含三级容错)。

    - 端点只需是 OpenAI 兼容服务: base_url 可带/不带 /v1, 也可直接粘贴完整
      ".../v1/chat/completions"; 需要鉴权的服务(DeepSeek 等公共 API)填 API Key 即可。
    - stream=True: 首字节尽早返回, 同时用 idle_timeout 看门狗检测卡死。
    - cancel_flag(可空): 置位后**立即中止在途请求** —— 取消消费任务并关闭连接
      (client.stream 上下文退出), vLLM 收到客户端断开会中止该请求释放显存槽位,
      而不是继续在后台计算。这是「停止按钮彻底关闭对推理 API 的占用」的关键。
    - 容错重试: 404/405(路径写法不对)自动换另一种 base_url 写法; 400(公共 API 拒绝
      非标准字段或 max_tokens 超限)自动去掉扩展字段并收敛预算重试一次。最多 4 次尝试。
    - 返回 {content, reasoning, tool_calls, finish, latency, usage} 或 {error, latency}。
    """
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "stream": True}
    if tools:
        payload["tools"] = tools
    if payload_extra:
        payload.update(payload_extra)
    attempts = []
    for u in openai_url_candidates(base_url, "/chat/completions"):
        attempts.append((u, payload))
        san = sanitize_openai_payload(payload)
        if san != payload:
            attempts.append((u, san))
    if not attempts:
        return {"error": "base_url 为空", "latency": 0.0}

    last = None
    i = 0
    while i < len(attempts):
        url, pl = attempts[i]
        res = await _call_once(client, url, pl, messages=messages, api_key=api_key,
                               idle_timeout=idle_timeout, cancel_flag=cancel_flag,
                               total_timeout=total_timeout, base_url=base_url, model=model)
        if not res.get("_retry"):
            res.pop("_retry", None)
            res.pop("_status", None)
            return res
        last = res
        status = res.get("_status") or 0
        nxt = None
        for k in range(i + 1, len(attempts)):
            if status in (404, 405) and attempts[k][0] != url:
                nxt = k
                break
            if status == 400 and attempts[k][1] != pl:
                nxt = k
                break
        if nxt is None:
            break
        i = nxt
    last = last or {"error": "请求未发出", "latency": 0.0}
    last.pop("_retry", None)
    status = last.pop("_status", 0) or 0
    last["error"] = (last.get("error") or "") + _endpoint_hint(status, model, base_url)
    return last


async def _call_once(client, url, pl, messages, api_key=None, idle_timeout=IDLE_TIMEOUT,
                     cancel_flag=None, total_timeout=None, base_url="", model=""):
    """单次尝试。返回结果里带 _retry/_status 表示"换个 URL 或换份载荷还能再试"。"""
    payload = pl
    t0 = time.time()
    tok = None
    if _call_tracker is not None:
        tok = _call_tracker.begin(base_url or url, model, messages, kind="chat")
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        async with client.stream(
            "POST", url, json=payload, headers=headers or None,
            timeout=httpx.Timeout(CONNECT_TIMEOUT, read=idle_timeout, write=WRITE_TIMEOUT, pool=CONNECT_TIMEOUT),
        ) as r:
            if r.status_code != 200:
                body = ""
                try:
                    body = (await r.aread()).decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                err = f"HTTP {r.status_code}: {body}"
                if tok is not None:
                    _call_tracker.end(tok, output="", error=err, messages=messages,
                                      extra={"latency": time.time() - t0, "status_code": r.status_code})
                return {"error": err, "latency": time.time() - t0,
                        "_retry": True, "_status": r.status_code}

            async def _consume():
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_calls: dict[int, dict] = {}
                finish = None
                usage: dict = {}
                try:
                    async for line in _aiter_lines_with_idle(r, idle_timeout):
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except Exception:
                            continue
                        ch = (chunk.get("choices") or [{}])[0]
                        delta = ch.get("delta") or {}
                        c = delta.get("content")
                        if c:
                            content_parts.append(c)
                        rc = delta.get("reasoning") or delta.get("reasoning_content")
                        if rc:
                            reasoning_parts.append(rc)
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            slot = tool_calls.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = (slot["name"] or "") + fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                        if chunk.get("usage") and not usage:
                            usage = chunk["usage"]
                except (asyncio.TimeoutError, httpx.ReadTimeout):
                    err = f"idle timeout: {idle_timeout:.0f}s 内无输出 (端点可能在排队/preemption, 已中止释放槽位)"
                    if tok is not None:
                        _call_tracker.end(tok, output="", error=err, messages=messages,
                                          extra={"latency": time.time() - t0})
                    return {"error": err, "latency": time.time() - t0}
                content_out = "".join(content_parts)
                if tok is not None:
                    _call_tracker.end(tok, output=content_out, error=None, messages=messages,
                                      extra={"latency": time.time() - t0, "finish": finish,
                                             "usage": usage,
                                             "n_tool_calls": len(tool_calls)})
                return {
                    "content": content_out,
                    "reasoning": "".join(reasoning_parts),
                    # 2026-09-16 修复(BFCL 790 题恒 0 分): 旧实现返回扁平结构
                    # {"id","name","arguments"}, 而 BFCL 评分器(run_bfcl_mt / score_bfcl /
                    # score_bfcl_mt)一律按 OpenAI 标准嵌套结构读 c["function"]["name"] ——
                    # 取不到函数名 → 参数为空 → 与实际期望永远不匹配, 单轮/多轮 BFCL 全部判 0。
                    # 统一输出标准嵌套结构(前端 inspector 显示也更规范)。
                    "tool_calls": [{"id": tool_calls[k].get("id") or f"call_{k}",
                                    "type": "function",
                                    "function": {"name": tool_calls[k].get("name") or "",
                                                 "arguments": tool_calls[k].get("arguments") or "{}"}}
                                   for k in sorted(tool_calls)],
                    "finish": finish,
                    "latency": time.time() - t0,
                    "usage": usage,
                }

            task = asyncio.create_task(_consume())
            waiters = {task}
            ctask = None
            if cancel_flag is not None:
                ctask = asyncio.create_task(cancel_flag.wait())
                waiters.add(ctask)
            ttask = None
            if total_timeout is not None:
                ttask = asyncio.create_task(asyncio.sleep(total_timeout))
                waiters.add(ttask)
            done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                for w in pending:
                    w.cancel()
                for w in pending:
                    try:
                        await w
                    except (asyncio.CancelledError, Exception):
                        pass
                return task.result()
            # 用户点了停止 或 单题总超时: 取消消费任务 -> for 循环/生成器被 CancelledError 打断,
            # async with 退出 -> HTTP 连接关闭 -> vLLM 中止该请求。
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            if ttask is not None and ttask in done and not (ctask and ctask in done):
                err = f"total timeout: {total_timeout:.0f}s 未完成 (已中止, 计入超时)"
            else:
                err = "cancelled"
            if tok is not None:
                _call_tracker.end(tok, output="", error=err, messages=messages,
                                  extra={"latency": time.time() - t0})
            return {"error": err, "latency": time.time() - t0}
    except Exception as e:
        err = str(e)[:300]
        if tok is not None:
            _call_tracker.end(tok, output="", error=err, messages=messages,
                              extra={"latency": time.time() - t0})
        return {"error": err, "latency": time.time() - t0}

def mt_tools_from_expected(expected):
    """从期望调用列表反推工具 schema: {'fn': {'arg': [type hints]}} -> function schema"""
    props = {}
    descs = {}
    for turn in expected or []:
        for cs in turn or []:
            p = parse_display_call(cs) if isinstance(cs, str) else None
            if not p: continue
            name, args = p
            descs.setdefault(name, [])
            for k, v in (args or {}).items():
                t = "number" if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit() else "string"
                props.setdefault(name, {})[k] = {"type": t, "description": f"arg {k}"}
    tools = []
    for name, ps in props.items():
        tools.append({"type": "function", "function": {
            "name": name, "description": f"Call {name}",
            "parameters": {"type": "object", "properties": ps, "required": list(ps.keys())}}})
    return tools

async def run_bfcl_mt(client, ep, q, concurrency_ep, cancel_flag=None):
    """多轮 BFCL: turns=[user...]; 每 user turn 允许模型连续多次工具调用 (模拟执行结果),
    直到模型自然停止, 再将该 turn 全部 calls 与期望比对评分。"""
    if cancel_flag and cancel_flag.is_set():
        return {"score": 0.0, "error": "cancelled", "tool_calls": [], "output": "", "latency": 0}
    try:
        p = json.loads(q["prompt"])
        turns = p.get("turns", [])
    except Exception:
        return {"score": 0.0, "error": "bad prompt", "tool_calls": [], "output": "", "latency": 0}
    expected = q["judge"].get("params", {}).get("expected") or q.get("answer") or []
    # 用期望反推工具 schema (保证工具集与期望一致, 评分公平)
    tools = mt_tools_from_expected(expected) or (p.get("tools") or [])
    fs_state = parse_fs(p.get("initial_config", ""))
    messages = []
    total_lat = 0.0
    all_calls = []
    round_scores = []
    for i, turn_msg in enumerate(turns):
        if cancel_flag and cancel_flag.is_set():
            return {"score": (sum(round_scores) / len(round_scores)) if round_scores else 0.0,
                    "error": "cancelled", "tool_calls": all_calls,
                    "output": json.dumps(round_scores), "latency": total_lat}
        messages.append({"role": "user", "content": turn_msg})
        turn_calls = []
        for _step in range(10):  # 单 turn 最多 10 次连续调用
            if cancel_flag and cancel_flag.is_set():
                return {"score": (sum(round_scores) / len(round_scores)) if round_scores else 0.0,
                        "error": "cancelled", "tool_calls": all_calls,
                        "output": json.dumps(round_scores), "latency": total_lat}
            mt = q.get("max_tokens", 2048)
            if ep.get("thinking"):
                mt = max(mt, 16384)
            resp = await call_endpoint(client, ep["base_url"], ep["model"], messages, tools=tools,
                                       max_tokens=mt, cancel_flag=cancel_flag,
                                       api_key=ep.get("api_key"))
            total_lat += resp.get("latency", 0)
            if resp.get("error"):
                ratio = (sum(round_scores) + 0.0) / max(1, len(round_scores) + 1)
                return {"score": ratio, "error": resp["error"], "tool_calls": all_calls,
                        "output": "", "latency": total_lat}
            calls = resp.get("tool_calls") or []
            if not calls:
                break  # 模型停止 -> 该 turn 完成
            turn_calls.extend(calls)
            all_calls.append(calls)
            for c in calls:
                tc_id = c.get("id") or f"call_{i}_{_step}_{len(messages)}"
                fn = c.get("function") or c   # 兼容嵌套(标准)与扁平(旧)两种 tool_call 形态
                fname = (fn.get("name") or "").split(".")[-1]
                try:
                    fargs = json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else (fn.get("arguments") or {})
                except Exception:
                    fargs = {}
                result = sim_tool(fs_state, fname, fargs if isinstance(fargs, dict) else {})
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
        row_exp = expected[i] if i < len(expected) else []
        score = score_bfcl_mt({"tool_calls": turn_calls}, "", expected=row_exp)
        round_scores.append(score)
        if score == 0.0:
            ratio = sum(round_scores) / len(round_scores)
            return {"score": ratio, "error": None, "tool_calls": all_calls,
                    "output": json.dumps(round_scores), "latency": total_lat, "stopped_round": i}
    return {"score": 1.0 if round_scores and all(s == 1.0 for s in round_scores) else (sum(round_scores)/len(round_scores) if round_scores else 0.0),
            "error": None, "tool_calls": all_calls,
            "output": "", "latency": total_lat, "round_scores": round_scores}

_JUDGE_NO_THINK = {"chat_template_kwargs": {"thinking": False}}


def _parse_judge_score(resp: dict):
    """评委响应解析: content 优先取第一个 1-10 整数;
    无 content 时(评委模型 thinking 把预算吃光)只认 reasoning 里的明确评分措辞
    (score/should be/X out of 10/X/10 等, 取最后一次出现), 避免把截断推理中的随机数字当分数。"""
    txt = str(resp.get("content") or "").strip()
    if txt:
        m = re.search(r"\b(10|[1-9])\b", txt)
        return int(m.group(1)) / 10.0 if m else None
    rs = str(resp.get("reasoning") or "")
    if not rs:
        return None
    pats = [r"(\d{1,2})\s*out of\s*10", r"(\d{1,2})\s*/\s*10\b",
            r"\b(?:score|rating|final(?: score)?|rate|evaluation)\s*(?:would be|is|:|=|of)?\s*(\d{1,2})\b",
            r"\b(?:should be|would be|give|scored?|scoring)\s*(?:it\s+)?(\d{1,2})\b",
            r"\b(?:final answer|answer)\s*(?:is|:)?\s*(\d{1,2})\b"]
    best = None
    for p in pats:
        for m in re.finditer(p, rs, re.I):
            v = int(m.group(1))
            if 1 <= v <= 10 and (best is None or m.start() >= best[0]):
                best = (m.start(), v)
    return best[1] / 10.0 if best else None

async def judge_subjective(client, judge_ep, q, output, max_tokens=1024, cancel_flag=None):
    """LLM-judge: 给回答打 1-10 分 (rubric 基础版)"""
    prompt = (f"Rate the following assistant response to the user request on a scale 1-10 "
              f"(10=perfect). Response QUALITY (correctness, helpfulness, relevance, format). "
              f"Output ONLY an integer.\n\n---\nUser: {q['prompt'][:3000]}\n\n"
              f"Assistant: {str(output)[:4000]}")
    resp = await call_endpoint(client, judge_ep["base_url"], judge_ep["model"],
                               [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                               cancel_flag=cancel_flag, payload_extra=_JUDGE_NO_THINK,
                               api_key=judge_ep.get("api_key"))
    if resp.get("error") and "cancelled" not in str(resp.get("error")):
        # 非 vLLM 评委端点可能拒绝 chat_template_kwargs -> 去掉扩展参数重试一次
        resp = await call_endpoint(client, judge_ep["base_url"], judge_ep["model"],
                                   [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                                   cancel_flag=cancel_flag, api_key=judge_ep.get("api_key"))
    if resp.get("error"): return None
    return _parse_judge_score(resp)

async def judge_sql(client, judge_ep, q, output, max_tokens=1024, cancel_flag=None):
    """SQL 等价性评委: 模型 SQL vs 参考 SQL + 预期结果特征 (row_count/columns/first_row)。"""
    try:
        ref = json.loads(q.get("answer") or "{}")
    except Exception:
        ref = {}
    prompt = ("You are a strict SQL judge (DuckDB dialect). A model was asked to write ONE SQL query for the question below.\n\n"
              f"=== QUESTION + SCHEMA ===\n{q['prompt'][:3500]}\n\n"
              f"=== REFERENCE SQL (known correct) ===\n{ref.get('reference_sql','(none)')[:4000]}\n"
              f"EXPECTED: row_count={ref.get('row_count')}; columns={ref.get('columns')}; first_row={json.dumps(ref.get('first_row'), ensure_ascii=False)[:800]}\n\n"
              f"=== MODEL ANSWER ===\n{str(output)[:4000]}\n\n"
              "Judge SEMANTIC equivalence: same tables/joins/filters/aggregations/grouping/ordering/limit and output columns as the reference. "
              "Differences in formatting, casing, whitespace, aliases, or column ordering are acceptable; ROUND-level numeric noise is acceptable. "
              "If the model answer contains no usable SQL query, or queries different semantics, it fails.\n"
              "Output ONLY an integer 1-10 (10=semantically equivalent and would return the expected result, "
              "7-9=minor issues, 4-6=partially correct, 1-3=wrong or no SQL). Output ONLY the integer.")
    resp = await call_endpoint(client, judge_ep["base_url"], judge_ep["model"],
                               [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                               cancel_flag=cancel_flag, payload_extra=_JUDGE_NO_THINK,
                               api_key=judge_ep.get("api_key"))
    if resp.get("error") and "cancelled" not in str(resp.get("error")):
        # 非 vLLM 评委端点可能拒绝 chat_template_kwargs -> 去掉扩展参数重试一次
        resp = await call_endpoint(client, judge_ep["base_url"], judge_ep["model"],
                                   [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                                   cancel_flag=cancel_flag, api_key=judge_ep.get("api_key"))
    if resp.get("error"): return None
    return _parse_judge_score(resp)

# ==================== SQL 确定性执行判分 ====================
# 做法: 用 SQLite **真实执行**模型生成的 SQL，再按四项比对
# (row_count / 列数 / 列名 / first_row 容差) 判定，
# **全程不使用 LLM 评委**，终态只有 pass/fail/error。7 张 AdventureWorks CSV
# (放在 tables/ 目录) → 首次用到时在内存 SQLite 建库一次 → 每条模型 SQL 真跑比对。
# 已核对(09-14): 25 道参考 SQL 在 SQLite 上原样执行, row_count/columns/first_row 与题面 100% 一致;
# 与人工标注对拍 21/25 一致, 4 条差异全部是主观评委放水
# (模型 SQL 实际返回 0 行/聚合口径错) → 执行判分比评委更准。
# 已知方言差: 模型若写 DuckDB 专有语法(QUALIFY / x::int / IS NOT DISTINCT FROM 等)在 SQLite 报错
# → 记 0 分并给 dialect_note(不抛异常、不把格子标成「错误」——请求本身是成功的)。
# 启用 LLM 评委时: 评委降级为「第二意见」——仅当执行判 0 分且执行本身没报错时追加跑一次,
# 结果写进 sql_check.judge_llm 供人工复核, 不参与格子颜色判定。
SQL_TABLE_NAMES = ["Product", "Sales", "Date", "Customer", "Reseller", "Sales_Order", "Sales_Territory"]
SQL_EXEC_PROGRESS_OPS = 100_000      # 进度回调采样步数(执行中按此频率检查墙钟超时)
SQL_EXEC_TIMEOUT_S = 12.0            # 单条模型 SQL 执行墙钟上限(防 CROSS JOIN 爆炸; 实测正常查询 <0.4s)

_sql_state: dict = {"con": None, "lock": threading.Lock(), "build_s": None, "dir": None, "fail": ""}


def sql_tables_dir():
    """7 张 CSV 全在的目录; 找不到返回 None(执行判分不可用, 自动回退评委/未评分)。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("QTEST_SQL_TABLES"), os.environ.get("QUALITY_SQL_TABLES"),
             os.path.join(here, "tables"), os.path.join(here, "data", "sql_tables")]
    for c in cands:
        if c and all(os.path.exists(os.path.join(c, t + ".csv")) for t in SQL_TABLE_NAMES):
            return os.path.abspath(c)
    return None


def _sql_build():
    """在内存 SQLite 建参考库(一次性, ~2s)。返回连接或 None。调用方必须持锁。"""
    if _sql_state["con"] is not None:
        return _sql_state["con"]
    d = sql_tables_dir()
    if not d:
        _sql_state["fail"] = "未找到 SQL 参考数据表(7 张 CSV, 见 tables/ 目录或 QTEST_SQL_TABLES)"
        return None
    t0 = time.time()
    try:
        con = sqlite3.connect(":memory:", check_same_thread=False)
        for t in SQL_TABLE_NAMES:
            p = os.path.join(d, t + ".csv")
            with open(p, newline="", encoding="utf-8-sig") as f:
                rd = csv.reader(f)
                cols = [h.strip() for h in next(rd)]
                con.execute('CREATE TABLE "%s" (%s)' % (t, ",".join('"%s" NUMERIC' % c for c in cols)))
                ins = 'INSERT INTO "%s" VALUES (%s)' % (t, ",".join("?" * len(cols)))
                batch = []
                for row in rd:
                    if not row:
                        continue
                    row = [c.strip() for c in row]
                    if len(row) < len(cols):
                        row += [""] * (len(cols) - len(row))
                    batch.append(tuple(row[:len(cols)]))
                    if len(batch) >= 20000:
                        con.executemany(ins, batch)
                        batch = []
                if batch:
                    con.executemany(ins, batch)
            con.commit()
        con.execute("PRAGMA query_only=ON")     # 建完上只读护栏(多语句 sqlite 天然拒绝)
        _sql_state["con"] = con
        _sql_state["dir"] = d
        _sql_state["build_s"] = round(time.time() - t0, 2)
        _sql_state["fail"] = ""
        return con
    except Exception as e:
        _sql_state["fail"] = f"建库失败 {type(e).__name__}: {str(e)[:120]}"
        return None


def warm_sql_exec():
    """供后端 startup 在线程里预热, 免得第一道 SQL 题付 2s 建库。"""
    with _sql_state["lock"]:
        return _sql_build() is not None


def sql_exec_status():
    """给 /meta 与前端提示用: 执行判分是否可用。"""
    return {"ready": _sql_state.get("con") is not None, "dir": _sql_state.get("dir"),
            "build_s": _sql_state.get("build_s"), "note": _sql_state.get("fail") or ""}


def _sql_clean(s):
    """照抄参考页列名归一化: 小写 + 去掉所有非字母数字。"""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _sql_close(exp, act):
    """照抄参考页 first_row 比对: 数值按期望小数位容差 5*10^-(d+1)+eps, 否则字符串全等。"""
    try:
        ea, eb = float(exp), float(act)
    except (TypeError, ValueError):
        return str(act).strip() == str(exp).strip()
    s = repr(float(exp)) if isinstance(exp, (int, float)) else str(exp)
    d = 0 if "." not in s else len(s.split(".", 1)[1])
    tol = 5 * (10 ** -(d + 1)) + 2.3e-16 * max(abs(ea), abs(eb), 1.0)
    return abs(ea - eb) <= tol


_SQL_FENCE_RE = re.compile(r"```(?:sql|mysql|duckdb|postgres)?\s*(.*?)```", re.S | re.I)
_SQL_START_ONLY_RE = re.compile(r"(?i)^[-—#/*\s]*(with|select)\b")
_CTE_START_RE = re.compile(r"(?i)^[-—#/*\s]*with\b\s+[A-Za-z_][A-Za-z0-9_]*\s+as\b")


def extract_sql(text):
    """从模型输出抽单条 SQL(去围栏/前导解释/尾随说明)。无 SQL 返回 None。

    修正 2026-09-15: 旧实现「任意位置首个 with/select」会把思考过程里的散文
    (如 "with space, ...Need quote identifiers.") 当成 SQL 正文, 整段判给
    SQLite 执行全部语法失败 -> SQL 题全 0 全红。
    新优先级:
      1) 最后一个 ```sql 围栏(最终答案一般以代码块收尾);
      2) 开头直接是 select, 或开头是标准 CTE 头 `WITH <名> AS (` —— 全文即 SQL;
         (只以 "with" 开头的散文不予采信, 回落下文规则)
      3) 倒序取最后一个「行首开头的 with/select」(最终答案 SQL 通常独占一行,
         前面的 with/select 多为推理散文/示例);
      4) 兜底: 全文最后一次 with/select 词命中。
    """
    if not text:
        return None
    fences = _SQL_FENCE_RE.findall(text)
    cand = (fences[-1] if fences else text).strip()
    m0 = _SQL_START_ONLY_RE.match(cand)
    if m0 and (m0.group(1).lower() == "select" or _CTE_START_RE.match(cand)):
        i = cand.rfind(";")
        return (cand[:i + 1] if i != -1 else cand).strip()
    ms = list(re.finditer(r"(?i)\b(with|select)\b", cand))
    if not ms:
        return None
    # 行首命中优先: 关键词之前到行首只有空白/注释符(独占一行的 SQL 起点)
    line_hits = []
    for m in ms:
        nl = cand.rfind("\n", 0, m.start())
        pre = cand[nl + 1:m.start()]
        if re.match(r"[-—#/*\s]*$", pre):
            line_hits.append(m)
    chosen = line_hits[-1] if line_hits else ms[-1]
    i = cand.rfind(";", chosen.start())
    out = (cand[chosen.start():i + 1] if i != -1 else cand[chosen.start():]).strip()
    return out or None


def judge_sql_exec(q, model_output):
    """真实执行模型 SQL 并按参考页四项口径判 1.0/0.0。
    返回 None = 执行判分不可用(表缺失/题目无参考结果), 调用方自行回退。
    返回 dict: {score, method, sql, checks{4}, actual{...}, expected{...}, error?, dialect_note?}"""
    try:
        ref = json.loads(q.get("answer") or "{}")
    except Exception:
        ref = {}
    if not ref.get("reference_sql") or not ref.get("columns"):
        return None
    with _sql_state["lock"]:
        con = _sql_build()
    if con is None:
        return None
    out = {"score": 0.0, "method": "sqlite_exec", "sql": None,
           "checks": {"row_count_match": False, "column_count_match": False,
                      "column_names_match": False, "first_row_match": False},
           "actual": {}, "expected": {"row_count": ref.get("row_count"),
                                      "columns": ref.get("columns"),
                                      "first_row": ref.get("first_row")}}
    sql = extract_sql(model_output)
    out["sql"] = sql
    if not sql:
        out["error"] = "模型输出里没有可执行的 SQL"
        return out
    if not re.match(r"(?is)^\s*(with|select)\b", sql):
        out["error"] = "仅允许 SELECT/WITH 查询(拒绝写操作/多语句)"
        return out
    t0 = time.time()
    deadline = t0 + SQL_EXEC_TIMEOUT_S
    try:
        with _sql_state["lock"]:
            try:
                con.set_progress_handler(lambda d=deadline: 1 if time.time() > d else 0,
                                         SQL_EXEC_PROGRESS_OPS)
                cur = con.execute(sql)
                cols = [d[0] for d in (cur.description or [])]
                rows = cur.fetchall()
            finally:
                try:
                    con.set_progress_handler(None, 0)
                except Exception:
                    pass
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        out["error"] = "执行失败: " + msg
        if isinstance(e, sqlite3.OperationalError) and re.search(
                r"syntax|unrecognized|no such (?:table|column|function)|incomplete|misuse", msg, re.I):
            out["dialect_note"] = ("本地比对引擎(SQLite)不支持该写法(题面参考实现用 DuckDB)。"
                                   "若确认在 DuckDB 下可执行, 请人工复核这一格。")
        return out
    out["actual"]["elapsed"] = round(time.time() - t0, 3)
    out["actual"]["row_count"] = len(rows)
    out["actual"]["columns"] = cols[:24]
    if rows:
        out["actual"]["first_row"] = {c: (str(v) if v is not None else None) for c, v in zip(cols, rows[0])}
    # ---- 以下四项比对 = 照抄参考页 verifyResults / isCorrect ----
    exp_cols = list(ref.get("columns") or [])
    actual_map = {_sql_clean(c): c for c in cols}
    exp_map = {_sql_clean(c): c for c in exp_cols}
    missing = [c for c in exp_cols if _sql_clean(c) not in actual_map]
    extra = [c for c in cols if _sql_clean(c) not in exp_map]
    ch = out["checks"]
    ch["row_count_match"] = len(rows) == ref.get("row_count")
    ch["column_count_match"] = len(cols) == len(exp_cols)
    ch["column_names_match"] = (not missing) and (not extra)
    out["actual"]["missing_columns"] = missing[:12]
    out["actual"]["extra_columns"] = extra[:12]
    diffs = []
    if rows:
        r0 = {c: v for c, v in zip(cols, rows[0])}
        for c in exp_cols:
            if c not in (ref.get("first_row") or {}):
                continue
            ev = ref["first_row"][c]
            ak = actual_map.get(_sql_clean(c))
            if ak is None:
                continue
            if not _sql_close(ev, r0.get(ak)):
                diffs.append({"column": c, "expected": ev,
                              "actual": (str(r0.get(ak)) if r0.get(ak) is not None else None)})
    ch["first_row_match"] = (not diffs) and len(rows) > 0
    out["actual"]["first_row_diffs"] = diffs[:12]
    out["score"] = 1.0 if all(ch.values()) else 0.0
    return out


async def judge_sql_exec_async(q, model_output):
    """同步执行判分放线程里跑(绝不独占事件循环, 见 09-13 /meta 卡 7s 教训)。
    返回 (score, check)。score=None 表示执行判分不可用, 调用方决定回退。"""
    try:
        chk = await asyncio.to_thread(judge_sql_exec, q, model_output)
    except Exception as e:
        return None, {"method": "sqlite_exec", "error": f"{type(e).__name__}: {str(e)[:160]}"}
    if chk is None:
        return None, None
    return chk.get("score"), chk


async def run_tier(qs, tier_cfg, endpoints, concurrency=CONCURRENCY, judge_endpoint=None,
                   on_progress=None, cancel_flag=None, results_cb=None, on_per_ep=None,
                   rounds=1, max_tokens=None, temperature=0.1, timeout=300, retries=2,
                   on_start=None, on_ep_done=None, trip_info=None):
    """执行档位: endpoints=[{name, base_url, model}], 返回结果列表
    每题对每个端点各执行一次; subjective 题由 judge_endpoint 评分 (可为 None=跳过主观题)
    on_per_ep(name, done, total): 每端点完成进度回调（并发汇报，勿阻塞）
    on_start(qid, round, ep_name): 单端点开始执行回调（实时热力图「进行中」标记，勿阻塞）
    on_ep_done(qid, round, ep_name): 单端点完成回调（移出「进行中」标记，勿阻塞）

    2026-09-12 加固:
    - 有界工作池: 不再一次性创建所有题的协程, 只保留 concurrency×N 个常驻 worker
      从队列取题, 内存/任务数恒定, 同时访问严格受每端点并发信号量约束。
    - 端点熔断: 某端点连续失败 >= EP_FAIL_LIMIT -> 标记熔断, 本轮不再向它发请求,
      避免「推理被卡住后仍在不断堆积新的访问」。
    - 取消传播: cancel_flag 置位后 worker 快速退出 (含 bfcl_mt 内部轮次)。
    """
    items = tier_cfg["items"]
    n_workers = max(1, min(concurrency * max(1, len(endpoints)), (len(items) or 1) * max(1, rounds)))
    client = httpx.AsyncClient()
    results = []
    done = 0
    total_n = len(items) * max(1, rounds)
    lock = asyncio.Lock()
    epsems = {ep["name"]: asyncio.Semaphore(concurrency) for ep in endpoints}
    ep_state = {ep["name"]: {"fail": 0, "tripped": False} for ep in endpoints}
    ep_done = {ep["name"]: 0 for ep in endpoints}
    queue: asyncio.Queue = asyncio.Queue()
    for rnd in range(1, max(1, rounds) + 1):
        for q in items:
            queue.put_nowait((q, rnd))

    async def one_ep(q, ep, rnd=1):
        """对单端点执行一题(第 rnd 轮), 带熔断与错误计数。返回结果 dict。"""
        st = ep_state[ep["name"]]
        if st["tripped"]:
            return {"qid": q["id"], "benchmark": q["benchmark"], "difficulty": q["difficulty"],
                    "diff_tier": diff_tier(q),   # 五档难度(小白/简单/中等/困难/极难)
                    "domain": q["domain"], "endpoint": ep["name"], "score": None,
                    "output": "", "reasoning": "", "tool_calls": "",
                    "latency": 0, "error": f"endpoint_quarantined (连续{EP_FAIL_LIMIT}次失败后熔断)",
                    "answer": q.get("answer"), "judge": q["judge"],
                    "max_tokens": q.get("max_tokens", 2048), "round": rnd}
        async with epsems[ep["name"]]:
            if cancel_flag and cancel_flag.is_set():
                return {"qid": q["id"], "benchmark": q["benchmark"], "difficulty": q["difficulty"],
                        "diff_tier": diff_tier(q),   # 五档难度(小白/简单/中等/困难/极难)
                        "domain": q["domain"], "endpoint": ep["name"], "score": None,
                        "output": "", "reasoning": "", "tool_calls": "",
                        "latency": 0, "error": "cancelled", "answer": q.get("answer"),
                        "judge": q["judge"], "max_tokens": q.get("max_tokens", 2048), "round": rnd}
            # 蓝格=信号量已到手、请求真正发出(排队等待不涂蓝); finally 保证摘除。
            if on_start:
                try:
                    on_start(q["id"], rnd, ep["name"])
                except Exception:
                    pass
            try:
                jtype = q["judge"]["type"]
                if jtype == "bfcl_mt":
                    r = await run_bfcl_mt(client, ep, q, concurrency, cancel_flag=cancel_flag)
                    err = r.get("error")
                    async with lock:
                        if err and err != "cancelled":
                            st["fail"] += 1
                            if st["fail"] >= EP_FAIL_LIMIT:
                                st["tripped"] = True
                        elif not err:
                            st["fail"] = 0
                        if all(s["tripped"] for s in ep_state.values()):
                            cancel_flag and cancel_flag.set()
                        ep_done[ep["name"]] += 1
                        if on_per_ep:
                            on_per_ep(ep["name"], ep_done[ep["name"]], total_n)
                    return {"qid": q["id"], "benchmark": q["benchmark"], "difficulty": q["difficulty"],
                            "diff_tier": diff_tier(q),   # 五档难度(小白/简单/中等/困难/极难)
                            "domain": q["domain"], "endpoint": ep["name"], "score": r["score"],
                            "output": r["output"][:4000], "reasoning": "", "tool_calls": json.dumps(r["tool_calls"], ensure_ascii=False)[:3000],
                            "latency": r["latency"], "error": err, "answer": q.get("answer"), "judge": q["judge"],
                            "round": rnd}
                try:
                    p = json.loads(q["prompt"])
                    msgs, tools = p["messages"], p.get("tools", [])
                    use_tools = True
                except Exception:
                    msgs, tools = [{"role": "user", "content": q["prompt"]}], []
                    use_tools = False
                valid_tools = tools if (jtype == "bfcl" and use_tools and tools) else None
                # 2026-09-14: max_tokens 原样传递, 不设上下限 —— 运行参数优先,
                # 用户选多大就用多大(不再强制 thinking 端点 ≥16384, 那会让单题生成拖到几分钟级)
                mt = max_tokens or q.get("max_tokens") or 2048
                # 单题失败重试(最多 retries 次); cancelled/成功不再重试
                err = None
                resp = None
                for attempt in range(max(0, retries) + 1):
                    if cancel_flag and cancel_flag.is_set():
                        break
                    resp = await call_endpoint(client, ep["base_url"], ep["model"], msgs,
                                               tools=valid_tools, max_tokens=mt, cancel_flag=cancel_flag,
                                               temperature=temperature, api_key=ep.get("api_key"),
                                               total_timeout=timeout)
                    err = resp.get("error")
                    if not err or err == "cancelled":
                        break
                    if attempt < retries:
                        await asyncio.sleep(min(1.0 * (attempt + 1), 5.0))
                if resp is None:
                    resp = {"error": "cancelled", "latency": 0.0}
                    err = "cancelled"
                # 2026-09-15: SQL 题 thinking 端点实测 ~50% 概率把 max_tokens 预算全吃在
                #   思考链上 -> content 为空; 且即使没吃光, budget 也在 SQL 写到一半时耗尽
                #   (finish=length), content 是截断的半截 SQL, 同样 UX 就是红格 + 执行失败。
                #   旧逻辑拿推理散文/半截 SQL 判分, SQLite 全语法错误 -> 桌面版 SQL 全红。
                #   => 两种失败态统一用 thinking=false 补一发(模型几秒内直出完整 SQL,
                #   与评委同一无思考口径), 不改用户端点设置——正常首发零开销, 只有翻车才补。
                if (jtype == "sql" and not err and resp):
                    _starved = (not (resp.get("content") or "").strip()
                                and (resp.get("reasoning") or ""))
                    _trunc = bool((resp.get("content") or "").strip()) and resp.get("finish") == "length"
                    if _starved or _trunc:
                        resp2 = await call_endpoint(client, ep["base_url"], ep["model"], msgs,
                                                    tools=valid_tools, max_tokens=mt,
                                                    cancel_flag=cancel_flag, temperature=temperature,
                                                    api_key=ep.get("api_key"), total_timeout=timeout,
                                                    payload_extra=_JUDGE_NO_THINK)
                        if not resp2.get("error") and (resp2.get("content") or "").strip():
                            resp = resp2
                            err = resp2.get("error")
                score = None
                sql_check = None
                if not err:
                    scorer = SCORERS.get(jtype, score_contains)
                    # thinking 端点答案可能放在 reasoning (content 结构化为空) -> 合并评分
                    out_for_score = resp
                    if jtype not in ("bfcl", "subjective"):
                        out_content = (resp.get("content") or "").strip()
                        if not out_content:
                            out_content = resp.get("reasoning") or ""
                        out_for_score = out_content
                    jp = dict(q["judge"].get("params") or {})
                    if jtype == "sql":
                        # 2026-09-14: SQL 判分 = 真实执行 + 四项结果比对, 评委可选。
                        #   执行判分可用 → 以它为准(1.0/0.0); 评委只在「执行判 0 且没报错」时补一个第二意见。
                        #   执行判分不可用(表缺失) → 回退旧行为(有评委用评委, 否则 None=已答未评分)。
                        score, sql_check = await judge_sql_exec_async(q, out_for_score)
                        if score is None:
                            if judge_endpoint:
                                score = await judge_sql(client, judge_endpoint, q, out_for_score,
                                                        cancel_flag=cancel_flag)
                                if sql_check is None:
                                    sql_check = {"method": "llm_judge(fallback)"}
                        elif judge_endpoint and score == 0.0 and sql_check and not sql_check.get("error"):
                            llm = await judge_sql(client, judge_endpoint, q, out_for_score,
                                                  cancel_flag=cancel_flag)
                            if llm is not None:
                                sql_check["judge_llm"] = llm
                    else:
                        if jtype == "code":
                            jp["prompt"] = q.get("prompt", "")
                        score = scorer(out_for_score, q.get("answer", ""), **jp)
                        if jtype == "subjective":
                            s = None
                            if judge_endpoint:
                                s = await judge_subjective(client, judge_endpoint, q,
                                                           (resp.get("content") or "") if resp.get("content") else (resp.get("reasoning") or ""),
                                                           cancel_flag=cancel_flag)
                            score = s if s is not None else None
                else:
                    score = 0.0
                async with lock:
                    if err and err != "cancelled":
                        st["fail"] += 1
                        if st["fail"] >= EP_FAIL_LIMIT:
                            st["tripped"] = True
                    elif not err:
                        st["fail"] = 0
                    # 全部端点熔断 -> 提前中止整个运行, 不再发新请求
                    if all(s["tripped"] for s in ep_state.values()):
                        # 把熔断事实告诉调用方(否则与用户手动取消无法区分, 历史显示成
                        # 扑朔迷离的 "cancelled")。
                        if trip_info is not None:
                            trip_info["tripped"] = [n for n, x in ep_state.items() if x["tripped"]]
                        cancel_flag and cancel_flag.set()
                    ep_done[ep["name"]] += 1
                    if on_per_ep:
                        on_per_ep(ep["name"], ep_done[ep["name"]], total_n)
                return {"qid": q["id"], "benchmark": q["benchmark"], "difficulty": q["difficulty"],
                        "diff_tier": diff_tier(q),   # 五档难度(小白/简单/中等/困难/极难)
                        "domain": q["domain"], "endpoint": ep["name"], "score": score,
                        "output": (resp.get("content") or "")[:4000],
                        "reasoning": (resp.get("reasoning") or "")[:4000],
                        "tool_calls": json.dumps(resp.get("tool_calls") or [], ensure_ascii=False)[:3000],
                        "latency": resp.get("latency"), "error": err, "answer": q.get("answer"),
                        "judge": q["judge"], "max_tokens": q.get("max_tokens", 2048), "round": rnd,
                        # SQL 题: 执行判分明细(比对四项 + 实际结果), 供 inspector 显示"为什么 0 分"
                        **({"sql_check": sql_check, "score_method": sql_check.get("method")}
                           if sql_check else {})}
            finally:
                if on_ep_done:
                    try:
                        on_ep_done(q["id"], rnd, ep["name"])
                    except Exception:
                        pass

    async def pool_worker():
        nonlocal done
        while True:
            if cancel_flag and cancel_flag.is_set():
                return
            try:
                q, rnd = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            qres = []
            for ep in endpoints:
                if cancel_flag and cancel_flag.is_set():
                    break
                qres.append(await one_ep(q, ep, rnd))
            if qres:
                async with lock:
                    results.extend(qres)
                    done += 1
                    if on_progress:
                        on_progress(done, total_n)
                    if results_cb:
                        await results_cb(qres)
            queue.task_done()

    try:
        await asyncio.gather(*[pool_worker() for _ in range(n_workers)])
    finally:
        await client.aclose()
    return results
