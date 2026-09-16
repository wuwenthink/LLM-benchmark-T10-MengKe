# 梦客AI工具箱-推理质量测试（纯 Python 单机版）

一个**纯 Python 文件夹**式的大模型推理质量评测台：不依赖 Docker / Node / 任何集群组件，
任意装了 Python 3.9+ 的机器上双击启动即可用浏览器跑评测。

> 欢迎加 QQ 群 **1028429001**，讨论本地部署 deepseek-v4.1 的相关方案！

- 后端：FastAPI + uvicorn + httpx（单进程，评测在后台异步跑）
- 前端：单文件 `web/index.html`（内联 JS/CSS，无需构建）
- 判分：确定性判分优先（SQLite 真实执行、单元测试、正则/选项/数值比对），主观题可选 LLM 评委

## 使用方法

### Windows
双击 **start.bat**（首次会自动建 `.venv` 并装依赖，然后启动服务并自动打开浏览器）。

### Linux / macOS / 其他
```bash
bash start.sh
# 或手动：
# python3 -m venv .venv && . .venv/bin/activate
# pip install -r requirements.txt
# python3 server.py
```

启动后浏览器打开 **http://127.0.0.1:17889/** 即可使用。

## 功能

- **多档位评测**：SQL（25 题，SQLite 真实执行比对判分，无需评委）+ T1~T10 通用题集
  （随 `data/questions.jsonl` 附带：知识 / 数学 / 代码 / 阅读 / 推理 / 指令遵循 / 工具调用 / 长文本 / 主观等 10 个领域）。
- **LLM-Judge（主观题评分）可选**：在页面勾选「启用 LLM-Judge」填 OpenAI 兼容端点即可；
  不启用完全不影响 SQL 等确定性题目的判分。
- **实时热力图**：题号表头 Q01..QNN、行内唯一蓝格=推进前沿、并发数写在行标签 `+N 在途`、
  格数恒等于 题数×端点×轮次；格子矩阵**每行最多 20 格**，超过 20 格自动换行继续显示。
- **多端点并发**：最多 8 个端点同时跑同一套题（用于模型横向对比），失败重试 + 连续失败自动熔断。
- **断点续测 / 历史 / 结果导出 CSV/JSON**；停止后可随时从历史续跑，只补测未完成的题轮。
- **题库浏览**：领域 → 类别 → 题目卡片 → 详情抽屉，可直接查看题面、参考答案与判分口径。

## 端点（被测模型）怎么填

页面底部「端点列表」手动填写 OpenAI 兼容推理服务：
- **名称（= 测试名称）**：**留空即默认显示模型名称**（如 `qwen2.5-7b-instruct`），需要区分时才自定义（如 `demo-box:8000`）
- base_url：如 `http://127.0.0.1:8000`（不要带 `/v1`，程序会自动补）
- model：如 `qwen2.5-7b-instruct`
- API Key：需要鉴权的服务才填（只用于请求头，不会落盘到历史记录）
- 思考：模型带思考模式（reasoning）就打勾

> 「🔄 自动获取运行中推理」按钮默认提示无进程：本工具不扫描任何集群。
> 若你的部署能提供推理实例清单，按 `server.py` 里 `/api/snapshot` 的注释实现该接口即可一键生成端点。

## 目录结构

```
.
├── start.bat            # Windows 一键启动
├── start.sh             # Linux/macOS 一键启动
├── server.py            # 入口: 路径设定 + 静态页 + 挂载 /api/quality 路由 + /api/snapshot
├── quality_engine.py    # 评测引擎(并发调 OpenAI 兼容端点 + SQLite 判分)
├── quality_api.py       # 后端 API(/api/quality/*)
├── benchmark_guard.py   # 评测期间「静默闸门」钩子(单机为空实现, 可自行替换)
├── requirements.txt     # fastapi / uvicorn / httpx
├── sql_questions.jsonl  # SQL 25 题题集
├── tables/              # 7 张 SQL 参考表 CSV(判分用, AdventureWorks 示例数据)
├── data/                # 题库 questions.jsonl + 运行结果/历史/骨架(运行时生成)
└── web/index.html       # 页面(内联 js/css, 单文件)
```

## 可选环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| QTEST_HOST | 127.0.0.1 | 监听地址；局域网访问设 `0.0.0.0` |
| QTEST_PORT | 17889 | 端口 |
| QUALITY_DATA_DIR | 本目录 `data/` | 题库/结果/历史目录 |
| QTEST_SQL_TABLES | 本目录 `tables/` | SQL 参考表目录 |

## 常见问题

- **端口被占**：改 `QTEST_PORT` 或换一台机器再跑。
- **首次 pip 装依赖很慢**：属正常，仅首次。可用镜像 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`。
- **SQL 判分不可用**：确认 `tables/` 下 7 张 CSV 都在；删掉 `data/` 里的历史再跑。
- **改了 `quality_api.py` / `quality_engine.py` 后要重启服务**（页面 `index.html` 改动刷新浏览器即可）。

## 题集来源

`data/questions.jsonl` 由公开评测基准整理合并而成（HumanEval、MBPP、GSM8K、MGSM、DROP、
ARC-Challenge、BBH、MMLU-Pro、SuperGPQA、IFEval、MT-Bench、AlpacaEval、BFCL、InfiniteBench 等），
`tables/` 为 Microsoft AdventureWorks 示例数据（客户/经销商名称均为虚构示例）。
若你要再分发本仓库，请自行核对各上游数据集的许可条款（部分基准带署名或非商用限制）。

## 许可

MIT，见 `LICENSE`。

## 更新记录

- 2026-09-16 修复：
  - 热力图格子矩阵每行最多 20 格，超过自动换行。
  - 结果统计里「按领域 / 按难度」明细表独立成块，不再与统计卡片/彼此重叠。
  - 测试名称默认显示**模型名称**（留空即用模型名，可自定义；多端点同名自动补 `@主机:端口`）。
  - 停止评测后立刻新测/续跑：旧任务不再被新运行"复活"并回写结果文件（每次运行独立 cancel 事件，收尾期间新运行返回 409）。
  - 热力图在「按题目 / 按 Benchmark」与「格子矩阵」之间来回切换时正确重绘。
  - humaneval 代码题真正执行 `check(candidate)`（此前任何能编译的实现都判满分）。
  - BFCL 工具调用改用标准嵌套结构传递并对齐评分（此前单轮/多轮 BFCL 恒 0 分）；`expected` 为空时按「不得调用」判分。
