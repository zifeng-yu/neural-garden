# Neural Garden

> 面向个人 Markdown 知识库的 AI 应用：提取知识、连接概念、检索原文并保留回答证据。

项目状态更新：2026-10-15。

Neural Garden 将 Markdown 文档转换为知识单元（Knowledge Unit）和概念（Concept），支持向量检索、BM25、混合检索和图检索，并基于原文生成带引用的回答。项目提供命令行、HTTP API、LangGraph Agent 和本地 MCP 入口。

当前已实现单机知识库的主要功能，工程重点是检索与回答评估、入库一致性和失败恢复。知识图谱已作为可选模式和增强步骤接入统一检索、RAG Python 接口、知识 Agent 工具及 MCP；默认仍使用向量检索，不开启图增强。Insight 独立保存，尚未参与主检索和问答链路。

## 主要功能与完成情况

| 功能 | 当前实现 |
|---|---|
| 文档索引 | 读取 Markdown，按标题和递归字符分块，当前分块参数为 500/100 字符 |
| 知识提取 | LLM 从分块提取 title、summary、keywords，形成知识单元 |
| 概念与关系 | 提取概念，结合向量候选和 LLM 进行归一化；概念关系保留 document/chunk 级证据 |
| 增量处理 | 根据文件名和内容 hash 选择跳过、更新重建、同内容复用或新文档处理 |
| 知识检索 | 支持 vector、bm25、hybrid、graph 四种模式；前三种可开启图增强，默认使用向量检索 |
| RAG 问答 | 检索后读取 SQLite 原文，组织来源并生成带引用的断言；支持证据不足返回 |
| 知识图谱 | 从 SQLite 构建 NetworkX MultiDiGraph，支持概念匹配、一跳扩展、检索融合，以及遍历、统计和可视化 |
| 用户洞察 | 保存 Insight 内容、行动项、关联概念和来源，并生成独立向量 |
| 搜索与反馈 | 保存搜索会话、最终结果及反馈事件，支持按日期查询热门词、零结果词和文档点击 Top5 |
| Agent | 两个 LangGraph 入口：知识检索与原文查证、使用指标查询；支持进程内追问 |
| HTTP API | FastAPI 提供健康检查和默认向量 RAG 问答 |
| MCP | 本地 stdio 服务，提供两个 Tools、一个原文 Resource 模板和一个查证 Prompt |

尚未实现 Reranker、远程 MCP 服务和资源订阅。图增强已有实现和离线功能测试，尚未完成真实检索质量对照评估。

## 架构与主要链路

```text
Markdown
  → 标题切分 + 递归字符分块
  → 知识单元提取 / 概念提取与归一化
  → SQLite 结构化存储 + Chroma 向量存储
  → 概念关系提取与原文证据保存

SQLite + Chroma → 显式同步 → Elasticsearch BM25 索引
SQLite         → 运行时构建 → NetworkX 知识图谱
```

| 链路 | 主要节点与输出 |
|---|---|
| 搜索 | 查询 → vector/bm25/hybrid/graph 召回 → 可选图增强与 RRF 融合 → SearchResult 列表 |
| RAG | 搜索 → 原文身份去重 → SQLite 原文 → S1/S2 来源 → LLM 断言与引用校验 |
| 知识 Agent | 模型与工具循环 → search_knowledge 定位 → read_chunk 查证 → 带 D#:C# 出处的回答 |
| 指标 Agent | 模型选择统计工具 → 日期校验 → SQLite 聚合 → 指标结果与解释 |
| HTTP | 请求参数校验 → 默认向量 RAG → 回答、断言和原文来源 |
| MCP Tools | 外部 Agent 调用检索和原文读取工具 → 返回资料 → 客户端组织回答 |
| MCP Resource | resources/read 按原文 URI 获取 JSON 资料 |
| MCP Prompt | prompts/get 填入问题 → 返回一条用户角色提示消息 |

MCP 服务不运行内部 LangGraph 循环，也不生成最终回答。HTTP API 调用固定 RAG 流程；知识 Agent 由模型决定工具调用。三类入口复用已有检索和原文能力。

### 存储职责

| 组件 | 保存内容与职责 |
|---|---|
| SQLite | 主要事实来源，保存文档、原文分块、知识单元、概念、关系证据、Insight、搜索和反馈，共 10 张表 |
| ChromaDB | knowledge、concept、insight 三个集合，分别保存知识单元、归一化概念和洞察向量 |
| Elasticsearch | 可重建的 BM25 检索副本，保存知识单元检索文本、ID 和原文 hash，不保存向量 |
| NetworkX | 运行时图结构，从 SQLite 构建，不承担持久化存储 |

知识单元向量化内容为 title + summary + keywords，ES 的 retrieval_text 保存同一份知识单元文本，回答证据则使用 SQLite 原始 chunk。ES 的 chunk_hash 对应原文版本。Elasticsearch 需要显式同步，文档索引器不会自动更新它。

### 文档更新行为

- 文件名和内容均未变化：跳过处理。
- 同名文件内容变化：删除旧数据后重新处理。
- 不同文件具有相同内容：复用 SQLite 提取结果，为新知识单元记录生成向量。
- 全新文档：完成分块、抽取、存储、向量化及关系证据处理。

目前没有跨 SQLite、Chroma 和 ES 的统一事务，也没有旧版本保护和自动恢复。首次入库也可能部分完成：知识提取失败时原文仍可保存，Embedding 失败时向量会缺失；同一文件重跑可能因内容 hash 未变化而直接跳过，不会自动补齐。

离线批量新增时，应先完成 SQLite/Chroma 入库，再显式同步 ES。同步前的 export_data() 会核对已有知识单元的 ID、文本和文档/分块关联；它不检查每个原文分块是否都成功生成知识单元，因此同步校验通过不等于所有分块均已可检索。

## 接口与运行入口

项目使用 Python `>=3.12.10` 和 uv 管理依赖，依赖锁定在 `uv.lock`。模型与 Embedding 使用 DashScope；实际模型调用需要 `.env` 中的 `API_KEY`。存储、模型、ES、日志和 Agent 参数统一由 `config.yaml` 配置。

### 命令行与 Python 入口

以下模块在项目根目录运行：

| 入口 | 用途 |
|---|---|
| `uv run python -m src.indexer` | 索引配置目录中的 Markdown |
| `uv run python -m src.repository.init` | 创建缺失 SQLite 表，保留已有数据 |
| `uv run python -m src.search "查询内容"` | 搜索知识单元，可选择 vector、bm25、hybrid、graph，并可开启图增强 |
| `uv run python -m src.es --mode upsert` | 校验 SQLite/Chroma 后同步 ES；另支持 skip、reset |
| `uv run python -m src.rag_answer "问题"` | 默认向量 RAG 问答 |
| `uv run python -m src.insight` | 创建 Insight，接收标题、内容、行动项等参数 |
| `uv run python -m src.knowledge_retrieval_agent_cli "问题"` | 知识检索 Agent |
| `uv run python -m src.usage_metrics_agent_cli "统计问题"` | 使用指标 Agent |
| `uv run python -m src.mcp_server` | 本地 stdio MCP 服务，通常由客户端启动 |
| `uv run python -m uvicorn src.api:app --host 127.0.0.1 --port 8766` | HTTP API |

常用检索命令：

```bash
uv run python -m src.search "HashSet 如何检测重复元素" --mode hybrid --top-k 3
uv run python -m src.search "专家预测" --mode graph --top-k 3
uv run python -m src.search "专家预测" --mode vector --use-graph --no-save-session
```

`--no-save-session` 关闭本次搜索记录保存。BM25 和混合检索需要已同步的 ES 索引；向量检索需要已有 knowledge 集合；图检索在 SQLite 存在概念但名称匹配无命中时读取已有 concept 集合，集合非空时调用查询 Embedding。

`answer_question(question, top_k=3, retrieval_mode="hybrid", use_graph=True)` 可在 Python 中选择 RAG 检索方式。`src.rag_answer` 命令行入口仍只接收问题文本，未提供模式参数；HTTP API 也仍固定使用默认向量检索。

图谱另通过 `src.graph.build_graph`、`search`、`log_stats` 和 `graph_png` 提供构建、遍历、统计及可视化。图片输出到 `graphPNG/`。

重置操作范围：`src.repository.init --reset` 仅重建 SQLite 项目表；`src.indexer --resetAllDB` 清空 SQLite 及三个 Chroma 集合后重新索引，均不重置 ES。ES 完整重建由 `src.es --mode reset` 执行。重置会删除对应存储中的已有业务数据。

### 检索与回答契约

`search()` 返回 `list[SearchResult]`，包含内容、document/chunk ID、分数、原始分数、排名与检索来源。正常零命中返回空列表，执行失败抛异常。混合检索按原文身份去重，通过 RRF 融合名次。不同模式的 score 含义不同，不能直接作为统一的相似度比较。

`search()` 和搜索 CLI 的 top_k 只要求正整数；RAG、HTTP API、知识 Agent 检索工具及 MCP 将其限制在 1～5。直接调用 hybrid_search() 时，candidate_k 必须不小于 top_k；统一 search() 的 hybrid 路径使用 top_k × 3 作为两路召回数量。

向量和混合检索使用已有 knowledge 集合，目录或集合缺失时抛异常，不自动创建空集合。已有空集合是正常状态，混合检索仍可返回 BM25 候选。

搜索默认在同一 SQLite 事务中保存 session 与最终结果；混合模式不重复保存两路候选。`save_session=False` 关闭记录保存，RAG、知识 Agent 和 MCP 内部检索均使用此设置。

### 图检索与图增强

- `retrieval_mode="graph"`：优先匹配概念名称和别名，没有名称命中时使用概念向量匹配；从起始概念及一跳邻居关联的知识单元中返回候选。
- `use_graph=True`：用于 vector、bm25、hybrid，先取 top_k × 2 个基础候选，再从候选关联概念展开，以 RRF 融合基础与图候选，最终返回最多 top_k 条。graph 模式直接执行图检索，不重复增强。
- 名称匹配不区分大小写；英文端点检查字母、数字和下划线边界，避免 AI 命中 explain；中文保留子串匹配，并支持“介绍AI技术”这类中英混排。
- 一跳扩展跳过 mentions 边，要求关系存在且有 evidence_role=both 的证据记录。扩展返回的是邻居概念关联的知识单元，不等同于关系证据原文本身；回答仍需读取对应 chunk 核验。

图相关请求目前每次从 SQLite 重建图，尚无缓存；候选数量和一跳关系数量受限。图增强只保存最终融合结果，不重复保存基础召回。

### RAG 来源与回答

RAG 读取原文并在 3000 字符预算内组织来源，返回 status、claims、sources。状态为 answered 或 insufficient_evidence；原文不可用、预算不足或模型响应校验失败抛异常。引用编号校验保证引用存在，不保证断言语义完整或忠实。

### HTTP API

| 接口 | 行为 |
|---|---|
| `GET /health` | 返回服务状态，不检查数据库或模型服务可用性 |
| `POST /answers` | 接收 question 和 top_k，调用默认向量 RAG，返回 status、claims、sources |

question 去除首尾空白后为 1～200 字符；top_k 为 1～5 的 JSON 整数，默认 3。额外字段或非法参数返回 422。证据不足返回 200，执行异常返回 500；回答来源和引用编号原样保留。API 当前不提供检索模式或反馈写入参数。

### Agent

两个 Agent 共用 `src/agent_graph/agent_graph.py`，由调用方提供 SYSTEM_PROMPT，执行模型与工具循环。会话检查点使用 InMemorySaver；同进程的 `--followup` 可以沿用上下文，进程退出后不保留会话。

知识 Agent 的 search_knowledge 工具支持四种 retrieval_mode 和 use_graph，由模型选择工具参数；摘要用于定位，read_chunk 返回原文与 D#:C# 出处。执行状态记录本轮模型调用次数和成功读取原文次数；这些计数不代表回答质量或引用语义已经通过校验。

指标工具为 query_popular_queries、query_zero_result_queries、query_popular_documents。日期使用上海时区的左闭右开区间，结果为分组次数或点击事件次数，不代表人数、点击率或完整记录总数。

### MCP

当前使用官方 Python SDK 2.3.0，依赖范围为 `>=2,<3`，传输方式为 stdio。协议消息通过 stdout 传输，应用日志写入 stderr 和日志文件。

| 类型 | 接口 | 行为 |
|---|---|---|
| Tool | `search_knowledge(query, top_k=3, retrieval_mode="vector", use_graph=False)` | 返回检索模式及最多五个候选摘要、document/chunk ID 和排名；每条摘要最多 300 字符，不保存搜索记录 |
| Tool | `read_chunk(document_id, chunk_id)` | 校验 ID 对应关系，返回原文、D#:C# 出处、hash 和截断标记，最多 4000 字符 |
| Resource 模板 | `ng://documents/{document_id}/chunks/{chunk_id}` | 返回 application/json 原文资料，复用 read_chunk 的读取逻辑；缺失原文抛 ResourceNotFoundError |
| Prompt | `grounded_answer(question)` | 返回包含查证要求和输入问题的一条用户角色消息，不自动调用工具或模型 |

retrieval_mode 支持 vector、bm25、hybrid、graph；query 去除首尾空白后为 1～200 字符，top_k 为 1～5。向量检索调用查询 Embedding，混合检索在 knowledge 集合非空时调用，BM25 不调用；图检索仅在名称匹配无命中且概念集合非空时调用。use_graph 对前三种模式开启图增强。

资源模板由 resources/templates/list 发现，具体 URI 由 resources/read 读取；当前没有固定 URI 列表。Prompt 由 prompts/list 发现、prompts/get 获取。Server instructions 描述检索、原文查证与引用规则，实际执行由客户端决定。

此前验证范围：Tools 在 Codex 中使用成功；Resource 和 Prompt 在 Inspector 与独立 Python stdio 客户端中获取成功。独立调用已验证 `2025-11-25`、`2026-07-28` 两个协议版本下的原文一致性、Prompt 参数填充和缺失资源报错。上述记录不代表新增图模式已通过真实客户端端到端复测。Codex 的 Resource 实际加载与原生 MCP Prompt 选择入口尚未核验。

当前没有 Streamable HTTP、远程认证、资源订阅或插件打包。只读标记是行为提示，原文底层查询仍使用普通 SQLite 连接，尚未强制限制整条读取链路的存储权限。

## 测试与评估

项目通过 pytest、Ruff 和 mypy 进行自动化检查，CI 配置位于 [ci.yml](.github/workflows/ci.yml)。2026-10-15 最新本地工作区验证：**258 项测试通过**，Ruff 检查、格式检查及 mypy 通过。混合检索的集合缺失、已有空集合和图检索的英文概念边界匹配均有回归用例。测试仍有一条 DashScope SDK 的 Assistants API 弃用警告，不影响通过。

测试覆盖分块、Repository、向量/BM25/混合检索、图名称匹配与向量回退、一跳扩展与图增强分发、ES 同步、搜索记录事务、RAG 引用与预算、Agent 工具循环、HTTP 契约、指标日期边界及配置加载。测试使用隔离存储和外部调用替身，不调用真实 DashScope 或 ES。

MCP 已有实际连接与独立客户端验证，但尚无持久化的专用协议回归测试；完整 ingestion 和 Insight 跨存储端到端测试也未建立。图检索已有临时 SQLite、真实 NetworkX 与模拟向量服务的测试，尚无真实 Embedding/LLM 的完整端到端质量验证。功能测试不代表真实模型的选工具或回答质量。

[评估目录](evaluation/README.md) 保存检索/RAG、知识检索 Agent、指标 Agent 三类资料：

| 类别 | 当前证据与限制 |
|---|---|
| 检索与 RAG | 7 题、三种模式共 21 条历史结果；发现限定词遗漏、有证据却拒答和引用支持不完整的问题 |
| 知识 Agent | 4 个完整历史场景及 1 个不完整追问片段；存在无资料追问转向无关材料、回答扩写的问题 |
| 指标 Agent | 历史模型日志使用旧版单工具；当前三个独立工具已有离线测试，模型选工具行为尚未完整复测 |

历史评估尚未按当前版本全部重跑，缺少完整语料快照和可重复执行入口，部分标注经过事后订正，不能据此判断整体泛化效果。

## 项目结构

```text
neural-garden/
├── config.yaml                    # 模型、存储、检索与运行配置
├── pyproject.toml / uv.lock        # 依赖声明与锁定
├── data/pilot/                    # 示例 Markdown
├── src/
│   ├── indexer.py                  # 入库与增量处理
│   ├── search.py / retrieval/      # 检索分发、向量、BM25、RRF、图检索与增强
│   ├── rag_answer.py / api.py      # 引用问答与 HTTP 接口
│   ├── mcp_server.py               # Tools、Resource、Prompt
│   ├── agent_graph/ / tools/       # Agent 编排与工具
│   ├── *_agent_cli.py              # 知识与指标 Agent 入口
│   ├── document/ / knowledge/      # 分块与知识提取
│   ├── knowledgeGraph/ / graph.py  # 概念、关系与图谱
│   ├── repository/ / vector_store/ # SQLite 与 Chroma
│   ├── es/                        # ES 管理与同步
│   ├── insight.py / feedback.py    # 洞察与反馈
│   ├── feedback_events_analytics/  # 使用指标聚合
│   └── config/ / util/ / embedding/
├── tests/                         # 隔离自动化测试
├── evaluation/                    # 评估方法、数据集与结果
└── .github/workflows/ci.yml
```

`.env`、业务 SQLite/Chroma、运行日志和缓存不提交到 Git；明确归档的评估日志作为证据保留。

## 当前限制与后续方向

- **入库恢复**：首次知识提取或 Embedding 失败可能留下部分入库状态，同内容重跑会跳过，尚无自动补提取或补向量机制；更新先删除旧数据，失败时可能丢失上一可用版本。批量命令结束和 ES 导出校验通过均不能证明每个原文分块已可检索。
- **数据版本**：hash 不包含模型、提示词或分块配置版本；源目录物理删除的文件不会自动清理存储数据。
- **ES 一致性**：需要手动同步，upsert 不删除旧 ID；重建后旧记录可能关联到复用的原文 ID，RAG 尚未校验 ES 原文 hash。
- **存储约束**：SQLite 未启用外键约束；概念归一化对模型返回候选范围的校验不完整。
- **图谱边界**：已接入可选检索与问答链路，但每次请求重建图，一跳扩展及候选截断的检索质量尚未评估；图统计不能直接解释为文档数量或热度。独立 src.graph 演示入口仍包含固定 ID 逻辑。
- **反馈闭环**：搜索入口未返回保存后的 session_id，交互端反馈关联未完善。
- **运行与 MCP**：Embedding 没有统一 retry/backoff；MCP 输出字段约束和专用协议测试仍待补齐。Inspector 网页端曾间歇延迟与超时，根因未确认。
- **效果验证**：评估仍为小样本历史记录，尚不能证明检索模式整体优劣或当前回答质量。

后续重点是建立版本化检索评估与 baseline，在相同语料和预算下比较 Vector、BM25、Hybrid、Graph 及图增强，再决定是否调整默认模式或加入 Reranker；同时完善入库状态、失败补处理、旧版本保护及跨存储一致性检查。当前没有证据证明图增强能整体提高回答质量。

---

*Project by 俞紫峰 · 2026*
