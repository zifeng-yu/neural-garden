# Neural Garden

> 一个面向个人 Markdown 知识库的 AI Engineering 项目：记录、连接、检索并保留证据。

Neural Garden 将 Markdown 文档处理为可检索的 Knowledge Unit 和 Concept，使用 SQLite 保存结构化数据与证据关系，使用 ChromaDB 保存向量数据，并通过 NetworkX 构建 Document–Concept 知识图谱。

项目当前处于从“功能原型”走向“可评估、可恢复、可交付 AI 应用”的工程化阶段。当前检索基线是 Knowledge Unit Vector Retrieval；Graph-Augmented Retrieval、Hybrid Retrieval 和基于证据的回答生成尚未进入默认搜索链路。

## 当前能力与边界

| 能力 | 当前状态 |
|---|---|
| Markdown + Recursive Character Chunking | 已实现，当前参数为 500/100 字符 |
| Knowledge Unit 提取 | 已实现，LLM 提取 title/summary/keywords |
| Concept 提取与归一化 | 已实现，Embedding 召回候选后由 LLM 判断同一实体 |
| Concept Relation 与 Evidence | 已实现，关系可回溯到 document/chunk |
| SQLite 结构化存储 | 已实现，10 张表 |
| ChromaDB 向量存储 | 已实现，knowledge/concept/insight 三个 collection |
| NetworkX 知识图谱 | 已实现，运行时从 SQLite 构建 `MultiDiGraph` |
| Insight | 已实现，保存 title/content/action_items/relation_concepts |
| 搜索与反馈数据 | 已实现，记录 session/result/feedback event |
| Retrieval Evaluation | 下一阶段，当前尚无可对比的离线 baseline |
| BM25/Hybrid/Reranker | 尚未实现 |
| Graph-Augmented Retrieval | 尚未接入主搜索链路 |
| Answer Generation/Citation | 尚未实现，因此当前不宣称是完整 RAG 系统 |
| HTTP API/Agent/MCP | 尚未实现 |

## 核心架构

```text
Markdown documents
       │
       ▼
Markdown header split + recursive character split
       │
       ├──────────────► LLM Knowledge Unit extraction
       │                         │
       │                         ▼
       │                 title + summary + keywords
       │                         │
       │                         ▼
       │                   Embedding / Chroma
       │
       └──────────────► Concept extraction
                                 │
                                 ▼
                     Embedding candidate recall
                                 │
                                 ▼
                       LLM concept normalization
                                 │
                                 ▼
                    Relation extraction + evidence

       ┌─────────────────────────┴─────────────────────────┐
       ▼                                                   ▼
SQLite                                               ChromaDB
metadata / relations / evidence                      vector search
       │
       ▼
NetworkX MultiDiGraph
```

### 存储职责

SQLite 是结构化数据的主要事实来源：

- 文档与分块：`documents`、`document_chunks`
- AI 抽取结果：`document_chunk_knowledge_units`、`document_chunk_concepts`
- 图关系与证据：`concept_relations`、`relation_evidence`
- 用户洞察：`insights`
- 搜索反馈：`search_sessions`、`search_results`、`feedback_events`

ChromaDB 保存三个向量 collection：

- `knowledge`：Knowledge Unit 的 title + summary + keywords
- `concept`：归一化 Concept
- `insight`：Insight 的结构化拼接文本

NetworkX 不作为持久化数据库。图在运行时从 SQLite 中的 Document、Concept 和 Relation 数据构建。

## 数据处理流程

### 1. 文档索引

1. 读取 `data/pilot/` 中的 Markdown 文档。
2. 先按 `#`、`##`、`###` 标题切分，再进行 Recursive Character Split。
3. 为每个 chunk 调用 LLM 提取 Knowledge Unit。
4. 提取并归一化 Concept。
5. 将文档、chunk、Knowledge Unit、Concept 保存到 SQLite。
6. 生成 Knowledge Unit/Concept Embedding 并写入 ChromaDB。
7. 从 chunk 中抽取 Concept Relation，并保存 chunk-level `relation_evidence`。

### 2. 增量处理

索引器基于文件名 hash 与内容 hash 判断处理路径：

- `skip`：文件名和内容均未变化，不重复处理。
- `del/new`：同一文件内容发生变化，删除旧数据后重新处理。
- `copy`：不同文件具有相同内容，复用 SQLite 中的抽取结果并为新的向量记录重新生成 Embedding。
- `new`：处理全新文档。

当前更新流程仍存在跨 SQLite/Chroma 的部分失败窗口，详见“当前限制”。

### 3. 搜索

当前 `search()`：

- 仅查询 ChromaDB `knowledge` collection。
- 返回结构化 `SearchResult`，包含 content、document/chunk ID、score、rank 和 retrieval/source type。
- 默认保存 `search_sessions` 与 `search_results`；离线实验可使用 `save_session=False` 关闭写入。

Insight 和 Graph 尚未参与默认检索排序。

## 快速开始

### 环境要求

- Python `>=3.12.10`
- [uv](https://docs.astral.sh/uv/)
- DashScope API Key

### 1. 安装锁定依赖

```bash
uv sync --locked
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
API_KEY=sk-your-api-key-here
```

API Key 只在实际调用 DashScope Embedding/LLM 时读取；导入模块和运行不依赖外部 API 的测试不需要有效 Key。

### 3. 初始化 SQLite

SQLite 数据库路径由 `config.yaml` 配置。首次运行索引器时，项目会自动创建父目录、数据库文件和缺失的数据表，无需手动初始化。

### 4. 索引 Markdown

仓库已包含小型 `data/pilot/` 示例语料，也可以替换为自己的 Markdown 文件：

```bash
uv run python -m src.indexer
```

完全清空 SQLite 和三个 Chroma collection 后重建：

```bash
uv run python -m src.indexer --resetAllDB
```

`--resetAllDB` 会删除现有索引、关系、反馈和 Insight 数据，仅应在确认需要全量重建时使用。

### 5. 搜索 Knowledge Unit

```bash
uv run python -m src.search "什么是负利率"
```

### 6. 创建 Insight

```bash
uv run python -m src.insight \
  --title "一句话总结" \
  --content "洞察内容" \
  --action-items '["行动 1", "行动 2"]' \
  --relation-concepts '["概念 1", "概念 2"]'
```

可选的 `--source-title` 可以把 Insight 关联到已有 Knowledge Unit title 或文档文件名。未显式提供 `relation_concepts` 时，系统依次尝试复用 chunk/document Concept，最后才调用 LLM 提取。

### 7. 构建图

```python
from src.graph import build_graph, graph_png, log_stats

graph = build_graph()
if graph is not None:
    log_stats(graph)
    graph_png(graph)
```

## 配置

主要配置位于 `config.yaml`：

```yaml
dashscope:
  embedding_model: "text-embedding-v1"
  llm_model: "qwen3.5-plus"

chroma:
  persist_directory: "data/chroma"
  knowledge_table_name: "knowledge"
  concept_table_name: "concept"
  insight_table_name: "insight"

pilot_dataset:
  path: "data/pilot"

sqlite:
  sqlite_directory: "data/sqlite/neural.db"
```

相对路径在运行时解析为项目根目录下的绝对路径，因此命令不依赖当前 shell 工作目录。

## 测试与质量门

本地执行与 GitHub Actions 使用相同的核心检查：

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src tests
API_KEY= uv run pytest
```

当前测试覆盖：

- Markdown splitter、hash、similarity 等纯函数
- SQLite Repository（使用 `:memory:`）
- Chroma 基础行为（使用临时目录）
- Knowledge Vector Search 的排序、持久化开关和失败分支
- Insight/Feedback 的部分 Repository 行为

测试不会访问真实 DashScope API，也不会修改项目的 SQLite/Chroma 数据。当前仍缺少完整 ingestion、Graph 和 Insight 跨存储流程的端到端测试。

CI 配置：[`.github/workflows/ci.yml`](.github/workflows/ci.yml)

## 项目结构

```text
neural-garden/
├── .github/workflows/ci.yml
├── .env.example
├── config.yaml
├── pyproject.toml
├── uv.lock
├── data/pilot/                     # 示例 Markdown 语料
├── src/
│   ├── indexer.py                  # ingestion 与增量处理
│   ├── search.py                   # Knowledge Vector Search
│   ├── graph.py                    # 图构建、查询和可视化入口
│   ├── insight.py                  # Insight 业务入口
│   ├── feedback.py                 # Feedback 业务入口
│   ├── config/                     # 配置与日志
│   ├── document/                   # Markdown splitter
│   ├── embedding/                  # DashScope Embedding
│   ├── knowledge/                  # Knowledge Unit 提取
│   ├── knowledgeGraph/             # Concept/Relation/Graph
│   ├── repository/                 # SQLite Repository
│   ├── vector_store/               # Chroma Repository
│   └── feedback_events_analytics/  # 反馈统计
└── tests/
    ├── test_core_modules.py
    ├── test_repository.py
    ├── test_insights.py
    ├── test_feedback.py
    ├── test_vector_store.py
    └── test_search.py
```

运行时生成的 `.env`、SQLite、Chroma、日志、coverage 和缓存文件不会提交到 Git。

## 关键设计取舍

### 为什么同时使用 SQLite 和 ChromaDB

SQLite 适合保存结构化实体、关系、状态和证据；ChromaDB 适合语义相似度召回。两者职责不同，当前项目不尝试用一个存储替代另一个。

### 为什么保存 relation evidence

Concept Relation 是 LLM 抽取结果，不能只保存无来源的图边。`relation_evidence` 将 relation 关联回 document/chunk，为后续 Graph Retrieval、citation 和关系质量评估保留依据。

### 为什么向量化 Knowledge Unit 而不是原始 chunk

当前 baseline 向量化 title + summary + keywords，希望减少原始文本噪声；但这可能损失精确数字和细节。项目不会预设该策略最优，后续将通过 Retrieval Evaluation 与 Raw Chunk Embedding 做受控比较。

### 为什么暂时使用 SQLite/NetworkX

当前是单机个人项目，SQLite 和 NetworkX 足以验证数据模型、检索策略与失败恢复设计。没有真实规模需求前，不为了技术栈数量引入 PostgreSQL、Neo4j、Elasticsearch 或分布式工作流系统。

## 当前限制与下一步

当前明确存在以下工程边界：

- 文档更新采用先删除旧数据再重建的方式，新处理失败时可能失去上一可用版本。
- SQLite 与 ChromaDB 没有共享事务，中途失败可能产生 missing/stale vectors。
- 内容 hash 未包含 prompt、model、embedding 和 chunk 配置版本，配置变化不会自动触发重建。
- 当前目录扫描不会自动清理已经从源目录物理删除的文档。
- Embedding 调用没有统一 retry/backoff。
- 当前没有 Retrieval Evaluation baseline，无法量化现有检索质量。
- 当前没有 Answer Generation、citation 或 no-answer 判断。

下一步按以下顺序推进：

1. 建立版本化 Retrieval Evaluation 数据集、指标和 Vector baseline。
2. 比较 BM25、Vector、RRF Hybrid，并按结果决定是否加入 Reranker。
3. 增加 ingestion status、pipeline version、旧版本保护和 SQLite–Chroma reconciliation。
4. 评估 Evidence-backed Graph Retrieval，再决定是否进入回答生成链路。

---

*Project by 俞紫峰 · 2026*
