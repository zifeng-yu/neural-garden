# Neural Garden

> 个人认知系统 · 记录 · 连接 · 调用

**Neural Garden** 是一个帮助你构建个人知识系统的工具。它基于 RAG（检索增强生成）和向量检索技术，让你的知识不再是孤立的笔记，而是可以相互连接、智能检索的认知网络。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 DashScope API Key
# 获取方式：https://dashscope.console.aliyun.com/apiKey
```

### 3. 索引知识

将你的 Markdown 笔记放入 `data/pilot/` 目录，然后运行：

```bash
# 正常索引（增量更新，跳过未变化的文件）
python -m src.indexer

# 清空向量库,sqlite库后重新索引
python -m src.indexer --resetAllDB
```

**增量更新机制**：
- 首次运行：全量索引所有文件
- 后续运行：自动检测文件变化，只更新有变化的文件
- 基于 `content_hash` 判断内容是否变化，避免重复调用 LLM 和 Embedding API

### 4. 搜索知识

```bash
python -m src.search "什么是 Neural Garden"
```

---

## 核心流程

### 流程一：文档转换成向量入库

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     
│  Markdown   │ ──> │   Chunk     │ ──> │  Embedding  │ ──> │  ChromaDB   │
│   文档      │     │  分块       │     │   向量化    │     │   入库      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**详细步骤**：

#### Step 1: 读取文档
```python
# src/indexer.py: load_document()
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
```

#### Step 2: Token 级分块
```python
# src/document/splitter.py: markdown_spilt()
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 按 Markdown 标题分割
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)
sections = markdown_splitter.split_text(markdown)

# 按 Token 递归分割（保证语义完整性）
recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,      # 每块 500 tokens
    chunk_overlap=100,   # 重叠 100 tokens
)
chunks = recursive_splitter.split_documents(sections)
```

**分块策略**：
- 先按 Markdown 标题分割（保持章节结构）
- 再按 Token 递归分割（避免切断语义）
- `chunk_size=500`：平衡检索精度和上下文长度
- `chunk_overlap=100`：保证跨块语义连贯

#### Step 3: 知识单元提取
```python
# src/indexer.py: index_file()
from src.knowledge.knowledgeUnit import extract_knowledge_unit

knowledgeUnit = extract_knowledge_unit(
    content=chunk,
    source=filename,
)
# 返回：KnowledgeUnit(title, summary, keywords)
```

**知识单元结构**：
- `title`: 小节标题
- `summary`: 摘要（200-300 字）
- `keywords`: 关键词列表（3-5 个）

#### Step 4: 向量化
```python
# src/indexer.py: index_file()
from src.embedding.getEmbedding import get_embedding

embedding = get_embedding(knowledgeUnit.to_embedding_text())
# 返回：list[float] (768 维)
```

**Embedding 文本**：`标题 + 摘要 + 关键词` 的组合文本，优化检索效果。

#### Step 5: 入库（增量更新）
```python
# src/indexer.py: index_file()
from src.vector_store.save_dao import save_knowlege, KnowledgeUnitDTO

save_knowlege(KnowledgeUnitDTO(
    id=chunk_id,
    embedding=embedding,
    text=knowledgeUnit.to_embedding_text(),
    metadata=KnowledgeUnitMetadata(
        file_name=filename,
        document_id=doc_id,
        document_chunk_id=chunk_id,
        title=knowledgeUnit.title,
        keywords=knowledgeUnit.keywords
    )
))
```

**增量更新逻辑**：
1. 基于文件名 + 内容哈希判断文件是否变化
2. 未变化 → 跳过（节省 API 成本）
3. 已变化 → `upsert` 更新（覆盖旧向量）

---

### 流程二：知识图谱构建

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Chunk      │ ──> │  概念提取   │ ──> │  概念归一化 │ ──> │  关系抽取   │ ──> │  NetworkX   │
│             │     │  (LLM)      │     │  (相似度)   │     │  (LLM)      │     │   概念图    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**详细步骤**：

#### Step 1: Chunk → 概念
```python
# src/knowledgeGraph/concepts.py: documents_to_concepts()
from src.knowledgeGraph.concepts_extractor import extract_concepts_from_text

concepts = extract_concepts_from_text(f"<文本标题>{title}</文本标题>\n<文本内容>{content}</文本内容>")
# 返回：["概念 1", "概念 2", ...] 最多 10 个
```

**提取规则**：
- 只提取名词/名词短语
- 优先：技术术语、理论、政策、机构、产品、事件
- 删除：普通描述词、时间地点、泛化词语

#### Step 2: 概念归一化
```python
# src/knowledgeGraph/concepts.py: normalized_concept()
# 内部归一（文档内）
concept_pairs = find_similar_concept_pairs(concepts, threshold=0.2)
connected_components = build_graph_and_find_components(concept_pairs)
normalized = llm_merge_clusters(connected_components)

# 外部归一（跨文档）
similar_concepts = chroma_search(query=concept, threshold=0.2, top_k=1)
if similar_concepts:
    normalized = llm_judge_same_entity(concept, similar_concepts[0])
```

**归一化目的**：避免同义词重复（如"AI"和"人工智能"合并为一个节点）。

#### Step 3: 概念入库
```python
# src/vector_store/save_dao.py: save_concept()
save_concept(ConceptDTO(
    id=hash(normalized_concept),
    embedding=get_embedding(normalized_concept),
    normalized_concept=normalized_concept
))
```

#### Step 4: 关系抽取
```python
# src/knowledgeGraph/concepts_relation.py: extract_relations_from_text()
relations = extract_relations_from_text(text, concepts)
# 返回：[["源概念", "关系类型", "目标概念"], ...]
```

**关系示例**：
```json
[
  ["央行", "实施", "量化宽松"],
  ["量化宽松", "影响", "经济"]
]
```

#### Step 5: 构建图
```python
# src/graph.py: build_graph()
import networkx as nx

G = nx.MultiDiGraph()  # 支持多边（同一对概念可能有多种关系）

# 添加文档节点
G.add_node(doc_id, type="document", title=doc_title)

# 添加概念节点
G.add_node(concept_id, type="concept")

# 添加边：文档 → 概念（mentions 关系）
G.add_edge(doc_id, concept_id, relation="mentions")

# 添加边：概念 → 概念（LLM 抽取的关系）
G.add_edge(source_concept, target_concept, relation=relation_type)
```

#### Step 6: 可视化
```python
# src/graph.py: graph_png()
from src.util.visualizeGraph import visualize_graph

visualize_graph(G, "graphPNG/", "概念图")
# 输出：PNG 图片
```

---

## 课程进度

### 主课程（7 课，已完成 6 课）

| 课次 | 主题 | 状态 | 核心功能 | 推送日期 |
|------|------|------|---------|----------|
| Lesson 01 | 5 分钟跑起来 | ✅ 完成 | 项目骨架 + Hello World 搜索 | 2026-07-19 |
| Lesson 02 | 知识单元提取 | ✅ 完成 | LLM 提取 + 向量化 | 2026-07-26 |
| Lesson 02.1 | 生产级增强 | ✅ 完成 | Pydantic 验证 + 内容哈希检测 | 2026-07-26 |
| Lesson 03 | 向量搜索入门 | ✅ 完成 | Chroma 搜索 + 相似度计算 | 2026-08-03 |
| Lesson 04 | 概念图构建 | ✅ 完成 | NetworkX + 概念归一化 | 2026-08-05 (提前) |
| Lesson 05 | Insight 记录 | ✅ 完成 | `insight.py` + SQLite + ChromaDB | 2026-08-11 (补) |
| Lesson 06 | 反馈闭环 | ✅ 完成 | `feedback.py` + 数据迭代 | 2026-08-23 |
| Lesson 07 | 自动化部署 | ⏳ 待推送 | Cron + MCP 封装 + 验收 | 待定 |

### 补充课程（额外材料）

| 课次 | 主题 | 状态 | 说明 |
|------|------|------|------|
| Extra 01 | SQLite 基础与实战 | ✅ 完成 | 生产表（8 张）vs 教程简化表 |
| Extra 02 | 反馈闭环与数据完整性 | ✅ 完成 | del/copy 逻辑实现指南 |
| Extra 03 | Embedding、向量、图深度解析 | ✅ 完成 | 专业共识 vs 代码现状对照 |

---

## 项目结构

```
neural-garden/
├── README.md
├── requirements.txt
├── config.yaml                 # 配置定义（API Key 从 .env 读取）
├── .env                        # 环境变量（敏感信息，勿提交）
├── .env.example                # 环境变量模板
├── src/
│   ├── __init__.py
│   ├── indexer.py              # 知识索引器（增量更新）
│   ├── search.py               # 向量搜索入口（支持 knowledge + insight 联合搜索）
│   ├── graph.py                # 知识图谱构建
│   ├── insight.py              # Insight 记录模块（CLI 入口）
│   ├── feedback.py             # 反馈闭环模块（骨架）
│   ├── similarity.py           # 相似度计算工具
│   ├── get_chroma_collection.py # Chroma 集合获取工具
│   ├── get_sqlite_connection.py # SQLite 连接工具
│   ├── config/
│   │   ├── config.py           # 配置加载
│   │   └── logging_config.py   # 日志配置
│   ├── document/
│   │   └── splitter.py         # Markdown 分块工具（按 title + token）
│   ├── embedding/
│   │   └── getEmbedding.py     # Embedding API 调用
│   ├── knowledge/
│   │   ├── knowledgeUnit.py    # 知识单元提取
│   │   └── knowledge_extractor.py
│   ├── knowledgeGraph/
│   │   ├── concepts.py         # 概念提取与归一化
│   │   ├── concepts_extractor.py
│   │   ├── concepts_relation.py # 关系抽取
│   │   └── knowledge_graph.py  # 图构建主逻辑
│   ├── repository/             # SQLite DAO 层（8 张表）
│   │   ├── create_table.py     # 表初始化
│   │   ├── documents.py        # 文档 CRUD
│   │   ├── document_chunks.py  # 分块 CRUD（含级联删除）
│   │   ├── document_chunk_knowledge_units.py
│   │   ├── document_chunk_concepts.py
│   │   ├── concept_relations.py
│   │   ├── relation_evidence.py
│   │   └── insights.py         # Insight CRUD（新增）
│   ├── vector_store/
│   │   ├── save_dao.py         # 向量存储保存（knowledge/concept/insight）
│   │   ├── query_dao.py        # 向量存储查询
│   │   ├── delete_dao.py       # 向量存储删除
│   │   └── reset.py            # 清空向量库
│   └── util/
│       ├── callDashscopellm.py # LLM 调用封装
│       ├── getHashValue.py     # 哈希工具
│       ├── graphStats.py       # 图统计信息
│       ├── visualizeGraph.py   # 图可视化
│       └── llmException.py     # LLM 异常处理
├── data/
│   ├── pilot/                  # 原始知识笔记（Markdown）
│   └── chroma/                 # ChromaDB 向量存储（自动生成）
├── tests/
│   ├── test_core_modules.py    # 核心模块测试（splitter/hash/similarity）
│   ├── test_repository.py      # repository 层测试（12 个测试）
│   ├── test_insights.py        # Insight 模块测试（6 个测试）
│   └── test_vector_store.py    # ChromaDB 测试（8 个测试）
└── tutorials/                  # 教程文档（不提交到仓库）
```

**架构说明**：
- **SQLite**（8 张生产表）：documents/document_chunks/document_chunk_knowledge_units/document_chunk_concepts/concept_relations/relation_evidence/insights
- **ChromaDB**（3 个 collection）：knowledge/concept/insight
- **NetworkX**（内存图）：MultiDiGraph（支持多边）
- **双层存储**：SQLite（元数据/关系/溯源）+ ChromaDB（Embedding/相似度检索）

---

## 配置说明

### config.yaml

```yaml
dashscope:
  api_key: ${API_KEY}           # 从 .env 读取
  embedding_model: "text-embedding-v1"
  llm_model: "qwen3.5-plus"

chroma:
  persist_directory: "data/chroma"
  knowledge_table_name: "knowledge"
  concept_table_name: "concept"
  insight_table_name: "insight"    # 新增

pilot_dataset:
  path: "data/pilot"

sqlite:
  sqlite_directory: "data/sqlite/neural.db"
```

### .env

```bash
API_KEY=sk-your-api-key-here
```

---

## 核心模块 API

### src/embedding/getEmbedding.py

```python
from src.embedding.getEmbedding import get_embedding

vector = get_embedding("你的文本")
# 返回：list[float] (768 维)
```

### src/indexer.py

```python
# 单文件索引
index_file(file_path)

# 批量索引
index_directory(dir_path)

# 命令行
python -m src.indexer           # 增量更新
python -m src.indexer --resetDB # 清空后重新索引
```

### src/search.py

```bash
python -m src.search "<查询内容>" [top_k]
```

**联合搜索**：同时搜索 knowledge collection 和 insight collection，返回综合结果。

### src/graph.py

```python
# 构建概念图
from src.graph import build_graph, graph_png, log_stats

G = build_graph()
graph_png(G)  # 输出 PNG
log_stats(G)  # 打印统计信息
```

### src/insight.py

```python
# CLI 方式创建 Insight
python src/insight.py \
  --title "一句话总结" \
  --content "洞察内容" \
  --action-items '["行动 1", "行动 2"]' \
  --relation-concepts '["概念 1", "概念 2"]' \
  --source-title "来源文档标题"
```

**relation_concepts 优先级**：用户输入 > chunk 概念 > document 概念 > LLM 自动提取

---

## 技术栈

| 组件 | 技术 |
|------|------|
| **关系存储** | SQLite（8 张生产表：documents/document_chunks/.../insights） |
| **向量存储** | ChromaDB（SQLite + HNSW 索引，3 个 collection：knowledge/concept/insight） |
| **图存储** | NetworkX（内存 MultiDiGraph） |
| **Embedding** | DashScope text-embedding-v1（768 维） |
| **LLM** | DashScope qwen3.5-plus（知识提取、概念提取、关系抽取） |
| **分块工具** | LangChain + tiktoken（按 title + token 递归分割） |
| **配置管理** | PyYAML + python-dotenv |
| **日志系统** | Python logging（滚动文件 + 分级控制） |
| **语言** | Python 3.9+ |

---

## 最终架构

### V1：纯向量检索（Lesson 01-03）

```
Markdown → Chunk → KnowledgeUnit → Embedding → ChromaDB → Search
```

### V2：概念图增强（Lesson 04-05）

```
Markdown → Chunk → KnowledgeUnit + Concept → ChromaDB + NetworkX → Graph
                                                    ↓
                                             Insight (SQLite + ChromaDB)
```

### V3：SQLite + ChromaDB 双层架构（Lesson 06-07 + Insight）

```
                    ┌──────────────┐
Markdown → Chunk →  │   SQLite     │ → 元数据/关系/溯源
                    │  (8 张表)     │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │  ChromaDB    │ → Embedding/相似度检索
                    │  (3 个 Collection) │
                    └──────────────┘
```

**双层架构优势**：
- **SQLite 擅长**：`SELECT * FROM concepts WHERE category = '货币政策'`（精确查询）
- **ChromaDB 擅长**：「找和『负利率』语义相似的概念」（模糊匹配）
- **合并结果**：结构化查询 + 相似度检索 → 完整答案

---

## 核心特性

### 增量更新

- **基于文件名的稳定 ID**：同一文件修改时 ID 不变，支持更新而非新增
- **内容哈希检测**：通过 `content_hash` 判断内容是否变化，未变化则跳过
- **节省成本**：避免重复调用 LLM 和 Embedding API
- **三种状态**：skip（未变化）/ del（删除）/ copy（复制）/ new（新增）

### 生产级设计

- **Pydantic 验证**：LLM 输出经过严格验证（字段类型、长度、有效性）
- **防幻觉 Prompt**：明确要求"不得根据常识补充文档没有的信息"
- **日志系统**：生产级日志配置（控制台 + 滚动文件 + 错误日志）
- **命令行参数**：支持 `--resetDB` 按需清空向量库
- **重试机制**：API 调用失败自动重试
- **数据完整性**：文档更新时自动清理旧数据，相同内容直接复制 ID（避免重复处理）
- **Insight 生命周期**：文档删除时，关联的 Insight 标记为 offline（不删除）

### 知识图谱

- **概念归一化**：相似度检索 + LLM 判断，合并同义词
- **关系抽取**：LLM 从文本中提取有向关系
- **可视化**：自动生成概念图 PNG
- **统计信息**：节点数、边数、连通分量等

### 测试覆盖

- **41 个单元测试**：repository 层（12 个）+ insights 模块（6 个）+ vector_store（8 个）+ core modules（16 个）
- **内存数据库测试**：使用 SQLite :memory: 和临时 ChromaDB 目录
- **CI 就绪**：`pytest tests/ -v` 一键运行

---

## 教程

本项目教程按周推送，每周一课：

| 课次 | 主题 | 推送日期 | 状态 |
|------|------|---------|------|
| Lesson 01 | 5 分钟跑起来 | 2026-07-19 | ✅ 完成 |
| Lesson 02 | 知识单元提取 | 2026-07-26 | ✅ 完成 |
| Lesson 03 | 向量搜索入门 | 2026-08-03 | ✅ 完成 |
| Lesson 04 | 概念图构建 | 2026-08-05 | ✅ 完成 |
| Lesson 05 | Insight 记录 | 2026-08-11 | ✅ 完成 |
| Lesson 06 | 反馈闭环 | 2026-08-23 | ✅ 完成 |
| Lesson 07 | 自动化部署 | 待定 | ⏳ 待推送 |

---

## 常见问题

### Q: API Key 从哪里获取？

访问 [DashScope 控制台](https://dashscope.console.aliyun.com/apiKey) 创建 API Key。

### Q: 向量存储在哪里？

默认存储在 `data/chroma/` 目录，可在 `config.yaml` 中修改 `chroma.persist_directory`。

### Q: 如何更换 Embedding 模型？

修改 `config.yaml` 中的 `dashscope.embedding_model`，目前支持：
- `text-embedding-v1`（默认）
- `text-embedding-v2`
- 其他 DashScope 支持的模型

### Q: 增量更新如何工作？

1. 首次运行：全量索引，记录每个文件的 `content_hash`
2. 后续运行：比较文件名哈希 ID 对应的 `content_hash`
3. 相同 → 跳过；不同 → 更新

### Q: 概念归一化如何工作？

1. 新概念提取后，在概念库中检索相似概念（余弦相似度阈值）
2. 如果有相似概念，用 LLM 判断是否指向同一实体
3. 如果是 → 使用已有概念名；否 → 创建新概念

### Q: Insight 如何工作？

1. 用户创建 Insight（CLI 或 API）
2. 存入 SQLite（结构化数据）+ ChromaDB（向量检索）
3. 搜索时联合查询 knowledge 和 insight 两个 collection
4. 文档删除时，关联的 Insight 标记为 offline（不删除，保留用户数据）

---

## 许可证

MIT License

---

*Project by 俞紫峰 · 2026*
