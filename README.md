# Neural Garden

> 个人认知系统 · 记录 · 连接 · 调用

**Neural Garden** 是一个帮助你构建个人知识系统的工具。它基于 RAG（检索增强生成）和向量检索技术，让你的知识不再是孤立的笔记，而是可以相互连接、智能检索的认知网络。

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

# 清空向量库后重新索引
python -m src.indexer --resetDB
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
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Markdown   │ ──> │  知识单元   │ ──> │  Embedding  │ ──> │  ChromaDB   │
│   文档      │     │   提取      │     │   向量化    │     │   入库      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**详细步骤**：

#### Step 1: 读取文档
```python
# src/indexer.py: load_document()
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
```

#### Step 2: 知识单元提取
```python
# src/indexer.py: index_file()
from src.knowledge.knowledgeUnit import extract_knowledge_unit

knowledgeUnit = extract_knowledge_unit(
    content=content,
    source=filename,
    uuid=hash(filename)  # 基于文件名生成稳定 ID
)
```

**知识单元结构**：
- `unit_id`: 唯一标识（文件名哈希）
- `title`: 文档标题
- `keywords`: 关键词列表（3-5 个）
- `summary`: 摘要（200-300 字）
- `content_hash`: 内容哈希（用于增量更新检测）

#### Step 3: 向量化
```python
# src/indexer.py: index_file()
from src.embedding.getEmbedding import get_embedding

embedding = get_embedding(knowledgeUnit.to_embedding_text())
# 返回 768 维向量
```

**Embedding 文本**：`标题 + 摘要 + 关键词` 的组合文本，优化检索效果。

#### Step 4: 入库（增量更新）
```python
# src/indexer.py: index_file()
# 检查是否已存在
existing = collection.get(ids=[uuid], include=["metadatas"])
if existing["ids"] and existing["metadatas"][0].get("content_hash") == content_hash:
    # 内容未变化，跳过
    return

# 新增或更新
collection.upsert(
    ids=[knowledgeUnit.unit_id],
    embeddings=[embedding],
    documents=[knowledgeUnit.to_embedding_text()],
    metadatas=[{
        "source": source,
        "title": knowledgeUnit.title,
        "keywords": json.dumps(knowledgeUnit.keywords),
        "full_content": content,
        "content_hash": content_hash,
    }]
)
```

**增量更新逻辑**：
1. 用文件名哈希作为 ID 查询
2. 比较 `content_hash` 判断内容是否变化
3. 未变化 → 跳过（节省 API 成本）
4. 已变化 → `upsert` 更新（覆盖旧向量）

---

### 流程二：知识图谱构建

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  文档列表   │ ──> │  概念提取   │ ──> │  概念归一化 │ ──> │  关系抽取   │ ──> │  NetworkX   │
│             │     │  (LLM)      │     │  (相似度)   │     │  (LLM)      │     │   概念图    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**详细步骤**：

#### Step 1: 文档 → 概念
```python
# src/knowledgeGraph/knowledge_graph.py: get_documents_to_concepts()
from src.knowledgeGraph.graph_extractor import extract_concepts_from_text

concepts = extract_concepts_from_text(f"<文本标题>{title}</文本标题>\n<文本内容>{content}</文本内容>")
# 返回：["概念 1", "概念 2", ...] 最多 10 个
```

**提取规则**：
- 只提取名词/名词短语
- 优先：技术术语、理论、政策、机构、产品、事件
- 删除：普通描述词、时间地点、泛化词语
- 同义概念只保留一个标准名称

#### Step 2: 概念归一化
```python
# src/knowledgeGraph/knowledge_graph.py: resolve_concept()
from src.vector_store.query_dao import search_by_threshold_concept

# 查询相似概念（余弦相似度阈值）
similar_concepts = search_by_threshold_concept(query=concept)

# 用 LLM 判断最相似的概念
from src.knowledgeGraph.graph_extractor import extract_max_similarity_concept
normalized = extract_max_similarity_concept(concept, [x.conceptName for x in similar_concepts])
# 返回："标准化概念名"
```

**归一化目的**：避免同义词重复（如"AI"和"人工智能"合并为一个节点）。

#### Step 3: 概念入库
```python
# src/knowledgeGraph/knowledge_graph.py: build_concept_graph()
from src.vector_store.save_dao import save_concept, ConceptDO, ConceptMetadata

conceptDO = ConceptDO(
    id=hash(concept_name),
    embedding=get_embedding(concept_name),
    conceptName=concept_name,
    metadata=ConceptMetadata(source=[doc_title])
)
save_concept(conceptDO)
```

#### Step 4: 关系抽取
```python
# src/knowledgeGraph/knowledge_graph.py: build_concept_graph()
from src.knowledgeGraph.graph_extractor import extract_relations_from_text

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
# src/knowledgeGraph/knowledge_graph.py: build_concept_graph()
import networkx as nx

G = nx.DiGraph()

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

### 主课程（按周推送）

| 课次 | 主题 | 状态 | 核心功能 | 推送日期 |
|------|------|------|---------|----------|
| Lesson 01 | 5 分钟跑起来 | ✅ 完成 | 项目骨架 + Hello World 搜索 | 2026-07-19 |
| Lesson 02 | 知识单元提取 | ✅ 完成 | LLM 提取 + 向量化 + 增量更新 | 2026-07-26 |
| Lesson 02.1 | 生产级增强 | ✅ 完成 | Pydantic 验证 + 内容哈希检测 + --resetDB 参数 | 2026-07-26 |
| Lesson 03 | 向量搜索入门 | ✅ 完成 | Chroma 搜索 + 相似度计算 | 2026-08-03 |
| Lesson 04 | 概念图构建 | ✅ 完成 | NetworkX + 概念归一化 + 重试机制 | 2026-08-05 (提前) |
| Lesson 05 | Insight 记录 | ✅ 完成 | `insight.py` + Markdown 导出 | 2026-08-11 (补) |
| Lesson 06 | 反馈闭环 | ✅ 完成 | `feedback.py` + 数据迭代 | 2026-08-23 |
| Lesson 07 | 自动化部署 | ⏳ 待推送 | Cron + MCP 封装 + 验收 | 2026-09-07 |

### 补充课程（额外材料，不通过 cron 推送）

| 课次 | 主题 | 状态 | 说明 |
|------|------|------|------|
| Extra 01 | SQLite 基础与实战 | ✅ 完成 | 生产表（8 张）vs 教程简化表（3 张）说明 |
| Extra 02 | 反馈闭环与数据完整性 | ✅ 完成 | del/copy/relation 逻辑实现指南 |

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
│   ├── search.py               # 向量搜索入口
│   ├── graph.py                # 知识图谱构建
│   ├── insight.py              # Insight 记录模块（Markdown 导出）
│   ├── feedback.py             # 反馈闭环模块（搜索日志/数据迭代）
│   ├── storage.py              # SQLite 存储模块（简化表：3 张）
│   ├── similarity.py           # 相似度计算工具
│   ├── get_chroma_collection.py # Chroma 集合获取工具
│   ├── get_sqlite_connection.py # SQLite 连接工具
│   ├── config/
│   │   ├── config.py           # 配置加载
│   │   └── logging_config.py   # 日志配置
│   ├── document/
│   │   └── markdown_split.py   # Markdown 分块工具
│   ├── embedding/
│   │   └── getEmbedding.py     # Embedding API 调用
│   ├── knowledge/
│   │   ├── knowledgeUnit.py    # 知识单元提取
│   │   └── concepts.py         # 概念提取与归一化
│   ├── knowledgeGraph/
│   │   ├── knowledge_graph.py  # 图构建主逻辑
│   │   └── graph_extractor.py  # 概念/关系抽取（LLM）
│   ├── repository/             # 生产表 DAO 层（8 张表）
│   │   ├── create_table.py     # 表初始化
│   │   ├── documents.py        # 文档 CRUD
│   │   ├── document_chunks.py  # 分块 CRUD（含级联删除）
│   │   ├── document_chunk_knowledge_units.py
│   │   ├── document_chunk_concepts.py
│   │   └── concept_relations.py
│   ├── vector_store/
│   │   ├── save_dao.py         # 向量存储保存
│   │   ├── query_dao.py        # 向量存储查询
│   │   └── reset.py            # 清空向量库
│   └── util/
│       ├── callDashscopellm.py # LLM 调用封装
│       ├── getHashValue.py     # 哈希工具
│       ├── graphStats.py       # 图统计信息
│       ├── visualizeGraph.py   # 图可视化
│       └── retryUtil.py        # 重试机制
├── data/
│   ├── pilot/                  # 原始知识笔记（Markdown）
│   └── chroma/                 # ChromaDB 向量存储（自动生成）
└── tests/
    ├── test_concept_normalization.py  # 概念归一化测试
    └── test_retry.py                  # 重试机制测试
```

**架构说明**：
- **简化表**（`src/storage.py`）：3 张表（concepts/insights/search_logs），用于教程学习
- **生产表**（`src/repository/*`）：8 张表（documents/document_chunks/...），用于生产环境
- **双层存储**：SQLite（元数据/关系）+ ChromaDB（Embedding/相似度检索）

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

pilot_dataset:
  path: "data/pilot"
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
index_file(file_path, persist_dir)

# 批量索引
index_directory(dir_path, persist_dir)

# 命令行
python -m src.indexer           # 增量更新
python -m src.indexer --resetDB # 清空后重新索引
```

### src/search.py

```bash
python -m src.search "<查询内容>" [top_k]
```

### src/graph.py

```python
# 构建概念图
from src.graph import build_graph, graph_png, log_stats

G = build_graph(pilot_dir)
graph_png(G)  # 输出 PNG
log_stats(G)  # 打印统计信息
```

### src/insight.py

```python
from src.insight import create_insight

insight = create_insight(
    title="一句话总结",
    trigger_content="触发内容",
    source="来源",
    content="洞察内容",
    related_concepts=["概念 1", "概念 2"],
    action_items=["行动 1", "行动 2"]
)

print(insight.to_markdown())
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| **关系存储** | SQLite（8 张生产表：documents/document_chunks/...） |
| **向量存储** | ChromaDB（SQLite + HNSW 索引，2 个 collection） |
| **图存储** | NetworkX（内存 DiGraph） |
| **Embedding** | DashScope text-embedding-v1（768 维） |
| **LLM** | DashScope qwen3.5-plus（知识提取、关系抽取） |
| **配置管理** | PyYAML + python-dotenv |
| **日志系统** | Python logging（滚动文件 + 分级控制） |
| **语言** | Python 3.9+ |

**架构说明**：
- **教程简化表**（`src/storage.py`）：3 张表（concepts/insights/search_logs），用于学习 SQLite 基础
- **生产表**（`src/repository/*`）：8 张表，支持增量更新（del/copy/new）和概念关系追溯
- **双层架构**：SQLite（元数据/关系/溯源）+ ChromaDB（Embedding/相似度检索）

---

## 最终架构

### V1：纯向量检索（Lesson 01-03）

```
Markdown → KnowledgeUnit → Embedding → ChromaDB → Search
```

### V2：概念图增强（Lesson 04-05）

```
Markdown → KnowledgeUnit + Concept → ChromaDB + NetworkX → Graph
                                           ↓
                                      Insight (Markdown)
```

### V3：SQLite + ChromaDB 双层架构（Lesson 06-07 + 补充课程）

```
                    ┌──────────────┐
Markdown → Chunk →  │   SQLite     │ → 元数据/关系/溯源
                    │  (8 张表)     │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │  ChromaDB    │ → Embedding/相似度检索
                    │  (2 个 Collection) │
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

### 生产级设计

- **Pydantic 验证**：LLM 输出经过严格验证（字段类型、长度、有效性）
- **防幻觉 Prompt**：明确要求"不得根据常识补充文档没有的信息"
- **日志系统**：生产级日志配置（控制台 + 滚动文件 + 错误日志）
- **命令行参数**：支持 `--resetDB` 按需清空向量库
- **重试机制**：API 调用失败自动重试（`src/util/retryUtil.py`）
- **增量更新**：基于 content_hash 检测文件变化，支持 del/copy/new 三种状态
- **数据完整性**：文档更新时自动清理旧数据，相同内容直接复制 ID（避免重复处理）

### 知识图谱

- **概念归一化**：相似度检索 + LLM 判断，合并同义词
- **关系抽取**：LLM 从文本中提取有向关系
- **可视化**：自动生成概念图 PNG
- **统计信息**：节点数、边数、连通分量等

---

## 教程

本项目教程按周推送，每周一课：

| 课次 | 主题 | 推送日期 |
|------|------|---------|
| Lesson 01 | 5 分钟跑起来 | 2026-07-19 |
| Lesson 02 | 知识单元提取 | 2026-07-26 |
| Lesson 03 | 向量搜索入门 | 2026-08-03 |
| Lesson 04 | 概念图构建 | 2026-08-09 |
| Lesson 05 | Insight 记录 | 2026-08-09 |
| Lesson 06 | 反馈闭环 | 2026-08-16 |
| Lesson 07 | 自动化部署 | 2026-08-23 |

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

---

## 许可证

MIT License

---

*Project by 俞紫峰 · 2026*
