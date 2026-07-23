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
python -m src.indexer
```

### 4. 搜索知识

```bash
python -m src.search "什么是 Neural Garden"
```

---

## 项目结构

```
neural-garden/
├── README.md
├── requirements.txt
├── config.py                 # 配置加载（支持 ${ENV} 语法）
├── config.yaml               # 配置定义（API Key 从 .env 读取）
├── .env                      # 环境变量（敏感信息，勿提交）
├── .env.example              # 环境变量模板
├── src/
│   ├── __init__.py
│   ├── indexer.py            # 知识索引器
│   ├── search.py             # 搜索入口
│   └── embedding/
│       └── getEmbedding.py   # 嵌入模型调用（DashScope）
├── data/
│   ├── pilot/                # 原始知识笔记（Markdown）
│   └── chroma/               # ChromaDB 向量存储（自动生成）
└── tests/
```

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
  table_name: "knowledge"

pilot_dataset:
  path: "data/pilot"
```

### .env

```bash
API_KEY=sk-your-api-key-here
```

---

## 核心模块

### src/embedding/getEmbedding.py

封装 DashScope Embedding API，提供统一的向量生成接口：

```python
from src.embedding.getEmbedding import get_embedding

vector = get_embedding("你的文本")
```

### src/indexer.py

知识索引器，支持：
- 单文件导入：`index_file(file_path, persist_dir)`
- 批量导入：`index_directory(dir_path, persist_dir)`

### src/search.py

向量搜索入口，支持语义检索：

```bash
python -m src.search "<查询内容>" [top_k]
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 向量存储 | ChromaDB |
| Embedding | DashScope text-embedding-v1 |
| LLM | DashScope qwen3.5-plus |
| 配置管理 | PyYAML + python-dotenv |
| 语言 | Python 3.9+ |

---

## 教程

本项目教程按周推送，每周一课：

| 课次 | 主题 | 推送日期 |
|------|------|---------|
| Lesson 01 | 5 分钟跑起来 | 2026-07-19 |
| Lesson 02 | 知识单元提取 | 2026-07-26 |
| Lesson 03 | 向量搜索入门 | 2026-08-03 |
| Lesson 04 | 概念图构建 | 2026-08-10 |
| Lesson 05 | Insight 记录 | 2026-08-17 |
| Lesson 06 | 反馈闭环 | 2026-08-24 |
| Lesson 07 | 自动化部署 | 2026-08-31 |

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

---

## 许可证

MIT License

---

*Project by 俞紫峰 · 2026*
