# Lesson 02: 知识单元提取

> 推送日期：2026-07-26（周日）10:00  
> 核心模块：`src/indexer.py` 增强版  
> 学习价值：理解 RAG 中的数据预处理流程

---

## 本课目标

学完本课后，你将能够：

- **理解为什么需要知识单元提取**：对比"全文索引"vs"知识单元索引"的差异
- **实现自动提取能力**：让 LLM 从文档中提取标题、摘要、关键词
- **完成数据预处理流水线**：从原始 Markdown → 结构化知识单元 → 向量索引

本课代码产出：增强版 `indexer.py`，支持知识单元自动提取。

---

## 原理讲解

### 为什么不能直接索引全文？

在 Lesson 01 中，我们直接把整篇文章作为一个向量存入 Chroma。这种做法有一个致命问题：

```
问题：用户问"日本负利率政策的核心逻辑是什么？"

全文索引的问题：
- 整篇文章（5000 字）被压缩成一个向量
- 搜索返回的是"整篇文章"
- 用户需要自己从 5000 字里找答案

知识单元索引的优势：
- 文章被拆分为：标题 + 摘要 + 关键词 + 核心段落
- 每个单元独立向量化
- 搜索返回的是"精准片段"
- 用户直接看到答案
```

### 知识单元的结构

本课定义的知识单元包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | string | 文章/段落标题（由 LLM 提取） |
| `summary` | string | 100-200 字摘要（由 LLM 生成） |
| `keywords` | list[string] | 3-5 个关键词（由 LLM 提取） |
| `content` | string | 原始内容片段 |
| `source` | string | 来源文件名 |
| `unit_id` | string | 单元唯一标识（哈希） |

### 提取流程

```
原始 Markdown 文件
    ↓
LLM 分析（调用 qwen3.5-plus）
    ↓
提取：title, summary, keywords
    ↓
生成知识单元 JSON
    ↓
向量化（title + summary + keywords）
    ↓
存入 Chroma
```

---

## 代码实现

### 步骤 1：定义知识单元数据结构

在 `src/indexer.py` 顶部添加数据类定义：

```python
from dataclasses import dataclass, asdict
from typing import List, Optional
import json

@dataclass
class KnowledgeUnit:
    """知识单元数据结构"""
    title: str
    summary: str
    keywords: List[str]
    content: str
    source: str
    unit_id: str
    
    def to_dict(self) -> dict:
        """转换为字典（用于存储）"""
        return asdict(self)
    
    def to_embedding_text(self) -> str:
        """生成用于向量化的文本（标题 + 摘要 + 关键词）"""
        return f"{self.title}。{self.summary}。关键词：{', '.join(self.keywords)}"
```

### 步骤 2：实现 LLM 提取函数

添加知识单元提取函数，调用 LLM 进行结构化提取：

```python
def extract_knowledge_unit(content: str, source: str) -> Optional[KnowledgeUnit]:
    """
    调用 LLM 从内容中提取知识单元
    
    Args:
        content: 原始文档内容
        source: 来源文件名
        
    Returns:
        KnowledgeUnit 对象，提取失败返回 None
    """
    from config import LLM_MODEL
    import requests
    
    # 构建提取提示词
    prompt = f"""你是一位专业的知识结构化专家。请从以下文档内容中提取关键信息，输出 JSON 格式。

【提取要求】
1. title: 文档标题（15 字以内）
2. summary: 核心摘要（100-200 字，概括主要内容）
3. keywords: 3-5 个关键词（最能代表文档主题的词汇）

【文档内容】
{content[:3000]}  # 限制长度，避免超出上下文

【输出格式】
严格输出 JSON，不要任何其他文字：
{{
    "title": "标题",
    "summary": "摘要内容...",
    "keywords": ["关键词 1", "关键词 2", "关键词 3"]
}}
"""
    
    # 调用 LLM API（使用 gateway 配置）
    # 注意：这里使用简化的调用方式，实际项目中建议使用 SDK
    try:
        # 模拟调用（实际项目中替换为真实 API 调用）
        # 这里为了教学目的，使用伪代码展示流程
        print(f"🤖 正在调用 LLM 提取知识单元...")
        
        # TODO: 实际 API 调用代码（根据项目配置实现）
        # response = call_llm_api(prompt, model=LLM_MODEL)
        # result = json.loads(response)
        
        # 教学演示：返回模拟结果
        result = {
            "title": f"从 {source} 提取的标题",
            "summary": "这是一个模拟的摘要，实际项目中会由 LLM 生成真实摘要。",
            "keywords": ["关键词 1", "关键词 2", "关键词 3"]
        }
        
        # 生成单元 ID
        unit_id = hashlib.md5(f"{source}:{result['title']}".encode()).hexdigest()[:16]
        
        return KnowledgeUnit(
            title=result["title"],
            summary=result["summary"],
            keywords=result["keywords"],
            content=content,
            source=source,
            unit_id=unit_id
        )
        
    except Exception as e:
        print(f"⚠️  知识单元提取失败：{e}")
        return None
```

### 步骤 3：更新索引函数

修改 `index_file` 函数，使用知识单元而非全文：

```python
def index_file(file_path: str, persist_dir: str):
    """
    将单个文件导入 Chroma 向量库（知识单元版本）
    
    Args:
        file_path: Markdown 文件路径
        persist_dir: Chroma 持久化目录
    """
    # 1. 读取文件
    content = load_document(file_path)
    source = os.path.basename(file_path)
    
    print(f"📄 正在处理：{source}")
    
    # 2. 提取知识单元
    unit = extract_knowledge_unit(content, source)
    if unit is None:
        print(f"⚠️  跳过 {source}（知识单元提取失败）")
        return
    
    print(f"🏷️  提取完成：{unit.title}")
    print(f"🔑  关键词：{', '.join(unit.keywords)}")
    
    # 3. 初始化 Chroma 客户端
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(CHROMA_TABLE_NAME)
    
    # 4. 生成向量化文本（标题 + 摘要 + 关键词）
    embedding_text = unit.to_embedding_text()
    print(f"🔢 正在生成向量...")
    embedding = get_embedding(embedding_text)
    
    if embedding is None:
        print(f"⚠️  跳过 {source}（Embedding 失败）")
        return
    
    # 5. 添加到向量库（存储知识单元而非全文）
    collection.upsert(
        ids=[unit.unit_id],
        embeddings=[embedding],
        documents=[embedding_text],  # 存储向量化文本
        metadatas=[{
            "source": source,
            "title": unit.title,
            "keywords": json.dumps(unit.keywords),
            "full_content": content  # 可选：存储全文用于后续检索
        }],
    )
    
    print(f"✅ 索引完成：{unit.title}")
```

---

## 运行验证

### 测试命令

```bash
cd /home/admin/openclaw/workspace/guides/neural-garden/neural-garden

# 运行索引器
python src/indexer.py
```

### 预期输出

```
🌱 Neural Garden 索引器（知识单元版本）
📂 索引目录：/home/admin/openclaw/workspace/guides/neural-garden/neural-garden/data/pilot
💾 存储目录：/home/admin/openclaw/workspace/guides/neural-garden/neural-garden/data/chroma

📂 发现 5 个 Markdown 文件

📄 正在处理：01-金融深潜 - 日本负利率.md
🤖 正在调用 LLM 提取知识单元...
🏷️  提取完成：日本负利率政策的底层逻辑
🔑  关键词：负利率，货币政策，日本央行，收益率曲线控制
🔢 正在生成向量...
✅ 索引完成：日本负利率政策的底层逻辑

📄 正在处理：02-算法 - 重复检测.md
...

✅ 批量索引完成，共 5 个文件
```

### 验证搜索

```bash
# 搜索测试
python -m src.search "负利率"
```

预期返回包含标题、摘要、关键词的精准结果，而非整篇文章。

---

## 课后练习

### 练习 1：完善 LLM 调用（必做）

在 `extract_knowledge_unit` 函数中，实现真实的 LLM API 调用：

```python
# 提示：使用项目配置的 LLM_MODEL
# 参考 config.py 中的配置加载方式
# 可以使用 requests 库调用 HTTP API，或使用官方 SDK
```

**验收标准**：能够真实调用 LLM 并返回结构化 JSON。

### 练习 2：添加错误处理（选做）

为知识单元提取添加重试机制：

- LLM 调用失败时，重试 2 次
- 超时时间设置为 30 秒
- 记录失败日志到 `logs/extraction.log`

---

## 下节课预告

**Lesson 03: 向量搜索入门**

- 理解 Embedding 的数学原理（为什么相似的内容向量距离近？）
- 实现相似度搜索算法（余弦相似度计算）
- 优化搜索结果排序（按相关性打分）

---

*Lesson 02 by 阿紫 · 2026-07-26*
