# Neural Garden 待办课程分析

**分析时间**：2026-08-24  
**代码版本**：`ddb63c4`

---

## 一、教程 vs 项目 对比

### 主课程（7 课）

| 课次 | 教程文件 | 项目实现 | 状态 | 差距分析 |
|------|---------|---------|------|---------|
| Lesson 01 | ✅ lesson-01.md | ✅ 项目骨架 + 搜索 | 完成 | - |
| Lesson 02 | ✅ lesson-02.md | ✅ indexer.py + knowledgeUnit | 完成 | - |
| Lesson 02.1 | ✅ lesson-02.1.md | ✅ Pydantic 验证 + 增量更新 | 完成 | - |
| Lesson 03 | ✅ lesson-03.md | ✅ search.py | 完成 | - |
| Lesson 04 | ✅ lesson-04.md | ✅ graph.py + concepts.py | 完成 | - |
| Lesson 05 | ✅ lesson-05-insight-record.md | ✅ insight.py + repository/insights.py | 完成 | - |
| Lesson 06 | ✅ lesson-06.md | ✅ indexer.py (del/copy 逻辑) | 完成 | - |
| Lesson 07 | ✅ lesson-07-comprehensive.md | ⚠️ 部分实现 | **待完善** | 见下方分析 |

### 补充课程

| 课次 | 教程文件 | 项目实现 | 状态 |
|------|---------|---------|------|
| Extra 01 | - | ✅ repository 层（8 张表） | 完成 |
| Extra 02 | ✅ lesson-07-feedback-integrity.md | ✅ indexer.py del/copy 逻辑 | 完成 |
| Extra 03 | ✅ embedding-vector-graph-deep-dive-v2.md | ✅ 代码实现 | 完成 |

---

## 二、Lesson 07 待完善内容

### 当前实现状态

| 功能 | 教程要求 | 项目实现 | 差距 |
|------|---------|---------|------|
| **Insight 模块** | SQLite + ChromaDB 双存储 | ✅ 完整实现 | - |
| **source_status offline** | SQLite + ChromaDB 同步更新 | ✅ 完整实现 | - |
| **联合搜索** | knowledge + insight | ✅ 完整实现 | - |
| **Cron 定时任务** | 每日增量索引 | ❌ 未实现 | 需创建 cron 配置 |
| **MCP 封装** | 封装为 MCP 工具 | ❌ 未实现 | 需创建 MCP server |
| **验收标准** | 性能测试 + 功能验收 | ❌ 未实现 | 需创建验收脚本 |

---

## 三、待办课程清单

### P0：Lesson 07 完善（必须完成）

#### 1. Cron 定时任务配置

**教程要求**：
- 每日自动索引新文档
- 失败重试机制
- 日志记录

**待实现**：
```bash
# 创建 cron 配置
crontab -e
# 添加：0 2 * * * cd /path/to/neural-garden && python -m src.indexer >> logs/indexer.log 2>&1
```

**验收标准**：
- [ ] 每日 02:00 自动运行
- [ ] 失败时发送通知（邮件/飞书）
- [ ] 日志保留 30 天

---

#### 2. MCP 封装

**教程要求**：
- 封装为 MCP 工具
- 支持远程调用
- 权限控制

**待实现**：
```python
# src/mcp_server.py
from mcp import Server

server = Server("neural-garden")

@server.tool("search")
def search_knowledge(query: str, top_k: int = 5) -> list:
    """搜索知识库"""
    pass

@server.tool("create_insight")
def create_insight(title: str, content: str, ...) -> int:
    """创建 Insight"""
    pass
```

**验收标准**：
- [ ] MCP server 可启动
- [ ] 支持 search/create_insight 工具
- [ ] 权限验证（API Key）

---

#### 3. 验收脚本

**教程要求**：
- 性能测试（搜索延迟 < 2 秒）
- 功能验收（所有 CRUD 操作）
- 数据完整性测试

**待实现**：
```bash
# tests/acceptance_test.py
def test_search_latency():
    start = time.time()
    search("测试查询")
    elapsed = time.time() - start
    assert elapsed < 2.0

def test_data_integrity():
    # 删除文档后，验证关联数据清理
    pass
```

**验收标准**：
- [ ] 所有测试通过
- [ ] 性能达标
- [ ] 生成验收报告

---

### P1：进阶课程（可选）

#### 4. 混合检索优化

**当前状态**：
- 只支持向量相似度搜索
- 无 rerank 机制

**待实现**：
```python
# src/search.py
def hybrid_search(query, top_k=10):
    # 1. 向量检索（召回 50 条）
    candidates = vector_search(query, top_k=50)
    
    # 2. Rerank（重排序）
    reranked = rerank(query, candidates, top_k=top_k)
    
    return reranked
```

**教程价值**：
- 讲解混合检索原理
- 对比纯向量检索 vs 混合检索
- 实战 rerank 模型集成

---

#### 5. 图遍历搜索

**当前状态**：
- 只支持向量搜索
- 概念图仅用于可视化

**待实现**：
```python
# src/graph_search.py
def graph_search(start_concept, relation_type, depth=2):
    """
    从起始概念出发，沿指定关系类型遍历图
    """
    G = build_graph()
    paths = nx.single_source_shortest_path(G, start_concept, cutoff=depth)
    return filter_by_relation(paths, relation_type)
```

**教程价值**：
- 讲解图遍历算法
- 对比向量搜索 vs 图遍历
- 实战 NetworkX 高级用法

---

#### 6. 用户反馈闭环

**当前状态**：
- `feedback.py` 只有骨架
- 无实际数据收集

**待实现**：
```python
# src/feedback.py
def record_search_feedback(query, results, clicked_index):
    """记录搜索反馈"""
    save_to_sqlite(SearchFeedback(
        query=query,
        results=results,
        clicked_index=clicked_index,
        dwell_time_seconds=30
    ))

def weekly_optimization():
    """每周优化"""
    # 1. 分析高频查询
    # 2. 加强相关概念索引
    # 3. 清理低质量知识单元
    pass
```

**教程价值**：
- 讲解反馈闭环原理
- 实战数据驱动优化
- 避免"构建完就结束"

---

### P2：扩展课程（长期）

#### 7. 多模态支持

**待实现**：
- 图片 OCR 提取文字
- 表格解析
- PDF 支持

**教程价值**：
- 扩展知识来源
- 实战多模态处理

---

#### 8. 分布式部署

**待实现**：
- Docker 容器化
- Kubernetes 部署
- 负载均衡

**教程价值**：
- 生产环境部署
-  scalability 设计

---

## 四、总结

### 已完成（6/7 主课 + 3 补充课）

- ✅ Lesson 01-06
- ✅ Extra 01-03
- ✅ Insight 模块完整实现
- ✅ 41 个单元测试

### 待完成（Lesson 07 完善）

- ⏳ Cron 定时任务
- ⏳ MCP 封装
- ⏳ 验收脚本

### 进阶方向（可选）

- ⏳ 混合检索优化
- ⏳ 图遍历搜索
- ⏳ 用户反馈闭环

---

**建议优先级**：
1. **P0**：完成 Lesson 07（Cron + MCP + 验收）
2. **P1**：混合检索优化（提升搜索质量）
3. **P2**：根据用户需求决定是否实现

*Analysis by 阿紫 · 2026-08-24*
