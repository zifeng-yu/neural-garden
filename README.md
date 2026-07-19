# Neural Garden

> 个人认知系统 · 记录 · 连接 · 调用

**Neural Garden** 是一个帮助你构建个人知识系统的工具。它基于 RAG（检索增强生成）和知识图谱技术，让你的知识不再是孤立的笔记，而是可以相互连接、智能调用的认知网络。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
# 编辑 config.yaml，填入你的 DashScope API Key

# 3. 运行搜索
python -m src.search "什么是 Neural Garden"
```

## 教程

本项目的教程按周推送，每周一课：

| 课次 | 主题 | 推送日期 |
|------|------|---------|
| Lesson 01 | 5 分钟跑起来 | 2026-07-19 |
| Lesson 02 | 知识单元提取 | 2026-07-26 |
| Lesson 03 | 向量搜索入门 | 2026-08-03 |
| Lesson 04 | 概念图构建 | 2026-08-10 |
| Lesson 05 | Insight 记录 | 2026-08-17 |
| Lesson 06 | 反馈闭环 | 2026-08-24 |
| Lesson 07 | 自动化部署 | 2026-08-31 |

教程文档位于 `tutorials/` 目录。

## 项目结构

```
neural-garden/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── search.py
├── data/
│   ├── pilot/
│   └── chroma/
└── tests/
```

## 技术栈

- **向量存储**: ChromaDB
- **Embedding**: DashScope text-embedding-v3
- **LLM**: DashScope qwen-plus
- **语言**: Python 3.9+

## 许可证

MIT License

---

*Project by 俞紫峰 · 2026*
