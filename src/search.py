"""
Neural Garden - 搜索入口
Lesson 01: Hello World 骨架版本
"""

import chromadb
from chromadb.config import Settings
import os

def search(query: str, top_k: int = 3):
    """
    向量搜索入口
    
    Args:
        query: 搜索查询
        top_k: 返回结果数量
    """
    # 1. 初始化 Chroma 客户端
    persist_dir = os.path.join(os.path.dirname(__file__), "../data/chroma")
    client = chromadb.Client(Settings(persist_directory=persist_dir))
    
    # 2. 获取集合
    collection = client.get_or_create_collection("knowledge")
    
    # 3. 查询（骨架版本，后续课程实现完整 Embedding 流程）
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # 4. 格式化输出
    print_results(results, query)

def print_results(results, query):
    """格式化输出搜索结果"""
    print(f"\n🔍 搜索查询：{query}")
    print("\n📌 匹配结果 (Top 3):")
    print("─" * 40)
    
    if results['documents'] and results['documents'][0]:
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            source = meta.get('source', 'unknown')
            print(f"[{i}] {source}")
            print(f"    {doc[:100]}...")
            print()
    else:
        print("暂无结果，请先运行索引命令添加知识。")
    
    print("─" * 40)
    print("✅ 搜索完成")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        search(query)
    else:
        print("用法：python -m src.search <查询内容>")
