"""
Neural Garden - 概念图构建模块
Lesson 04: 概念图构建
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


def extract_concepts_from_text(text: str, llm_client=None) -> List[str]:
    """
    从文本中提取概念
    
    Args:
        text: 输入文本
        llm_client: LLM 客户端（可选）
        
    Returns:
        概念列表
    """
    # 使用 LLM 提取概念
    prompt = f"""从以下文本中提取关键概念（名词短语），输出 JSON 列表格式。

文本：
{text[:2000]}  # 限制长度

输出格式（只输出 JSON 列表）：
["概念 1", "概念 2", "概念 3", ...]
"""
    
    # 调用 LLM（简化示例，实际项目中使用配置的 LLM）
    # response = llm_client.generate(prompt)
    # concepts = json.loads(response)
    
    # 教学演示：返回模拟结果
    concepts = ["概念 A", "概念 B", "概念 C"]
    
    return concepts


def extract_relations_from_text(text: str, concepts: List[str], llm_client=None) -> List[Tuple]:
    """
    从文本中提取概念之间的关系
    
    Args:
        text: 输入文本
        concepts: 概念列表
        llm_client: LLM 客户端（可选）
        
    Returns:
        关系列表 [(概念 1, 关系类型，概念 2), ...]
    """
    prompt = f"""从以下文本中提取概念之间的关系。

已知概念：{concepts}

文本：
{text[:2000]}

输出格式（只输出 JSON 列表）：
[["概念 1", "关系类型", "概念 2"], ...]

关系类型示例：是一种、包括、影响、导致、相关
"""
    
    # 调用 LLM（简化示例）
    # response = llm_client.generate(prompt)
    # relations = json.loads(response)
    
    # 教学演示：返回模拟结果
    relations = [("概念 A", "相关", "概念 B")]
    
    return relations


def build_concept_graph(documents: List[Dict], llm_client=None) -> nx.Graph:
    """
    从文档列表构建概念图
    
    Args:
        documents: 文档列表，每个文档包含 {'title': str, 'content': str}
        llm_client: LLM 客户端（可选）
        
    Returns:
        NetworkX 图对象
    """
    G = nx.Graph()
    
    for doc in documents:
        title = doc.get('title', '未知')
        content = doc.get('content', '')
        full_text = f"{title}\n{content}"
        
        # 提取概念
        concepts = extract_concepts_from_text(full_text, llm_client)
        
        # 添加节点
        for concept in concepts:
            G.add_node(concept, source=title)
        
        # 提取关系并添加边
        relations = extract_relations_from_text(full_text, concepts, llm_client)
        for source, relation, target in relations:
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, relation=relation)
    
    return G


def visualize_graph(G: nx.Graph, output_path: str, title: str = "概念图"):
    """
    可视化概念图
    
    Args:
        G: NetworkX 图对象
        output_path: 输出图片路径
        title: 图标题
    """
    plt.figure(figsize=(12, 10))
    
    # 使用 spring_layout 布局
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # 绘制边标签（关系类型）
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ 概念图已保存至：{output_path}")


def search_by_concept(G: nx.Graph, concept: str, depth: int = 2) -> List[str]:
    """
    按概念搜索关联概念
    
    Args:
        G: NetworkX 图对象
        concept: 起始概念
        depth: 搜索深度（默认 2 层）
        
    Returns:
        关联概念列表
    """
    if concept not in G.nodes:
        logger.warning(f"⚠️  概念 '{concept}' 不在图中")
        return []
    
    # 使用 BFS 搜索
    related_concepts = []
    for node in nx.single_source_shortest_path_length(G, concept, cutoff=depth):
        if node != concept:
            related_concepts.append(node)
    
    return related_concepts


def get_graph_stats(G: nx.Graph) -> Dict:
    """
    获取图的统计信息
    
    Args:
        G: NetworkX 图对象
        
    Returns:
        统计信息字典
    """
    stats = {
        "节点数": G.number_of_nodes(),
        "边数": G.number_of_edges(),
        "密度": nx.density(G),
        "连通分量数": nx.number_connected_components(G),
        "平均度": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
    }
    
    # 找到度最高的节点（最核心的概念）
    if G.number_of_nodes() > 0:
        degree_dict = dict(G.degree())
        top_concepts = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        stats["核心概念 (Top 5)"] = top_concepts
    
    return stats


if __name__ == "__main__":
    # 测试代码
    print("=== Neural Garden 概念图模块测试 ===\n")
    
    # 准备测试数据
    documents = [
        {'title': '日本负利率政策', 'content': '负利率是一种货币政策，由央行实施，影响银行利润和信贷政策'},
        {'title': '量化宽松', 'content': '量化宽松是一种货币政策，包括购买国债，影响央行资产负债表'},
        {'title': '通胀目标', 'content': '通胀目标是货币政策的目标之一，央行通过调整利率实现通胀目标'},
    ]
    
    # 构建概念图
    print("正在构建概念图...")
    G = build_concept_graph(documents)
    
    # 获取统计信息
    stats = get_graph_stats(G)
    print('\n=== 概念图统计 ===')
    for k, v in stats.items():
        print(f'{k}: {v}')
    
    # 可视化
    output_path = 'test_concept_graph.png'
    visualize_graph(G, output_path, '测试概念图')
    
    # 搜索关联概念
    related = search_by_concept(G, '货币政策', depth=2)
    print(f'\n与"货币政策"相关的概念（2 层）: {related}')
    
    print('\n✅ 测试完成')
