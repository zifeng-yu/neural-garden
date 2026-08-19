import logging

import matplotlib.pyplot as plt
import networkx as nx

import src.config.logging_config as logging_config
from src.knowledgeGraph.knowledge_graph import NodeAttribute, NodeType

logger = logging.getLogger(__name__)


def visualize_graph(G: nx.MultiDiGraph, output_path: str, title: str = "概念图"):
    """
    可视化概念图
    Args:
        G: NetworkX 图对象
        output_path: 输出对象路径
        title: 图标题
    """
    FONT = "PingFang SC"

    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G, k=1.5, iterations=200, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue", alpha=0.3)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)

    labels = {}
    for node, data in G.nodes(data=True):
        if data.get(NodeAttribute.TYPE.value) == NodeType.DOCUMENT.value:
            labels[node] = data.get(NodeAttribute.TITLE.value, node)
        else:
            labels[node] = node
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=10, font_family=FONT, font_weight="bold"
    )

    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_family=FONT, font_size=8
    )

    plt.title(title, fontdict={"family": FONT})
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path + f"{title}.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ 概念图已保存至：{output_path}")
