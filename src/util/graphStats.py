import logging
from dataclasses import asdict, dataclass, field

import networkx as nx

import src.config.logging_config as logging_config
from src.knowledgeGraph.knowledge_graph import NodeAttribute, NodeType

logger = logging.getLogger(__name__)


@dataclass
class DiGraphStats:
    """graph统计结构"""

    # 节点数
    number_of_nodes: int = 0
    # 边数
    number_of_edges: int = 0
    # 密度
    density: float = 0.0
    # 强连通分量数
    number_of_strongly_connected: int = 0
    # 弱连通分量数
    number_of_weakly_connected: int = 0
    # 平均度
    avg_degree: float = 0.0
    # 文档最多的相同概念:
    max_doc_same_concept: list[tuple[str, int]] = field(default_factory=list)
    # 入度做多的概念：
    max_in_degree_concept: list[tuple[str, int]] = field(default_factory=list)
    # pagerank
    pagerank: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_stats_text(self) -> str:
        """生成可读文本"""
        lines = [
            "===== Graph Statistics =====",
            f"节点数: {self.number_of_nodes}",
            f"边数: {self.number_of_edges}",
            f"密度: {self.density:.4f}",
            f"强连通分量数: {self.number_of_strongly_connected}",
            f"弱连通分量数: {self.number_of_weakly_connected}",
            f"平均度: {self.avg_degree:.2f}",
            "",
            "文档热度 Top5:",
        ]

        for concept, count in self.max_doc_same_concept:
            lines.append(f"  {concept}: {count}")

        lines.append("")
        lines.append("概念中心 Top5:")

        for concept, count in self.max_in_degree_concept:
            lines.append(f"  {concept}: {count}")

        lines.append("")
        lines.append("PageRank Top5:")

        for concept, score in self.pagerank:
            lines.append(f"  {concept}: {score:.4f}")

        return "\n".join(lines)


def get_DiGraph_stats(G: nx.DiGraph) -> DiGraphStats:
    """
    获取图的统计信息
    Args:
        G: NetworkX 图对象
    Returns:
        统计信息字典
    """
    digraphStats = DiGraphStats(
        number_of_nodes=G.number_of_nodes(),
        number_of_edges=G.number_of_edges(),
        density=nx.density(G),
        number_of_strongly_connected=nx.number_strongly_connected_components(G),
        number_of_weakly_connected=nx.number_weakly_connected_components(G),
        avg_degree=(
            sum(dict(G.degree()).values()) / G.number_of_nodes()
            if G.number_of_nodes() > 0
            else 0
        ),
    )

    # 找到度最高的节点（最核心的概念）
    if G.number_of_nodes() > 0:
        degree_dict = dict(G.in_degree())
        degree_dict = {
            k: v
            for k, v in degree_dict.items()
            if G.nodes[k][NodeAttribute.TYPE.value] == NodeType.CONCEPT.value
        }
        top_concepts = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        digraphStats.max_doc_same_concept = top_concepts
    concept_nodes = [
        node
        for node, data in G.nodes(data=True)
        if data.get(NodeAttribute.TYPE.value) == NodeType.CONCEPT.value
    ]
    if concept_nodes:
        concept_graph = G.subgraph(concept_nodes)
        top_concept = sorted(
            concept_graph.in_degree(), key=lambda x: x[1], reverse=True
        )[:5]
        digraphStats.max_in_degree_concept = top_concept

        pagerank = nx.pagerank(concept_graph)
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
        digraphStats.pagerank = top_pagerank
    return digraphStats
