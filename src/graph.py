"""
构建知识图
1. 获取文档
2. 构建图
3. 查询
"""

import logging
from typing import Any

import networkx as nx

import src.config.logging_config as logging_config
from src.knowledgeGraph.knowledge_graph import (
    NodeAttribute,
    NodeType,
    build_concept_graph,
    search_by_node,
)
from src.util.graphStats import get_DiGraph_stats
from src.util.visualizeGraph import visualize_graph

logger = logging.getLogger(__name__)


def build_graph() -> nx.MultiDiGraph | None:
    return build_concept_graph()


def graph_png(G: nx.MultiDiGraph):
    # 可视化
    visualize_graph(G, "graphPNG//", "Pilot Dataset 概念图")


def log_stats(G: nx.MultiDiGraph):
    # 分析统计信息
    digraphtStats = get_DiGraph_stats(G)
    logger.info(digraphtStats.to_stats_text())


def search(G: nx.MultiDiGraph, concept: str, depth: int) -> list[dict[str, Any]]:
    return search_by_node(G, concept, depth)


if __name__ == "__main__":
    knowledge_graph = build_graph()
    if knowledge_graph is not None:
        graph_png(knowledge_graph)
        log_stats(knowledge_graph)

    # 以下为test
    if knowledge_graph is not None:
        count = 0
        for u, v in knowledge_graph.edges():
            if (
                knowledge_graph.nodes[u].get(NodeAttribute.TYPE.value)
                == NodeType.CONCEPT
                and knowledge_graph.nodes[v].get(NodeAttribute.TYPE.value)
                == NodeType.CONCEPT
            ):
                count += 1
        logger.info(f"概念相连 数量 {count}")

        from src.util.getHashValue import get_hash_value as hash

        titles = [1, 2, 4, 5, 6, 7]
        titles_concept_map = {}
        for title in titles:
            doc_hava_concepts = list(knowledge_graph.successors(title))
            titles_concept_map[title] = doc_hava_concepts
        logger.info(titles_concept_map)
        from collections import defaultdict

        concept_in_titles = defaultdict(list)
        for k, v in titles_concept_map.items():
            for concept in v:
                concept_in_titles[concept].append(k)

        logger.info(f"概念数量:{len(concept_in_titles)}")
        logger.info(concept_in_titles)
        for k, v in concept_in_titles.items():
            if len(v) > 1:
                logger.info(f"{k} <- {v}")

        paths = nx.single_source_shortest_path(
            knowledge_graph, titles_concept_map[titles[0]][0], cutoff=2
        )
        for node, path in paths.items():
            logger.info(f"node:{node}, path:{path}")
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]

                edge_data = knowledge_graph.get_edge_data(u, v)

                logger.info(f"{u} -> {v} ,{edge_data.get('relation')}")

        from collections import deque

        result = []
        queue = deque([(titles[2], [titles[2]])])

        while queue:
            node, path = queue.popleft()

            if len(path) - 1 >= 5:
                continue

            for next_node in knowledge_graph.successors(node):
                new_path = path + [next_node]
                result.append(new_path)

                queue.append((next_node, new_path))

        logger.info([x for x in result if len(x) > 2])

        # for title in titles:
        #     hash_value = hash(title)
        #     doc_hava_concepts = list(knowledge_graph.successors(hash_value))
        #     # logger.info(f"{title} 拥有的概念：{doc_hava_concepts}")
        #     # 文档 -> 概念 遍历
        #     for concept in doc_hava_concepts:
        #         search_result = search(knowledge_graph, concept, 10)
        #         # logger.info(f"{concept}能走到的概念集合:{search_result}")
        #         # 文档 -> 概念 -> 概念 遍历
        #         for dict in search_result:
        #             # 概念 <- 文档
        #             docs = knowledge_graph.predecessors(dict["id"])
        #             related_docs = []
        #             for doc in docs:
        #                 if (
        #                     knowledge_graph.nodes[doc].get(NodeAttribute.TYPE.value)
        #                     == NodeType.DOCUMENT.value
        #                 ):
        #                     if (
        #                         knowledge_graph.nodes[doc].get(
        #                             NodeAttribute.TITLE.value
        #                         )
        #                         != title
        #                     ):
        #                         related_docs.append(
        #                             knowledge_graph.nodes[doc].get(
        #                                 NodeAttribute.TITLE.value
        #                             )
        #                         )
        #             logger.info(f"{title} -> {concept} <- docs {related_docs}")
