"""
构建知识图
1. 获取文档
2. 构建图
3. 查询
"""

import logging
import os
from typing import Any

import networkx as nx

import src.config.logging_config as logging_config
from src.config.config import PILOT_DATASET_PATH
from src.knowledgeGraph.knowledge_graph import (
    KnowledgeDocument,
    NodeAttribute,
    NodeType,
    build_concept_graph,
    search_by_node,
)
from src.util.graphStats import get_DiGraph_stats
from src.util.visualizeGraph import visualize_graph

logger = logging.getLogger(__name__)


def load_documents(file_path: str) -> str:
    """
    读取 Markdown 文件内容

    Args:
        file_path: 文件路径

    Returns:
        文件内容字符串
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def index_directory(dir_path: str) -> list[KnowledgeDocument] | None:
    """
    批量索引目录下所有 .md 文件

    Args:
        dir_path: 目录路径
    """
    md_files = [f for f in os.listdir(dir_path) if f.endswith(".md")]

    if not md_files:
        print(f"⚠️  未找到 .md 文件：{dir_path}")
        return

    logger.info(f"📂 发现 {len(md_files)} 个 Markdown 文件\n")

    result = []
    for filename in md_files:
        file_path = os.path.join(dir_path, filename)
        knowledge_document = KnowledgeDocument(
            title=os.path.basename(file_path), content=load_documents(file_path)
        )
        result.append(knowledge_document)

    logger.info(f"\n✅ 批量生成文件对象，共 {len(md_files)} 个文件")
    return result


def build_graph(dir_path: str) -> nx.DiGraph | None:
    knowledgeDocuments = index_directory(dir_path)
    if knowledgeDocuments is None:
        return None
    return build_concept_graph(knowledgeDocuments)


def graph_png(G: nx.DiGraph):
    # 可视化
    visualize_graph(G, "graphPNG//", "Pilot Dataset 概念图")


def log_stats(G: nx.DiGraph):
    # 分析统计信息
    digraphtStats = get_DiGraph_stats(G)
    logger.info(digraphtStats.to_stats_text())


def search(G: nx.DiGraph, concept: str, depth: int) -> list[dict[str, Any]]:
    return search_by_node(G, concept, depth)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    pilot_dir = os.path.join(base_dir, PILOT_DATASET_PATH)
    knowledge_graph = build_graph(pilot_dir)
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

        titles = [
            "01-金融深潜 - 日本负利率.md",
            "02-算法 - 重复检测.md",
            "03-生物学 - 昂贵的代价.md",
            "04-公众号 - 花自己的钱.md",
            "05-认知科学 - 专家预测.md",
        ]
        titles_concept_map = {}
        for title in titles:
            hash_value = hash(title)
            doc_hava_concepts = list(knowledge_graph.successors(hash_value))
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

                logger.info(f"{u} -> {v} ,{edge_data.get("relation")}")

        from collections import deque

        result = []
        hash_value = hash(titles[2])
        queue = deque([(hash_value, [hash_value])])

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
