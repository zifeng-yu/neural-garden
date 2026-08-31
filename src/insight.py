from dataclasses import dataclass

from src.embedding.getEmbedding import get_embedding
from src.knowledgeGraph.concepts_extractor import extract_concepts_from_text
from src.repository.document_chunk_concepts import (
    query_by_chunk_id as query_by_chunk_id_chunks_concepts,
)
from src.repository.document_chunk_concepts import (
    query_by_document_id as query_by_document_id_chunks_concepts,
)
from src.repository.document_chunk_knowledge_units import (
    query_by_title as query_by_title_knowledge_units,
)
from src.repository.documents import query_by_file_name as query_by_file_name_documents
from src.repository.insights import SourceStatusEnum, SourceTypeEnum
from src.repository.insights import query_by_id as query_by_id_insights
from src.repository.insights import save_insights as save_insights_sqlite
from src.vector_store.save_dao import (
    InsightDTO,
    InsightMetadata,
    log_insight_collection_count,
)
from src.vector_store.save_dao import save_insight as save_insight_chroma


def create_insight(
    title: str,
    content: str,
    action_items: list[str] | None,
    relation_concepts: list[str] | None,
    from_document_title: str | None,
) -> int:
    if action_items is None:
        action_items = []
    document_id = None
    document_chunk_id = None
    source_type = SourceTypeEnum.USER
    if from_document_title is not None:
        knowledge_unit_list = query_by_title_knowledge_units(from_document_title)
        if knowledge_unit_list:
            document_id = knowledge_unit_list[0].document_id
            document_chunk_id = knowledge_unit_list[0].document_chunk_id
        else:
            document_list = query_by_file_name_documents(from_document_title)
            if document_list:
                document_id = document_list[0].id
    if relation_concepts is None:
        chunks_concepts_list = []
        if document_chunk_id is not None:
            chunks_concepts_list = query_by_chunk_id_chunks_concepts(document_chunk_id)
        elif document_id is not None:
            chunks_concepts_list = query_by_document_id_chunks_concepts(document_id)

        if chunks_concepts_list:
            relation_concepts = list(
                {
                    c.normalized_concept
                    for c in chunks_concepts_list
                    if c.normalized_concept
                }
            )
        else:
            full_text = f"<文本标题>{title}</文本标题>\n<文本内容>{content}</文本内容>"
            relation_concepts = extract_concepts_from_text(full_text)
    if document_id is not None:
        source_type = SourceTypeEnum.DOCUMENT
    return save_insights_sqlite(
        title,
        content,
        action_items,
        relation_concepts,
        source_type,
        document_id,
        document_chunk_id,
        SourceStatusEnum.ONLINE,
        InsightEmbeddingTextDTO(
            title, content, action_items, relation_concepts
        ).to_embedding_text(),
    )


@dataclass
class InsightEmbeddingTextDTO:
    title: str
    content: str
    action_items: list[str]
    relation_concepts: list[str]

    def to_embedding_text(self) -> str:
        """生成用于向量化的文本（标题 + 摘要 + 关键词）"""
        return (
            f"{self.title}。{self.content}。行动项：{', '.join(self.action_items)}。"
            f"关联概念:{', '.join(self.relation_concepts)}"
        )


def create_insight_vector(insight_id: int):
    insight_do = query_by_id_insights(insight_id)
    if not insight_do:
        return
    insight_embedding_text_dto = InsightEmbeddingTextDTO(
        insight_do.title,
        insight_do.content,
        insight_do.action_items,
        insight_do.relation_concepts,
    )
    embedding_result = get_embedding(insight_embedding_text_dto.to_embedding_text())
    if embedding_result is None:
        return
    insight_dto = InsightDTO(
        str(insight_do.id),
        embedding_result,
        insight_embedding_text_dto.to_embedding_text(),
        InsightMetadata(
            insight_do.id,
            insight_do.source_type.value,
            insight_do.source_status.value,
            insight_do.document_id,
            insight_do.document_chunk_id,
        ),
    )

    save_insight_chroma(insight_dto)
    log_insight_collection_count()


def user_insight_data(
    title: str,
    content: str,
    action_items: list[str] | None,
    relation_concepts: list[str] | None,
    from_document_title: str | None,
):
    sqlite_id = create_insight(
        title, content, action_items, relation_concepts, from_document_title
    )
    create_insight_vector(sqlite_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--title",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--content",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--action-items",
    )
    parser.add_argument(
        "--relation-concepts",
        type=str,
    )
    parser.add_argument(
        "--source-title",
        type=str,
    )
    args = parser.parse_args()
    import json

    user_insight_data(
        args.title,
        args.content,
        json.loads(args.action_items) if args.action_items else None,
        json.loads(args.relation_concepts) if args.relation_concepts else None,
        args.source_title if args.source_title else None,
    )
