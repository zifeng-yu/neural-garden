import argparse
import logging
import os

from src.config.config import (
    PILOT_DATASET_PATH,
)
from src.config.logging_config import setup_logging
from src.document.splitter import markdown_spilt
from src.embedding.getEmbedding import get_embedding
from src.get_sqlite_connection import get_sqlite_connection
from src.knowledge.knowledgeUnit import extract_knowledge_unit
from src.knowledgeGraph.concepts import (
    concept_clustering_in_document,
    documents_to_concepts,
    normalized_concept,
)
from src.knowledgeGraph.concepts_relation import extract_relations_from_text
from src.repository.concept_relations import RelationType, insert_concept_relation
from src.repository.concept_relations import (
    delete_by_id as delete_by_id_concept_relations,
)
from src.repository.document_chunk_concepts import (
    copy_documents_chunk_concepts,
    query_by_chunk_id,
    query_by_not_document_id_in_normalized_concepts,
)
from src.repository.document_chunk_concepts import (
    delete_by_document_id as delete_by_document_id_chunk_concepts,
)
from src.repository.document_chunk_concepts import (
    query_by_document_id as query_by_document_id_chunk_concetps,
)
from src.repository.document_chunk_knowledge_units import (
    copy_documents_chunk_knowledge_unit,
)
from src.repository.document_chunk_knowledge_units import (
    delete_by_document_id as delete_by_document_id_chunk_knowledge_units,
)
from src.repository.document_chunk_knowledge_units import (
    query_by_document_id as query_by_document_id_chunk_knowledge_units,
)
from src.repository.document_chunk_knowledge_units import (
    query_by_ids as query_by_ids_knowunit,
)
from src.repository.document_chunks import copy_document_chunks
from src.repository.document_chunks import (
    delete_by_document_id as delete_by_document_id_chunks,
)
from src.repository.document_chunks import query_by_ids as query_by_id_chunks
from src.repository.document_insert_domain import (
    InsertChunk,
    InsertConcepts,
    InsertDoc,
    InsertDocumentDomain,
    InsertKnowledgeUnit,
    save_documents_domain,
)
from src.repository.documents import (
    copy_documents,
    query_by_content_hash,
    query_by_file_name_hash,
)
from src.repository.documents import delete_by_id as delete_by_id_documents
from src.repository.documents import query_by_id as query_by_id_document
from src.repository.init import sqlite_table_create_if_not, sqlite_table_reset
from src.repository.insights import mark_source_status_offline_by_document_id
from src.repository.insights import query_by_id as query_by_id_insights
from src.repository.relation_evidence import (
    EvidenceRoleEnum,
    copy_relation_evidence,
    insert_relation_evidence,
)
from src.repository.relation_evidence import (
    delete_by_document_id as delete_by_document_id_relation_evidence,
)
from src.repository.relation_evidence import (
    query_by_document_id as query_by_document_id_relation_evidence,
)
from src.repository.relation_evidence import (
    query_by_relation_id as query_by_relation_id_relation_evidence,
)
from src.util.getHashValue import get_hash_value as hash
from src.util.index_logging import index_stage
from src.vector_store.delete_dao import (
    delete_by_id_knowledge as delete_by_id_knowledge_chroma,
)
from src.vector_store.delete_dao import (
    delete_by_normalized_concept_hash as delete_by_normalized_concept_hash_chroma,
)
from src.vector_store.query_dao import (
    log_concept_collection_size,
    log_knowledgeUnit_collection_size,
)
from src.vector_store.reset import resetDB_CONCEPT, resetDB_INSIGHT, resetDB_KNOWLEDGE
from src.vector_store.save_dao import (
    ConceptDTO,
    InsightMetadata,
    KnowledgeUnitDTO,
    KnowledgeUnitMetadata,
    save_concept,
    save_knowlege,
    update_insight_metadata,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="索引配置目录中的 Markdown 文档")
    parser.add_argument(
        "--resetAllDB",
        action="store_true",
        help="清空 SQLite 和三个 Chroma 集合后重新索引，不重置 ES",
    )
    args = parser.parse_args()
    setup_logging()
    if args.resetAllDB:
        logger.warning("正在全量重建：清空 SQLite 和三个 Chroma 集合，不重置 ES")
        resetDB_KNOWLEDGE()
        resetDB_CONCEPT()
        resetDB_INSIGHT()
        sqlite_table_reset()
    else:
        sqlite_table_create_if_not()

    logger.info("🌱 Neural Garden 索引器")
    logger.info("📂 索引目录：%s", PILOT_DATASET_PATH)
    index_directory(PILOT_DATASET_PATH)

    logger.info("\n" + "=" * 40)
    logger.info("💡 提示：运行以下命令搜索知识：")
    logger.info("   python -m src.search <查询内容>")


def load_document(file_path: str) -> str:
    """
    读取 Markdown 文件内容

    Args:
        file_path: 文件路径

    Returns:
        文件内容字符串
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def process_file(content_hash: str, file_name_hash: str, source: str) -> str:

    file_result = query_by_file_name_hash(file_name_hash)

    if file_result:
        if file_result.content_hash == content_hash:
            return "skip"
        document_id = file_result.id
        # 1. 删除 relation evidence，并清理已经没有 evidence 的 relation
        relation_evidence_dos = query_by_document_id_relation_evidence(document_id)
        relation_ids = {x.relation_id for x in relation_evidence_dos}
        #  删除 knowledge unit 的 Chroma 数据 提前数据准备
        know_unit_do_list = query_by_document_id_chunk_knowledge_units(document_id)
        know_unit_do_ids = [str(x.id) for x in know_unit_do_list]
        # 3. 删除不再被其他 document 使用的 concept 的 Chroma数据 提前数据准备
        documents_concepts_do_list = query_by_document_id_chunk_concetps(document_id)
        documents_concepts_do_normalized_concetps_list = [
            x.normalized_concept for x in documents_concepts_do_list
        ]
        other_document_concepts_do_list = (
            query_by_not_document_id_in_normalized_concepts(
                document_id, documents_concepts_do_normalized_concetps_list
            )
        )
        other_document_concepts_do_normalized_concetps_list = [
            x.normalized_concept for x in other_document_concepts_do_list
        ]
        delete_concepts = set(documents_concepts_do_normalized_concetps_list) - set(
            other_document_concepts_do_normalized_concetps_list
        )
        documents_concepts_do_list_group_normalized_concept = {
            d.normalized_concept: d.normalized_concept_hash
            for d in documents_concepts_do_list
        }
        need_delete_normalized_concept_hash_list = [
            documents_concepts_do_list_group_normalized_concept[need_delete]
            for need_delete in delete_concepts
        ]
        try:
            with get_sqlite_connection() as conn:
                delete_by_document_id_relation_evidence(conn, document_id)
                for relation_id in relation_ids:
                    relation_evidence_dos_by_relation_id = (
                        query_by_relation_id_relation_evidence(conn, relation_id)
                    )
                    if not relation_evidence_dos_by_relation_id:
                        delete_by_id_concept_relations(conn, relation_id)
                delete_by_document_id_chunk_concepts(conn, document_id)
                delete_by_document_id_chunk_knowledge_units(conn, document_id)
                delete_by_document_id_chunks(conn, document_id)
                delete_by_id_documents(conn, document_id)
        except Exception:
            logger.exception("del 出现错误，事务回滚 ")
            return "del_failed"

        delete_by_id_knowledge_chroma(know_unit_do_ids)

        delete_by_normalized_concept_hash_chroma(
            need_delete_normalized_concept_hash_list
        )
        insight_ids = mark_source_status_offline_by_document_id(document_id)
        if insight_ids:
            for insight_id in insight_ids:
                insight_do = query_by_id_insights(insight_id)
                if insight_do:
                    insight_metadata = InsightMetadata(
                        insight_do.id,
                        insight_do.source_type.value,
                        insight_do.source_status.value,
                        insight_do.document_id,
                        insight_do.document_chunk_id,
                    )
                    update_insight_metadata(str(insight_do.id), insight_metadata)
        return "need_del"

    content_result = query_by_content_hash(content_hash)

    if content_result:
        try:
            with get_sqlite_connection() as conn:
                new_doucument_id = copy_documents(
                    conn, content_result, source, file_name_hash
                )
                old_new_chunk_id_map = copy_document_chunks(
                    conn, content_result.id, new_doucument_id
                )
                new_know_ids = copy_documents_chunk_knowledge_unit(
                    conn, content_result.id, new_doucument_id, old_new_chunk_id_map
                )
                copy_documents_chunk_concepts(
                    conn, content_result.id, new_doucument_id, old_new_chunk_id_map
                )
                copy_relation_evidence(
                    conn, content_result.id, new_doucument_id, old_new_chunk_id_map
                )
        except Exception:
            logger.exception("copy 出现错误，事务回滚 ")
            return "copy_failed"

        new_know_do_list = query_by_ids_knowunit(new_know_ids)
        if new_know_do_list:
            for new_know_do in new_know_do_list:
                embedding_result = get_embedding(new_know_do.embedding_text)
                if not embedding_result:
                    continue
                save_knowlege(
                    KnowledgeUnitDTO(
                        str(new_know_do.id),
                        embedding_result,
                        new_know_do.embedding_text,
                        KnowledgeUnitMetadata(
                            source,
                            new_know_do.document_id,
                            new_know_do.document_chunk_id,
                            new_know_do.title,
                            new_know_do.keywords,
                        ),
                    )
                )
        return "need_copy"

    return "need_new"


def index_file(file_path: str):
    """
    将单个文件导入 Chroma 向量库

    Args:
        file_path: Markdown 文件路径
        persist_dir: Chroma 持久化目录
    """
    # 读取文件
    content = load_document(file_path)
    source = os.path.basename(file_path)

    with index_stage(logger, "document", source=source) as document_stats:
        logger.info(f"📄 正在索引：{source}")

        # 2. 查询是否有记录
        file_name_hash = hash(source)
        content_hash = hash(content)
        action = process_file(content_hash, file_name_hash, source)

        document_stats["action"] = action
        if action != "need_del" and action != "need_new":
            return

        # 1. docDO
        insert_doc = InsertDoc(source, file_name_hash, content_hash)
        # 2. chunk
        chunks = markdown_spilt(content)
        if chunks is None:
            return
        logger.info("Document chunks source=%s chunks=%d", source, len(chunks))
        # 组insert对象
        insert_chunks = []
        split_no = 1
        for chunk in chunks:
            # 3 获取知识单元 - 这里从chunk来获取
            with index_stage(
                logger, "knowledge_extract", chunk_no=split_no, chars=len(chunk)
            ) as stats:
                knowledgeUnit = extract_knowledge_unit(
                    content=chunk, id=f"filename:{source},splitNo:{split_no}"
                )
                stats["outcome"] = "skipped" if knowledgeUnit is None else "ok"
            insert_chunk = InsertChunk(split_no, chunk, hash(chunk))
            split_no += 1
            if knowledgeUnit is None:
                logger.warning(
                    "Knowledge missing; retaining chunk source=%s chunk_no=%d",
                    source,
                    insert_chunk.split_no,
                )
                insert_chunks.append(insert_chunk)
                continue
            insert_chunk.know_unit = InsertKnowledgeUnit(
                knowledgeUnit.title,
                knowledgeUnit.summary,
                knowledgeUnit.keywords,
                knowledgeUnit.to_embedding_text(),
            )
            # 4 concept
            with index_stage(
                logger, "concept_extract", chunk_no=insert_chunk.split_no
            ) as stats:
                get_concepts = documents_to_concepts(knowledgeUnit.title, chunk)
                if get_concepts:
                    insert_concepts = [
                        InsertConcepts(x, x, hash(x)) for x in get_concepts
                    ]
                    insert_chunk.concepts = insert_concepts

                stats["concepts"] = len(get_concepts or [])
            insert_chunks.append(insert_chunk)

        document_stats["chunks"] = len(insert_chunks)
        document_stats["knowledge"] = sum(
            x.know_unit is not None for x in insert_chunks
        )
        document_stats["knowledge_skipped"] = (
            len(insert_chunks) - document_stats["knowledge"]
        )
        logger.info(f"归一前 数据 {[a.concepts for a in insert_chunks]}")
        # 文档内部概念聚集（归一）
        with index_stage(logger, "normalize_internal"):
            concept_clustering_in_document(insert_chunks)
        logger.info(f"内部归一后 数据 {[a.concepts for a in insert_chunks]}")
        # 文档外部概念聚聚（归一）
        with index_stage(logger, "normalize_external"):
            normalized_concept(insert_chunks)
        logger.info(f"外部归一后 数据 {[a.concepts for a in insert_chunks]}")

        with index_stage(logger, "sqlite_save"):
            save_db_result = save_documents_domain(
                InsertDocumentDomain(insert_doc, insert_chunks)
            )

        logger.info(f"save_db_result {save_db_result}")

        with index_stage(
            logger, "knowledge_vectors", saved=0, missing=0
        ) as vector_stats:
            # 保存knowledgeUnit chromaDB
            if "knowledge_unit_ids" in save_db_result:
                knowledgeUnit_ids = save_db_result["knowledge_unit_ids"]
                knowledgeUnit_list = query_by_ids_knowunit(knowledgeUnit_ids)
                if knowledgeUnit_list:
                    for unit in knowledgeUnit_list:
                        get_embedding_result = get_embedding(unit.embedding_text)
                        if not get_embedding_result:
                            vector_stats["missing"] += 1
                            logger.warning(
                                "Vector missing source=%s stage=%s",
                                source,
                                vector_stats["stage"],
                            )
                            continue
                        save_knowlege(
                            KnowledgeUnitDTO(
                                str(unit.id),
                                get_embedding_result,
                                unit.embedding_text,
                                KnowledgeUnitMetadata(
                                    source,
                                    unit.document_id,
                                    unit.document_chunk_id,
                                    unit.title,
                                    unit.keywords,
                                ),
                            )
                        )
                        vector_stats["saved"] += 1
        with index_stage(logger, "concept_vectors", saved=0, missing=0) as vector_stats:
            # 保存 concept chromaDB
            if "document_id" in save_db_result:
                documentDO = query_by_id_document(save_db_result["document_id"])
                if documentDO:
                    document_chunk_concepts_dolist = (
                        query_by_document_id_chunk_concetps(documentDO.id)
                    )
                    for document_chunk_concepts_do in document_chunk_concepts_dolist:
                        get_embedding_result = get_embedding(
                            document_chunk_concepts_do.normalized_concept
                        )
                        if not get_embedding_result:
                            vector_stats["missing"] += 1
                            logger.warning(
                                "Vector missing source=%s stage=%s",
                                source,
                                vector_stats["stage"],
                            )
                            continue
                        save_concept(
                            ConceptDTO(
                                document_chunk_concepts_do.normalized_concept_hash,
                                get_embedding_result,
                                document_chunk_concepts_do.normalized_concept,
                            )
                        )

                        vector_stats["saved"] += 1
        log_concept_collection_size()
        log_knowledgeUnit_collection_size()

        with index_stage(logger, "relations"):
            # 建立chunk内部概念连接
            if "chunk_ids" in save_db_result:
                chunks_do = query_by_id_chunks(save_db_result["chunk_ids"])
                if chunks_do:
                    for chunk_do in chunks_do:
                        concept_dos = query_by_chunk_id(chunk_do.id)
                        if concept_dos:
                            chunk_content = chunk_do.content
                            concepts = [x.normalized_concept for x in concept_dos]
                            with index_stage(
                                logger,
                                "relation_extract",
                                chunk_id=chunk_do.id,
                                concepts=len(concepts),
                            ) as relation_stats:
                                chunk_concept_relations = extract_relations_from_text(
                                    chunk_content, concepts
                                )
                                relation_stats["relations"] = len(
                                    chunk_concept_relations or []
                                )
                                relation_stats["outcome"] = (
                                    "skipped"
                                    if chunk_concept_relations is None
                                    else "ok"
                                )
                            if chunk_concept_relations is None:
                                continue
                            for (
                                concept_source,
                                realtion,
                                concept_target,
                            ) in chunk_concept_relations:
                                if (
                                    concept_source in concepts
                                    and concept_target in concepts
                                ):
                                    concept_realtion_id = insert_concept_relation(
                                        concept_source,
                                        concept_target,
                                        realtion,
                                        RelationType.DOCUMENT_INTERNAL,
                                    )
                                    if concept_realtion_id is not None:
                                        insert_relation_evidence(
                                            concept_realtion_id,
                                            chunk_do.document_id,
                                            chunk_do.id,
                                            EvidenceRoleEnum.BOTH,
                                        )


def index_directory(dir_path: str):
    """
    批量索引目录下所有 .md 文件

    Args:
        dir_path: 目录路径
        persist_dir: Chroma 持久化目录
    """
    md_files = [f for f in os.listdir(dir_path) if f.endswith(".md")]

    if not md_files:
        logger.warning("⚠️  未找到 .md 文件：%s", dir_path)
        return

    logger.info(f"📂 发现 {len(md_files)} 个 Markdown 文件\n")

    for filename in md_files:
        file_path = os.path.join(dir_path, filename)
        index_file(file_path)

    logger.info(f"✅ 批量索引完成，共 {len(md_files)} 个文件")
    log_concept_collection_size()
    log_knowledgeUnit_collection_size()


if __name__ == "__main__":
    main()
