import logging
import os

import src.config.logging_config as logging_config
from src.config.config import (
    PILOT_DATASET_PATH,
)
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
)
from src.repository.document_chunk_concepts import (
    delete_by_document_id as delete_by_document_id_chunk_concepts,
)
from src.repository.document_chunk_concepts import (
    query_by_document_id as query_by_document_id_chunk_concetps,
)
from src.repository.document_chunk_concepts import (
    query_by_not_document_id_and_in_normalized_concepts as query_by_not_document_id_and_in_normalized_concepts_chunk_concepts,
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
from src.repository.init import sqlite_table_init
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
from src.vector_store.delete_dao import (
    delete_by_id_knowledge as delete_by_id_knowledge_chroma,
)
from src.vector_store.delete_dao import (
    delete_by_normalized_concet_hash_concept as delete_by_normalized_concet_hash_concept_chroma,
)
from src.vector_store.query_dao import (
    log_concept_collection_size,
    log_knowledgeUnit_collection_size,
)
from src.vector_store.reset import resetDB_CONCEPT, resetDB_KNOWLEDGE
from src.vector_store.save_dao import (
    ConceptDTO,
    KnowledgeUnitDTO,
    KnowledgeUnitMetadata,
    save_concept,
    save_knowlege,
)

logger = logging.getLogger(__name__)


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
        delete_by_document_id_relation_evidence(document_id)

        for relation_id in relation_ids:
            relation_evidence_dos_by_relation_id = (
                query_by_relation_id_relation_evidence(relation_id)
            )
            if not relation_evidence_dos_by_relation_id:
                delete_by_id_concept_relations(relation_id)

        # 2. 删除 knowledge unit 的 Chroma 数据
        know_unit_do_list = query_by_document_id_chunk_knowledge_units(document_id)
        know_unit_do_ids = [str(x.id) for x in know_unit_do_list]
        delete_by_id_knowledge_chroma(know_unit_do_ids)

        # 3. 删除不再被其他 document 使用的 concept
        documents_concepts_do_list = query_by_document_id_chunk_concetps(document_id)
        documents_concepts_do_normalized_concetps_list = [
            x.normalized_concept for x in documents_concepts_do_list
        ]
        other_document_concepts_do_list = (
            query_by_not_document_id_and_in_normalized_concepts_chunk_concepts(
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
        delete_by_normalized_concet_hash_concept_chroma(
            need_delete_normalized_concept_hash_list
        )

        # 4. 删除 SQLite 中 document 相关数据
        delete_by_document_id_chunk_concepts(document_id)
        delete_by_document_id_chunk_knowledge_units(document_id)
        delete_by_document_id_chunks(document_id)
        delete_by_id_documents(document_id)

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
                new_chunk_concepts_ids = copy_documents_chunk_concepts(
                    conn, content_result.id, new_doucument_id, old_new_chunk_id_map
                )
                new_relation_evidence_ids = copy_relation_evidence(
                    conn, content_result.id, new_doucument_id, old_new_chunk_id_map
                )
        except Exception:
            logger.exception("copy 出现错误，事务回滚 ")
            return "copy_failed"

        new_know_do_list = query_by_ids_knowunit(new_know_ids)
        if new_know_do_list:
            for new_know_do in new_know_do_list:
                save_knowlege(
                    KnowledgeUnitDTO(
                        str(new_know_do.id),
                        get_embedding(new_know_do.embedding_text),
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

    logger.info(f"📄 正在索引：{source}")

    # 2. 查询是否有记录
    file_name_hash = hash(source)
    content_hash = hash(content)
    action = process_file(content_hash, file_name_hash, source)

    if action != "need_del" and action != "need_new":
        return

    # 1. docDO
    insert_doc = InsertDoc(source, file_name_hash, content_hash)
    # 2. chunk
    chunks = markdown_spilt(content)
    if chunks is None:
        return
    # 组insert对象
    insert_chunks = []
    split_no = 1
    for chunk in chunks:
        # 3 获取知识单元 - 这里从chunk来获取
        knowledgeUnit = extract_knowledge_unit(
            content=chunk, id=f"filename:{source},splitNo:{split_no}"
        )
        insert_chunk = InsertChunk(split_no, chunk, hash(chunk))
        split_no += 1
        if knowledgeUnit is None:
            logger.info(f"知识提取失败，source {source}")
            insert_chunks.append(insert_chunk)
            continue
        insert_chunk.know_unit = InsertKnowledgeUnit(
            knowledgeUnit.title,
            knowledgeUnit.summary,
            knowledgeUnit.keywords,
            knowledgeUnit.to_embedding_text(),
        )
        # 4 concept
        get_concepts = documents_to_concepts(knowledgeUnit.title, chunk)
        if get_concepts:
            insert_concepts = [InsertConcepts(x, x, hash(x)) for x in get_concepts]
            insert_chunk.concepts = insert_concepts

        insert_chunks.append(insert_chunk)

    logger.info(f"归一前 数据 {[a.concepts for a in insert_chunks]}")
    # 文档内部概念聚集（归一）
    concept_clustering_in_document(insert_chunks)
    logger.info(f"内部归一后 数据 {[a.concepts for a in insert_chunks]}")
    # 文档外部概念聚聚（归一）
    normalized_concept(insert_chunks)
    logger.info(f"外部归一后 数据 {insert_chunks}")

    save_db_result = save_documents_domain(
        InsertDocumentDomain(insert_doc, insert_chunks)
    )

    logger.info(f"save_db_result {save_db_result}")

    # 保存knowledgeUnit 准备con
    if "knowledge_unit_ids" in save_db_result:
        knowledgeUnit_ids = save_db_result["knowledge_unit_ids"]
        knowledgeUnit_list = query_by_ids_knowunit(knowledgeUnit_ids)
        if knowledgeUnit_list:
            for unit in knowledgeUnit_list:
                save_knowlege(
                    KnowledgeUnitDTO(
                        str(unit.id),
                        get_embedding(unit.embedding_text),
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
    # 保存 concept
    if "document_id" in save_db_result:
        documentDO = query_by_id_document(save_db_result["document_id"])
        if documentDO:
            document_chunk_concepts_dolist = query_by_document_id_chunk_concetps(
                documentDO.id
            )
            for document_chunk_concepts_do in document_chunk_concepts_dolist:
                save_concept(
                    ConceptDTO(
                        document_chunk_concepts_do.normalized_concept_hash,
                        get_embedding(document_chunk_concepts_do.normalized_concept),
                        document_chunk_concepts_do.normalized_concept,
                    )
                )

    log_concept_collection_size()
    log_knowledgeUnit_collection_size()

    # 建立chunk内部概念连接
    if "chunk_ids" in save_db_result:
        chunks_do = query_by_id_chunks(save_db_result["chunk_ids"])
        if chunks_do:
            for chunk_do in chunks_do:
                concept_dos = query_by_chunk_id(chunk_do.id)
                if concept_dos:
                    chunk_content = chunk_do.content
                    concepts = [x.normalized_concept for x in concept_dos]
                    chunk_concept_relations = extract_relations_from_text(
                        chunk_content, concepts
                    )
                    for (
                        concept_source,
                        realtion,
                        concept_target,
                    ) in chunk_concept_relations:
                        if concept_source in concepts and concept_target in concepts:
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
        print(f"⚠️  未找到 .md 文件：{dir_path}")
        return

    logger.info(f"📂 发现 {len(md_files)} 个 Markdown 文件\n")

    for filename in md_files:
        file_path = os.path.join(dir_path, filename)
        index_file(file_path)

    logger.info(f"\n✅ 批量索引完成，共 {len(md_files)} 个文件")
    log_concept_collection_size()
    log_knowledgeUnit_collection_size()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if "--resetDB" in args:
        resetDB_KNOWLEDGE()
        resetDB_CONCEPT()
        sqlite_table_init()

    # 配置路径
    base_dir = os.path.dirname(os.path.dirname(__file__))
    pilot_dir = os.path.join(base_dir, PILOT_DATASET_PATH)

    # 默认索引 pilot/ 目录
    logger.info("🌱 Neural Garden 索引器（骨架版本）")
    logger.info(f"📂 索引目录：{pilot_dir}")

    index_directory(pilot_dir)

    logger.info("\n" + "=" * 40)
    logger.info("💡 提示：运行以下命令搜索知识：")
    logger.info(f"   python -m src.search <查询内容>")
