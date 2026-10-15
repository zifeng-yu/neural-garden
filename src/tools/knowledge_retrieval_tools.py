"""知识检索工具：向量检索知识单元及按ID读取原文chunk。"""

import json

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from src.repository.document_chunks import query_by_ids
from src.retrieval.hybrid_search import normalize_hits
from src.search import RetrievalMode, search


class SearchInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
    )
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(ge=1, le=5, default=3)
    retrieval_mode: RetrievalMode = "vector"
    use_graph: bool = False


class ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: int = Field(gt=0, strict=True)
    chunk_id: int = Field(gt=0, strict=True)


@tool(args_schema=SearchInput)
def search_knowledge(
    query: str, top_k: int, retrieval_mode: RetrievalMode, use_graph: bool = False
) -> str:
    """搜索个人知识库的知识单元，只返回候选摘要和ID。
    需要准确引用时，再调用read_chunk读取对应原文。
    """
    result = search(
        query=query,
        top_k=top_k,
        show_score=False,
        save_session=False,
        retrieval_mode=retrieval_mode,
        use_graph=use_graph,
    )

    normalize_result = normalize_hits(result)[:top_k]
    hits = []
    for n in normalize_result:
        hits.append(
            {
                "document_id": n.document_id,
                "chunk_id": n.document_chunk_id,
                "rank": n.rank,
                "summary": n.content[:300],
            }
        )
    return json.dumps(
        {"retrieval_mode": retrieval_mode, "hits": hits}, ensure_ascii=False
    )


@tool(args_schema=ReadInput)
def read_chunk(document_id: int, chunk_id: int) -> str:
    """按检索返回的document_id和chunk_id读取原文，供回答核对与引用。"""
    chunk_do = query_by_ids([chunk_id])
    if len(chunk_do) == 0 or chunk_do[0].document_id != document_id:
        return """{"status":"not_found"}"""
    source_id = f"D{document_id}:C{chunk_id}"
    return json.dumps(
        {
            "status": "found",
            "source_id": source_id,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "content_hash": chunk_do[0].content_hash,
            "text": chunk_do[0].content[:4000],
            "truncated": len(chunk_do[0].content) > 4000,
        },
        ensure_ascii=False,
    )


TOOLS = [search_knowledge, read_chunk]
