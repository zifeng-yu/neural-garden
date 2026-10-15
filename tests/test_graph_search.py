"""当前图检索契约：临时 SQLite、真实 NetworkX，模型与向量库使用替身。"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import networkx as nx
import pytest

import src.get_sqlite_connection as connections
from src.knowledgeGraph.knowledge_graph import build_concept_graph
from src.repository.create_table import create_tables
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.retrieval import graph_search as module
from src.retrieval.search_result import SearchResult


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "graph.sqlite"
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    # 未显式模拟的外部调用直接失败，避免访问业务数据和模型。
    monkeypatch.setattr(
        module,
        "get_embedding",
        Mock(side_effect=AssertionError("unexpected embedding")),
    )
    monkeypatch.setattr(
        module,
        "get_existing_collection",
        Mock(side_effect=AssertionError("unexpected Chroma access")),
    )
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        create_tables(conn)
        conn.execute(
            "INSERT INTO documents(id,file_name,file_name_hash,content_hash) "
            "VALUES(1,'graph.md','file-hash','document-hash')"
        )
        for chunk_id in range(1, 7):
            conn.execute(
                "INSERT INTO document_chunks"
                "(id,document_id,split_no,content,content_hash) VALUES(?,1,?,?,?)",
                (chunk_id, chunk_id, f"original {chunk_id}", f"hash-{chunk_id}"),
            )
            conn.execute(
                "INSERT INTO document_chunk_knowledge_units"
                "(document_id,document_chunk_id,title,summary,keywords,embedding_text) "
                "VALUES(1,?,?,?,?,?)",
                (chunk_id, "title", "summary", "[]", f"embedding {chunk_id}"),
            )
        # A 的映射故意乱序；B 与 A 共享 chunk 1，并有关系证据之外的 chunk 5。
        mappings = {"A": [3, 1], "B": [5, 1, 4], "C": [6], "D": [2]}
        for name, ids in mappings.items():
            for chunk_id in ids:
                conn.execute(
                    "INSERT INTO document_chunk_concepts"
                    "(document_id,document_chunk_id,concept,normalized_concept,"
                    "normalized_concept_hash) VALUES(1,?,?,?,?)",
                    (chunk_id, name, name, name),
                )
        relations = [(7, "A", "B"), (8, "A", "C"), (9, "D", "A")]
        for relation_id, source, target in relations:
            conn.execute(
                "INSERT INTO concept_relations"
                "(id,source_concept,target_concept,relation,relation_type) "
                "VALUES(?,?,?,'uses','document_internal')",
                (relation_id, source, target),
            )
            conn.execute(
                "INSERT INTO relation_evidence"
                "(relation_id,document_id,document_chunk_id,evidence_role) "
                "VALUES(?,1,1,'both')",
                (relation_id,),
            )
    return path


def chunk_ids(hits):
    return [hit.document_chunk_id for hit in hits]


def test_expansion_returns_ranked_knowledge_candidates(database):
    hits = module._expand_from_concepts(build_concept_graph(), ["A"])
    assert all(isinstance(hit, SearchResult) for hit in hits)
    assert chunk_ids(hits) == [1, 2, 3, 4, 5, 6]
    assert [hit.content for hit in hits] == [f"embedding {i}" for i in range(1, 7)]
    assert [hit.rank for hit in hits] == list(range(1, 7))
    assert [hit.score for hit in hits] == pytest.approx([1 / i for i in range(1, 7)])
    assert all(hit.document_id == 1 for hit in hits)
    assert all(hit.source_type is SourceTypeEnum.GRAPH_KNOWLEDGE for hit in hits)
    assert all(hit.retrieval_type is RetrievalTypeEnum.GRAPH for hit in hits)
    assert all(hit.raw_score is None for hit in hits)


@pytest.mark.parametrize("limit,expected", [(1, [1, 4, 3, 5]), (2, [1, 2, 3, 4, 5])])
def test_relation_limit_interleaves_directions_after_filtering_mentions(
    database, limit, expected
):
    hits = module._expand_from_concepts(
        build_concept_graph(), ["A"], relation_limit_per_seed=limit
    )
    assert chunk_ids(hits) == expected


def test_candidate_limit_applies_after_merge(database):
    hits = module._expand_from_concepts(build_concept_graph(), ["A"], candidate_k=3)
    assert chunk_ids(hits) == [1, 2, 3]
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_multiple_seeds_are_direct_before_interleaving(database):
    hits = module._expand_from_concepts(build_concept_graph(), ["A", "B"])
    assert chunk_ids(hits) == [1, 2, 3, 6, 4, 5]


def test_relation_limit_is_independent_for_each_seed(database):
    hits = module._expand_from_concepts(
        build_concept_graph(), ["A", "D"], relation_limit_per_seed=1
    )
    assert chunk_ids(hits) == [1, 4, 2, 5, 3]


def test_edge_insertion_order_does_not_change_results(database):
    graph = build_concept_graph()
    reversed_graph = nx.MultiDiGraph()
    for source, target, key, data in reversed(list(graph.edges(keys=True, data=True))):
        reversed_graph.add_edge(source, target, key=key, **data)
    assert module._expand_from_concepts(
        graph, ["A"], relation_limit_per_seed=2
    ) == module._expand_from_concepts(reversed_graph, ["A"], relation_limit_per_seed=2)


@pytest.mark.parametrize(
    "reason", ["missing_relation", "missing_evidence", "source_only"]
)
def test_invalid_relation_is_skipped_without_consuming_quota(database, reason):
    graph = build_concept_graph()
    with sqlite3.connect(database) as conn:
        if reason == "source_only":
            conn.execute(
                "UPDATE relation_evidence SET evidence_role='source' "
                "WHERE relation_id=7"
            )
        else:
            conn.execute("DELETE FROM relation_evidence WHERE relation_id=7")
            if reason == "missing_relation":
                conn.execute("DELETE FROM concept_relations WHERE id=7")
    hits = module._expand_from_concepts(graph, ["A"], relation_limit_per_seed=2)
    assert chunk_ids(hits) == [1, 2, 3, 6]


def test_self_loop_is_counted_once_across_both_directions(database):
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO concept_relations"
            "(id,source_concept,target_concept,relation,relation_type) "
            "VALUES(1,'A','A','uses','document_internal')"
        )
        conn.execute(
            "INSERT INTO relation_evidence"
            "(relation_id,document_id,document_chunk_id,evidence_role) "
            "VALUES(1,1,1,'both')"
        )
    hits = module._expand_from_concepts(
        build_concept_graph(), ["A"], relation_limit_per_seed=2
    )
    assert chunk_ids(hits) == [1, 4, 3, 5]


def test_missing_knowledge_unit_is_skipped(database):
    with sqlite3.connect(database) as conn:
        conn.execute(
            "DELETE FROM document_chunk_knowledge_units WHERE document_chunk_id=5"
        )
    assert chunk_ids(module._expand_from_concepts(build_concept_graph(), ["A"])) == [
        1,
        2,
        3,
        4,
        6,
    ]


def test_empty_graph_keeps_direct_candidates(database):
    assert chunk_ids(module._expand_from_concepts(nx.MultiDiGraph(), ["A"])) == [1, 3]


def test_seed_absent_from_graph_keeps_direct_candidates(database):
    graph = build_concept_graph()
    graph.remove_node("A")
    assert chunk_ids(module._expand_from_concepts(graph, ["A"])) == [1, 3]


@pytest.mark.parametrize("seeds", [[], ["missing"]])
def test_no_matching_seed_returns_empty_list(database, seeds):
    assert module._expand_from_concepts(build_concept_graph(), seeds) == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_relation_limit_is_rejected(database, limit):
    with pytest.raises(ValueError, match="relation_limit_per_seed"):
        module._expand_from_concepts(
            nx.MultiDiGraph(), ["A"], relation_limit_per_seed=limit
        )


def test_name_matching_is_case_insensitive_and_deduplicates_aliases(database):
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO document_chunk_concepts"
            "(document_id,document_chunk_id,concept,normalized_concept,"
            "normalized_concept_hash) VALUES(1,1,'alias','A','A')"
        )
    assert module._match_concepts("ALIAS a", min_similarity=0.5) == ["A"]
    assert module._match_concepts("a b", min_similarity=0.5, seed_limit=1) == ["A"]


@pytest.mark.parametrize(
    "name,query,expected_match",
    [
        ("AI", "explain indexing", False),
        ("AI", "chair", False),
        ("AI", "AI2", False),
        ("AI", "my_AI", False),
        ("AI", "AI_model", False),
        ("AI", "(ai), applications", True),
        ("AI", "介绍AI技术", True),
        ("专家", "狐狸型专家预测", True),
        ("C++", "介绍C++的用途", True),
        ("C++", "C++17", False),
        ("AI模型", "介绍AI模型", True),
        ("AI模型", "OpenAI模型", False),
        ("neural network", "a NEURAL NETWORK?", True),
        ("neural network", "neural networks", False),
    ],
)
def test_name_matching_respects_english_boundaries(
    database, monkeypatch, name, query, expected_match
):
    monkeypatch.setattr(
        module,
        "query_concepts_all",
        Mock(return_value=[SimpleNamespace(concept=name, normalized_concept=name)]),
    )
    collection = Mock()
    collection.count.return_value = 0
    get_collection = Mock(return_value=collection)
    monkeypatch.setattr(module, "get_existing_collection", get_collection)

    assert module._match_concepts(query) == ([name] if expected_match else [])
    if expected_match:
        get_collection.assert_not_called()
    else:
        get_collection.assert_called_once()


def test_english_substring_does_not_block_vector_fallback(database, monkeypatch):
    monkeypatch.setattr(
        module,
        "query_concepts_all",
        Mock(
            return_value=[
                SimpleNamespace(concept=name, normalized_concept=name)
                for name in ("AI", "索引")
            ]
        ),
    )
    collection = Mock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "documents": [["索引", "AI"]],
        "distances": [[0.1, 0.9]],
    }
    monkeypatch.setattr(
        module, "get_existing_collection", Mock(return_value=collection)
    )
    embedding = Mock(return_value=[0.1, 0.2])
    monkeypatch.setattr(module, "get_embedding", embedding)

    assert module._match_concepts("explain indexing") == ["索引"]
    embedding.assert_called_once_with("explain indexing")
    collection.query.assert_called_once()


def test_vector_fallback_filters_threshold_stale_names_and_duplicates(
    database, monkeypatch
):
    collection = Mock()
    collection.count.return_value = 6
    collection.query.return_value = {
        "documents": [["missing", "B", "B", "A", "D", "C"]],
        "distances": [[0.01, 0.1, 0.15, 0.25, 0.3, 0.6]],
    }
    monkeypatch.setattr(
        module, "get_existing_collection", Mock(return_value=collection)
    )
    monkeypatch.setattr(module, "get_embedding", Mock(return_value=[0.1, 0.2]))
    assert module._match_concepts(
        "无名称命中", min_similarity=0.75, vector_candidate_k=10, seed_limit=3
    ) == ["B", "A"]
    assert module._match_concepts(
        "无名称命中", min_similarity=0.75, vector_candidate_k=10, seed_limit=1
    ) == ["B"]
    assert collection.query.call_args.kwargs["n_results"] == 6
    assert collection.query.call_args.kwargs["query_embeddings"] == [0.1, 0.2]


def test_empty_concept_collection_does_not_request_embedding(database, monkeypatch):
    collection = Mock()
    collection.count.return_value = 0
    monkeypatch.setattr(
        module, "get_existing_collection", Mock(return_value=collection)
    )
    assert module._match_concepts("无名称命中", min_similarity=0.5) == []
    collection.query.assert_not_called()


def test_embedding_failure_is_not_an_empty_result(database, monkeypatch):
    collection = Mock()
    collection.count.return_value = 4
    monkeypatch.setattr(
        module, "get_existing_collection", Mock(return_value=collection)
    )
    monkeypatch.setattr(module, "get_embedding", Mock(return_value=None))
    with pytest.raises(RuntimeError, match="embedding"):
        module._match_concepts("无名称命中", min_similarity=0.5)


def test_concepts_from_hits_uses_rank_deduplicates_and_limits(database):
    def hit(chunk_id, rank, document_id=1):
        return SearchResult(
            "candidate",
            document_id,
            chunk_id,
            1.0,
            rank,
            SourceTypeEnum.CHROMA_KNOWLEDGE,
            RetrievalTypeEnum.VECTOR,
        )

    hits = [hit(6, 4), hit(1, 2), hit(2, 3), hit(None, 1), hit(1, 0, 99)]
    assert module._concepts_from_hits(hits, seed_limit=3) == ["A", "B", "D"]
    assert module._concepts_from_hits(hits, seed_limit=4) == ["A", "B", "D", "C"]
    assert module._concepts_from_hits([]) == []


@pytest.mark.parametrize("route", ["query", "hits"])
@pytest.mark.parametrize("save_session", [True, False])
@pytest.mark.parametrize("empty", [True, False])
def test_graph_wrappers_save_only_returned_results(
    database, route, save_session, empty
):
    graph = build_concept_graph()
    if route == "query":
        query = "" if empty else "A"
        result = module.graph_search_by_query(
            query,
            graph,
            min_similarity=0.5,
            top_k=2,
            save_session=save_session,
        )
        retrieval_type = RetrievalTypeEnum.GRAPH
        source_type = SourceTypeEnum.GRAPH_KNOWLEDGE
    else:
        query = "original question"
        hits = (
            []
            if empty
            else [
                SearchResult(
                    "base candidate",
                    1,
                    1,
                    0.9,
                    1,
                    SourceTypeEnum.CHROMA_KNOWLEDGE,
                    RetrievalTypeEnum.VECTOR,
                )
            ]
        )
        result = module.graph_search_by_hits(
            graph,
            hits,
            query=query,
            top_k=2,
            candidate_k=5,
            save_session=save_session,
        )
        retrieval_type = RetrievalTypeEnum.HYBRID
        source_type = SourceTypeEnum.HYBRID_KNOWLEDGE
    assert bool(result) is not empty
    assert len(result) == (0 if empty else 2)
    with connections.get_sqlite_read_connection() as conn:
        sessions = conn.execute("SELECT * FROM search_sessions").fetchall()
        rows = conn.execute("SELECT * FROM search_results ORDER BY rank").fetchall()
    if not save_session:
        assert sessions == []
        assert rows == []
        return
    assert len(sessions) == 1
    assert sessions[0]["query"] == query
    assert sessions[0]["result_count"] == len(result)
    assert len(rows) == len(result)
    for row, hit in zip(rows, result, strict=True):
        assert row["session_id"] == sessions[0]["session_id"]
        assert row["document_id"] == hit.document_id
        assert row["document_chunk_id"] == hit.document_chunk_id
        assert row["rank"] == hit.rank
        assert row["score"] == pytest.approx(hit.score)
        expected_raw = hit.score if hit.raw_score is None else hit.raw_score
        assert row["raw_score"] == pytest.approx(expected_raw)
        assert row["retrieval_type"] == retrieval_type.value
        assert row["source_type"] == source_type.value


@pytest.mark.parametrize("route", ["query", "hits"])
def test_graph_wrapper_failure_does_not_save_session(database, monkeypatch, route):
    monkeypatch.setattr(
        module,
        "_expand_from_concepts",
        Mock(side_effect=RuntimeError("expansion failed")),
    )
    with pytest.raises(RuntimeError, match="expansion failed"):
        if route == "query":
            module.graph_search_by_query(
                "A", build_concept_graph(), top_k=3, min_similarity=0.5
            )
        else:
            module.graph_search_by_hits(
                build_concept_graph(), [], top_k=3, query="question"
            )
    with connections.get_sqlite_read_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM search_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM search_results").fetchone()[0] == 0


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
@pytest.mark.parametrize("save_session", [True, False])
def test_graph_enhancement_saves_once_at_final_stage(
    database, monkeypatch, mode, save_session
):
    import src.search as dispatcher

    hits = [
        SearchResult(
            "base candidate",
            1,
            1,
            0.9,
            1,
            SourceTypeEnum.CHROMA_KNOWLEDGE,
            RetrievalTypeEnum.VECTOR,
        )
    ]
    backend = Mock(return_value=hits)
    name = {"vector": "search_vector", "bm25": "search_bm25", "hybrid": "hybrid_search"}
    monkeypatch.setattr(dispatcher, name[mode], backend)
    result = dispatcher.search(
        "  original question  ",
        retrieval_mode=mode,
        use_graph=True,
        save_session=save_session,
    )
    assert backend.call_args.kwargs["save_session"] is False
    with connections.get_sqlite_read_connection() as conn:
        sessions = conn.execute("SELECT * FROM search_sessions").fetchall()
        rows = conn.execute("SELECT * FROM search_results ORDER BY rank").fetchall()
    assert len(sessions) == int(save_session)
    if save_session:
        assert sessions[0]["query"] == "original question"
        assert sessions[0]["result_count"] == len(result)
        assert [row["document_chunk_id"] for row in rows] == chunk_ids(result)
    else:
        assert rows == []


@pytest.mark.parametrize("top_k", [1, 5])
def test_search_pure_graph_respects_final_limit_and_saved_count(database, top_k):
    from src.search import search

    result = search("A", retrieval_mode="graph", top_k=top_k)
    assert len(result) == top_k
    with connections.get_sqlite_read_connection() as conn:
        assert (
            conn.execute("SELECT result_count FROM search_sessions").fetchone()[0]
            == top_k
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM search_results").fetchone()[0] == top_k
        )


@pytest.mark.parametrize("entry", ["tool", "mcp"])
@pytest.mark.parametrize("mode,use_graph", [("graph", False), ("vector", True)])
def test_external_tools_reach_real_search_and_graph_without_saving(
    database, monkeypatch, entry, mode, use_graph
):
    import asyncio
    import json

    import src.search as dispatcher
    from src.tools.knowledge_retrieval_tools import search_knowledge

    base_hits = [
        SearchResult(
            "base candidate",
            1,
            1,
            0.9,
            1,
            SourceTypeEnum.CHROMA_KNOWLEDGE,
            RetrievalTypeEnum.VECTOR,
        )
    ]
    backend = Mock(return_value=base_hits)
    monkeypatch.setattr(dispatcher, "search_vector", backend)
    args = {"query": "A", "top_k": 2, "retrieval_mode": mode, "use_graph": use_graph}
    if entry == "tool":
        payload = json.loads(search_knowledge.invoke(args))
    else:
        from src.mcp_server import mcp

        response = asyncio.run(mcp.call_tool("search_knowledge", args))
        assert not response.is_error
        assert response.content
        payload = response.structured_content
        assert payload is not None
    assert payload["retrieval_mode"] == mode
    assert len(payload["hits"]) == 2
    assert [hit["rank"] for hit in payload["hits"]] == [1, 2]
    assert all(hit["document_id"] == 1 for hit in payload["hits"])
    if use_graph:
        assert backend.call_args.kwargs["top_k"] == 4
        assert backend.call_args.kwargs["save_session"] is False
    else:
        backend.assert_not_called()
    with connections.get_sqlite_read_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM search_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM search_results").fetchone()[0] == 0
