"""统一检索入口只调用所选模式，并透传保存开关。"""

from unittest.mock import Mock

import pytest

import src.search as dispatcher
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.retrieval.search_result import SearchResult


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
@pytest.mark.parametrize("save_session", [True, False])
def test_dispatch_only_runs_selected_route(monkeypatch, mode, save_session):
    source_type, retrieval_type = {
        "vector": (SourceTypeEnum.CHROMA_KNOWLEDGE, RetrievalTypeEnum.VECTOR),
        "bm25": (SourceTypeEnum.ES_KNOWLEDGE, RetrievalTypeEnum.KEYWORD),
        "hybrid": (SourceTypeEnum.HYBRID_KNOWLEDGE, RetrievalTypeEnum.HYBRID),
    }[mode]
    hits = [SearchResult("摘要", 1, 10, 0.5, 1, source_type, retrieval_type)]
    names = {
        "vector": "search_vector",
        "bm25": "search_bm25",
        "hybrid": "hybrid_search",
    }
    routes = {}
    for candidate, name in names.items():
        routes[candidate] = Mock(
            return_value=hits,
            side_effect=None
            if candidate == mode
            else AssertionError("不能调用其他模式"),
        )
        monkeypatch.setattr(dispatcher, name, routes[candidate])
    result = dispatcher.search(
        " HashSet ",
        top_k=2,
        retrieval_mode=mode,
        save_session=save_session,
    )
    assert result is hits
    kwargs = routes[mode].call_args.kwargs
    assert kwargs["query"] == "HashSet"
    assert kwargs["save_session"] is save_session
    assert kwargs["candidate_k" if mode == "bm25" else "top_k"] == 2
    if mode == "hybrid":
        assert kwargs["candidate_k"] == 6
    for candidate, route in routes.items():
        if candidate != mode:
            route.assert_not_called()


def test_default_is_vector_and_preserves_embedding_failure(monkeypatch):
    vector = Mock(side_effect=RuntimeError("embedding失败"))
    monkeypatch.setattr(dispatcher, "search_vector", vector)
    with pytest.raises(RuntimeError, match="embedding失败"):
        dispatcher.search("问题", save_session=False)
    vector.assert_called_once_with(
        query="问题", top_k=3, show_score=True, save_session=False
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"query": " "}, {"top_k": 0}, {"top_k": True}, {"retrieval_mode": "unknown"}],
)
def test_invalid_dispatch_input_does_not_run_retrieval(monkeypatch, kwargs):
    routes = [Mock(side_effect=AssertionError("非法参数不能查询")) for _ in range(3)]
    for name, route in zip(
        ["search_vector", "search_bm25", "hybrid_search"], routes, strict=True
    ):
        monkeypatch.setattr(dispatcher, name, route)
    with pytest.raises(ValueError):
        dispatcher.search(**{"query": "问题", **kwargs})
    for route in routes:
        route.assert_not_called()


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_graph_enhancement_keeps_more_base_candidates(monkeypatch, mode):
    base_hits = [
        SearchResult(
            "摘要",
            1,
            10,
            0.5,
            1,
            SourceTypeEnum.CHROMA_KNOWLEDGE,
            RetrievalTypeEnum.VECTOR,
        )
    ]
    backend = Mock(return_value=base_hits)
    names = {
        "vector": "search_vector",
        "bm25": "search_bm25",
        "hybrid": "hybrid_search",
    }
    monkeypatch.setattr(dispatcher, names[mode], backend)
    graph = object()
    monkeypatch.setattr(dispatcher, "build_concept_graph", Mock(return_value=graph))
    final_hits = []
    enhance = Mock(return_value=final_hits)
    monkeypatch.setattr(dispatcher, "graph_search_by_hits", enhance)
    assert (
        dispatcher.search("问题", top_k=5, retrieval_mode=mode, use_graph=True)
        is final_hits
    )
    kwargs = backend.call_args.kwargs
    assert kwargs["candidate_k" if mode == "bm25" else "top_k"] == 10
    assert kwargs["save_session"] is False
    if mode == "hybrid":
        assert kwargs["candidate_k"] == 15
    enhance.assert_called_once_with(
        graph, base_hits, query="问题", save_session=True, top_k=5
    )


@pytest.mark.parametrize("mode", ["graph", "hybrid"])
@pytest.mark.parametrize("use_graph", [False, True])
def test_cli_forwards_mode_and_graph_flag(monkeypatch, mode, use_graph):
    import sys

    backend = Mock(return_value=[])
    monkeypatch.setattr(dispatcher, "search", backend)
    monkeypatch.setattr(dispatcher, "setup_logging", Mock())
    args = ["search", "固定问题", "--mode", mode, "--top-k", "2", "--no-save-session"]
    if use_graph:
        args.append("--use-graph")
    monkeypatch.setattr(sys, "argv", args)
    dispatcher.main()
    backend.assert_called_once_with(
        "固定问题",
        top_k=2,
        retrieval_mode=mode,
        save_session=False,
        use_graph=use_graph,
    )
