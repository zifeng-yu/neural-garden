import importlib.util
import json
import sqlite3
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import indexer_two_stage as staged

_SPEC = importlib.util.spec_from_file_location(
    "two_stage_runner", Path(__file__).parents[1] / "deploy/ubuntu/runner.py"
)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


@pytest.fixture
def job(tmp_path, monkeypatch):
    import src.get_chroma_collection as chroma
    import src.get_sqlite_connection as sqlite
    import src.knowledgeGraph.concepts as concepts
    import src.vector_store.query_dao as queries

    run = tmp_path / "runs/job"
    project = run / "project"
    pilot = project / "data/pilot"
    pilot.mkdir(parents=True)
    (pilot / "01-one.md").write_text("# First\n\nAlpha and Beta are connected.")
    (pilot / "02-two.md").write_text("# Second\n\nAlias and Beta are connected.")
    (project / "config.yaml").write_text("configuration")
    (project / "uv.lock").write_text("lock")
    (run / "commit.txt").write_text("commit-a")
    (run / "logs").mkdir()
    (run / "logs/app.log").write_text("phase1 logs\n")
    (tmp_path / "exports").mkdir()
    staged.write_json(
        run / "input-manifest.json",
        {p.name: staged.digest(p) for p in sorted(pilot.glob("*.md"))},
    )
    staged.write_json(run / "probe.json", {"status": "passed"})
    monkeypatch.setattr(sqlite, "SQLITE_TABLE", str(run / "data/sqlite/neural.db"))
    monkeypatch.setattr(chroma, "PERSIST_DIRECTORY", str(run / "data/chroma"))
    calls = []

    def extract(content):
        calls.append(content)
        return SimpleNamespace(title="Title", summary="Summary", keywords=["topic"])

    def vector(text):
        return [0.0, 1.0, 0.0] if text == "Beta" else [1.0, 0.0, 0.0]

    monkeypatch.setattr(staged, "extractor_by_llm", extract)
    monkeypatch.setattr(
        staged.pipeline,
        "documents_to_concepts",
        lambda title, content: ["Alias" if "Second" in content else "Alpha", "Beta"],
    )
    monkeypatch.setattr(
        staged.pipeline, "concept_clustering_in_document", lambda chunks: None
    )
    monkeypatch.setattr(staged.pipeline, "get_embedding", vector)
    monkeypatch.setattr(concepts, "get_embedding", vector)
    monkeypatch.setattr(queries, "get_embedding", vector)
    monkeypatch.setattr(
        concepts,
        "extract_max_similarity_concept",
        lambda concept, similarity_concepts: (
            [similarity_concepts[0][0]] if similarity_concepts else []
        ),
    )
    monkeypatch.setattr(
        staged.pipeline,
        "extract_relations_from_text",
        lambda content, names: (
            [(names[0], "related_to", names[1])] if len(names) >= 2 else []
        ),
    )
    return SimpleNamespace(
        base=tmp_path, run=run, project=project, pilot=pilot, calls=calls
    )


def test_concurrent_phase1_never_opens_database_and_stops_for_review(job, monkeypatch):
    import src.get_chroma_collection as chroma

    barrier = threading.Barrier(2)
    original = staged.prepare_document

    def prepare(path):
        barrier.wait(timeout=5)
        return original(path)

    monkeypatch.setattr(staged, "prepare_document", prepare)
    monkeypatch.setattr(
        chroma, "get_collection", lambda *a: pytest.fail("phase1 opened Chroma")
    )
    monkeypatch.setattr(
        staged.pipeline,
        "sqlite_table_create_if_not",
        lambda: pytest.fail("phase1 opened SQLite"),
    )
    staged.phase1(job.project, job.run, workers=2)
    runner.phase1_package(job.project, job.run, job.base)
    assert not (job.run / "data").exists()
    assert not (job.run / "SUCCESS.txt").exists()
    assert (job.run / "PHASE1_READY.txt").exists()
    assert (
        json.loads((job.run / "status.json").read_text())["status"]
        == "awaiting_phase1_review"
    )
    report = json.loads((job.run / "phase1-report.json").read_text())
    assert report["order"] == ["01-one.md", "02-two.md"]
    with tarfile.open(next((job.base / "exports").glob("*.tar.gz"))) as archive:
        assert "logs/app.log" in archive.getnames()
        assert "phase1/0001.json" in archive.getnames()
        assert not any(".env" in n for n in archive.getnames())


def test_phase2_uses_prepared_data_and_previous_document_concepts(job, monkeypatch):
    staged.phase1(job.project, job.run)
    runner.phase1_package(job.project, job.run, job.base)
    assert len(job.calls) == 2
    monkeypatch.setattr(
        staged,
        "extractor_by_llm",
        lambda text: pytest.fail("phase2 re-extracted knowledge"),
    )
    monkeypatch.setattr(
        staged.pipeline,
        "documents_to_concepts",
        lambda *a: pytest.fail("phase2 re-extracted concepts"),
    )
    monkeypatch.setattr(
        staged.pipeline,
        "concept_clustering_in_document",
        lambda *a: pytest.fail("phase2 repeated internal normalization"),
    )
    staged.phase2(job.project, job.run)
    runner.verify(job.project, job.run)
    runner.package(job.project, job.run, job.base)
    with sqlite3.connect(job.run / "data/sqlite/neural.db") as conn:
        assert conn.execute(
            "SELECT file_name FROM documents ORDER BY id"
        ).fetchall() == [("01-one.md",), ("02-two.md",)]
        concepts = conn.execute(
            "SELECT normalized_concept FROM document_chunk_concepts WHERE document_id=2"
        ).fetchall()
        assert set(concepts) == {("Alpha",), ("Beta",)}
        assert conn.execute("SELECT count(*) FROM relation_evidence").fetchone()[0] == 2
    assert (job.run / "SUCCESS.txt").exists()
    assert len(job.calls) == 2
    with pytest.raises(RuntimeError, match="already exists"):
        staged.phase2(job.project, job.run)


@pytest.mark.parametrize("changed", ["artifact", "source", "config", "skip"])
def test_phase2_rejects_modified_inputs_before_writing(job, changed):
    staged.phase1(job.project, job.run)
    if changed == "artifact":
        with (job.run / "phase1/0001.json").open("a") as f:
            f.write(" ")
    elif changed == "source":
        (job.pilot / "01-one.md").write_text("modified")
    elif changed == "config":
        (job.project / "config.yaml").write_text("modified")
    else:
        staged.write_json(job.run / "skipped-chunks.json", [{"source": "fake"}])
    with pytest.raises(RuntimeError):
        staged.phase2(job.project, job.run)
    assert not (job.run / "data").exists()


def test_knowledge_failure_is_preserved_across_both_phases(job, monkeypatch):
    def fail(text):
        raise ValueError("missing keywords")

    monkeypatch.setattr(staged, "extractor_by_llm", fail)
    staged.phase1(job.project, job.run)
    staged.phase2(job.project, job.run)
    runner.verify(job.project, job.run)
    report = json.loads((job.run / "validation.json").read_text())
    assert report["status"] == "verified_with_skips"
    assert report["counts"]["document_chunks"] == 2
    assert report["counts"]["document_chunk_knowledge_units"] == 0
    assert len(report["skipped_chunks"]) == 2


def test_phase1_fatal_failure_exports_logs_and_blocks_phase2(job, monkeypatch):
    def fail(*args):
        raise RuntimeError("concept extraction failed")

    monkeypatch.setattr(staged.pipeline, "documents_to_concepts", fail)
    with pytest.raises(RuntimeError, match="Phase 1 failed"):
        staged.phase1(job.project, job.run)
    runner.phase1_package(job.project, job.run, job.base)
    assert not (job.run / "PHASE1_READY.txt").exists()
    assert list((job.base / "exports").glob("*-failed.tar.gz"))
    with pytest.raises(RuntimeError, match="incomplete"):
        staged.phase2(job.project, job.run)
    assert not (job.run / "data").exists()


def test_phase2_requires_review_package(job, monkeypatch):
    staged.phase1(job.project, job.run)
    monkeypatch.setattr(runner, "load_runtime", lambda *args: None)
    # Restore aliases changed by staged_runtime after this test.
    import src.knowledgeGraph.concepts as concepts
    import src.vector_store.query_dao as queries

    for module in (staged.pipeline, concepts, queries):
        monkeypatch.setattr(module, "get_embedding", module.get_embedding)
    with pytest.raises(RuntimeError, match="review package"):
        runner.staged_runtime(job.project, job.run, job.base, "phase2", 2)
    assert not (job.run / "data").exists()


def test_phase2_rejects_changed_review_report(job, monkeypatch):
    staged.phase1(job.project, job.run)
    runner.phase1_package(job.project, job.run, job.base)
    with (job.run / "phase1-report.json").open("a") as f:
        f.write(" ")
    monkeypatch.setattr(runner, "load_runtime", lambda *args: None)
    import src.knowledgeGraph.concepts as concepts
    import src.vector_store.query_dao as queries

    for module in (staged.pipeline, concepts, queries):
        monkeypatch.setattr(module, "get_embedding", module.get_embedding)
    with pytest.raises(RuntimeError, match="report changed"):
        runner.staged_runtime(job.project, job.run, job.base, "phase2", 2)
    assert not (job.run / "data").exists()
