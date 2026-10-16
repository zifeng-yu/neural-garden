"""部署数据校验、失败传播与导出边界；无真实网络、无业务库操作。"""

import importlib.util
import json
import sqlite3
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.repository.create_table import create_tables
from src.util.getHashValue import get_hash_value

SPEC = importlib.util.spec_from_file_location(
    "ubuntu_runner", Path(__file__).parents[1] / "deploy/ubuntu/runner.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def sample(tmp_path):
    base = tmp_path
    run = base / "runs/one"
    project = run / "project"
    pilot = project / "data/pilot"
    pilot.mkdir(parents=True)
    text = "# 样本\n\n这是完整原文。"
    article = pilot / "01-sample.md"
    article.write_text(text)
    runner.write_json(
        run / "input-manifest.json", {article.name: runner.sha256(article)}
    )
    runner.write_json(run / "skipped-chunks.json", [])
    database = run / "data/sqlite/neural.db"
    database.parent.mkdir(parents=True)
    from src.document.splitter import markdown_spilt

    chunk = markdown_spilt(text)[0]
    with sqlite3.connect(database) as conn:
        create_tables(conn)
        conn.execute(
            "INSERT INTO documents(id,file_name,file_name_hash,content_hash) "
            "VALUES(1,?,?,?)",
            (article.name, get_hash_value(article.name), get_hash_value(text)),
        )
        conn.execute(
            "INSERT INTO document_chunks(id,document_id,split_no,content,content_hash) "
            "VALUES(1,1,1,?,?)",
            (chunk, get_hash_value(chunk)),
        )
        conn.execute(
            "INSERT INTO document_chunk_knowledge_units "
            "(id,document_id,document_chunk_id,title,summary,keywords,embedding_text) "
            "VALUES(1,1,1,'标题','摘要','[]','检索文本')"
        )
    return SimpleNamespace(
        base=base, run=run, project=project, db=database, chunk=chunk
    )


def test_complete_source_and_explicit_invalid_chunk(sample):
    data, skips = runner.validate_sqlite(sample.project, sample.run)
    assert len(data["document_chunk_knowledge_units"]) == 1 and not skips
    with sqlite3.connect(sample.db) as conn:
        conn.execute("DELETE FROM document_chunk_knowledge_units")
    with pytest.raises(RuntimeError, match="Missing knowledge"):
        runner.validate_sqlite(sample.project, sample.run)
    runner.write_json(
        sample.run / "skipped-chunks.json",
        [
            {
                "source": "filename:01-sample.md,splitNo:1",
                "content_hash": get_hash_value(sample.chunk),
            }
        ],
    )
    _, skips = runner.validate_sqlite(sample.project, sample.run)
    assert len(skips) == 1


def test_changed_input_is_rejected(sample):
    (sample.project / "data/pilot/01-sample.md").write_text("changed")
    with pytest.raises(RuntimeError, match="pilot files changed"):
        runner.validate_sqlite(sample.project, sample.run)


def test_cross_document_id_mismatch_is_rejected(sample):
    with sqlite3.connect(sample.db) as conn:
        conn.execute("UPDATE document_chunk_knowledge_units SET document_id=999")
    with pytest.raises(RuntimeError, match="foreign keys"):
        runner.validate_sqlite(sample.project, sample.run)


def test_embedding_failure_is_retried_then_propagated(monkeypatch):
    from dashscope import TextEmbedding

    calls = []

    def failure(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=503)

    monkeypatch.setattr(TextEmbedding, "call", failure)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        runner.embedding("原文")
    assert len(calls) == 3


def test_extraction_failures_are_recorded_and_skipped(tmp_path, monkeypatch):
    import src.indexer as indexer
    import src.knowledge.knowledge_extractor as extractor
    import src.knowledgeGraph.concepts as concepts
    import src.vector_store.query_dao as queries

    for module, name in [
        (indexer, "extract_knowledge_unit"),
        (indexer, "extract_relations_from_text"),
        (indexer, "get_embedding"),
        (concepts, "get_embedding"),
        (queries, "get_embedding"),
    ]:
        monkeypatch.setattr(module, name, getattr(module, name))

    def invalid(content):
        raise extractor.ContentNotExtrator("model rejected")

    monkeypatch.setattr(extractor, "extractor_by_llm", invalid)
    strict, skips = runner.install_strict_indexer(tmp_path)
    assert strict.extract_knowledge_unit("内容", "filename:a.md,splitNo:1") is None
    assert len(skips) == 1

    def technical(content):
        raise TimeoutError("network error")

    monkeypatch.setattr(extractor, "extractor_by_llm", technical)
    strict, skips = runner.install_strict_indexer(tmp_path)
    assert strict.extract_knowledge_unit("内容", "filename:a.md,splitNo:1") is None
    assert skips[0]["reason"] == "extraction_failed"
    assert skips[0]["error_type"] == "TimeoutError"


def test_package_preserves_wal_and_excludes_credentials(sample):
    runner.write_json(
        sample.run / "validation.json", {"status": "verified", "skipped_chunks": []}
    )
    runner.write_json(sample.run / "probe.json", {"status": "passed"})
    (sample.run / "commit.txt").write_text("test-commit")
    (sample.project / "uv.lock").write_text("test-lock")
    (sample.project / "config.yaml").write_text("server-paths")
    (sample.project / ".env").write_text("API_KEY=do-not-export")
    (sample.run / "data/chroma").mkdir()
    (sample.run / "data/chroma/segment.bin").write_bytes(b"segment")
    (sample.base / "exports").mkdir()
    (sample.run / "logs").mkdir()
    (sample.run / "logs/app.log").write_text("Embedding seconds=0.42\n")
    (sample.run / "logs/error.log").write_text("Knowledge extraction failed\n")
    conn = sqlite3.connect(sample.db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("UPDATE documents SET file_name_hash='committed-in-wal'")
        conn.commit()
        runner.package(sample.project, sample.run, sample.base)
    finally:
        conn.close()
    archive = next((sample.base / "exports").glob("*.tar.gz"))
    with tarfile.open(archive) as tar:
        assert not any(".env" in name or ".venv" in name for name in tar.getnames())
        assert "data/chroma/segment.bin" in tar.getnames()
        assert "SHA256SUMS" in tar.getnames()
        assert tar.extractfile("logs/app.log").read() == b"Embedding seconds=0.42\n"
        assert "logs/error.log" in tar.getnames()
        assert b"logs/app.log" in tar.extractfile("SHA256SUMS").read()
    with sqlite3.connect(sample.run / "export/data/sqlite/neural.db") as conn:
        assert (
            conn.execute("SELECT file_name_hash FROM documents").fetchone()[0]
            == "committed-in-wal"
        )
    assert (sample.run / "SUCCESS.txt").exists()


def test_failed_validation_cannot_produce_archive(sample):
    runner.write_json(sample.run / "validation.json", {"status": "failed"})
    with pytest.raises(RuntimeError, match="unverified"):
        runner.package(sample.project, sample.run, sample.base)
    assert not (sample.run / "SUCCESS.txt").exists()


def test_probe_failure_stops_before_embedding(tmp_path, monkeypatch):
    import src.util.callDashscopellm as llm

    def fail(*args):
        raise RuntimeError("HTTP 401")

    monkeypatch.setattr(llm, "generate", fail)
    monkeypatch.setattr(runner, "embedding", lambda _: pytest.fail("probe must stop"))
    with pytest.raises(RuntimeError, match="401"):
        runner.probe(tmp_path)
    assert not (tmp_path / "probe.json").exists()


@pytest.fixture
def fresh_build(tmp_path, monkeypatch):
    import src.get_chroma_collection as chroma
    import src.get_sqlite_connection as sqlite
    import src.indexer as indexer
    import src.knowledge.knowledge_extractor as extractor
    import src.knowledgeGraph.concepts as concepts
    import src.vector_store.query_dao as queries

    run = tmp_path / "runs/build"
    project = run / "project"
    pilot = project / "data/pilot"
    pilot.mkdir(parents=True)
    (pilot / "one.md").write_text("# 样本\n\n这是需要建立索引的原文。")
    runner.write_json(
        run / "input-manifest.json",
        {
            "one.md": runner.sha256(pilot / "one.md"),
        },
    )
    runner.write_json(run / "probe.json", {"status": "passed"})
    (run / "commit.txt").write_text("fixture-commit")
    monkeypatch.setattr(sqlite, "SQLITE_TABLE", str(run / "data/sqlite/neural.db"))
    monkeypatch.setattr(chroma, "PERSIST_DIRECTORY", str(run / "data/chroma"))
    for module, name in [
        (indexer, "extract_knowledge_unit"),
        (indexer, "get_embedding"),
        (concepts, "get_embedding"),
        (queries, "get_embedding"),
    ]:
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(indexer, "documents_to_concepts", lambda *args: ["样本"])
    monkeypatch.setattr(indexer, "extract_relations_from_text", lambda *args: [])
    monkeypatch.setattr(runner, "embedding", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        extractor,
        "extractor_by_llm",
        lambda text: SimpleNamespace(
            title="标题",
            summary="摘要",
            keywords=["样本"],
        ),
    )
    return SimpleNamespace(run=run, project=project, extractor=extractor, chroma=chroma)


def test_real_indexer_to_sqlite_and_chroma_passes_validation(fresh_build):
    runner.build(fresh_build.project, fresh_build.run)
    runner.verify(fresh_build.project, fresh_build.run)
    report = json.loads((fresh_build.run / "validation.json").read_text())
    assert report["status"] == "verified"
    assert report["counts"]["document_chunks"] == 1
    assert report["counts"]["document_chunk_knowledge_units"] == 1
    assert report["counts"]["document_chunk_concepts"] == 1


def test_missing_chroma_vector_cannot_pass_validation(fresh_build):
    from src.config.config import CHROMA_KNOWLEDGE_TABLE_NAME

    runner.build(fresh_build.project, fresh_build.run)
    collection = fresh_build.chroma.get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    collection.delete(ids=["1"])
    with pytest.raises(RuntimeError, match="ID mismatch"):
        runner.verify(fresh_build.project, fresh_build.run)
    assert not (fresh_build.run / "validation.json").exists()


@pytest.mark.parametrize("field", ["documents", "metadatas", "embeddings"])
def test_missing_chroma_payload_is_rejected(fresh_build, monkeypatch, field):
    import chromadb

    from src.config.config import CHROMA_KNOWLEDGE_TABLE_NAME

    runner.build(fresh_build.project, fresh_build.run)
    result = fresh_build.chroma.get_collection(CHROMA_KNOWLEDGE_TABLE_NAME).get(
        include=["documents", "metadatas", "embeddings"]
    )
    result[field] = None
    client = SimpleNamespace(
        get_collection=lambda name: SimpleNamespace(get=lambda **kwargs: result)
    )
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="Missing Chroma validation payload"):
        runner.verify(fresh_build.project, fresh_build.run)
    assert not (fresh_build.run / "validation.json").exists()


@pytest.mark.parametrize("failure_kind", ["timeout", "missing_keywords"])
def test_failed_extraction_keeps_chunk_and_finishes(
    fresh_build, monkeypatch, failure_kind
):
    def fail(text):
        if failure_kind == "timeout":
            raise TimeoutError("SDK timeout")
        return fresh_build.extractor.Response_extractor.model_validate(
            {"title": "标题", "summary": "摘要"}
        )

    monkeypatch.setattr(fresh_build.extractor, "extractor_by_llm", fail)
    runner.build(fresh_build.project, fresh_build.run)
    runner.verify(fresh_build.project, fresh_build.run)
    report = json.loads((fresh_build.run / "validation.json").read_text())
    assert report["status"] == "verified_with_skips"
    assert report["counts"]["document_chunks"] == 1
    assert report["counts"]["document_chunk_knowledge_units"] == 0
    assert report["skipped_chunks"][0]["reason"] == "extraction_failed"
    assert report["skipped_chunks"][0]["error_type"] == (
        "TimeoutError" if failure_kind == "timeout" else "ValidationError"
    )


def test_configure_uses_absolute_isolated_paths(tmp_path, monkeypatch):
    import yaml

    project = tmp_path / "runs/new/project"
    run = project.parent
    (project / "data/pilot").mkdir(parents=True)
    (project / "data/pilot/old-git-article.md").write_text("old Git article")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "one.md").write_text("article")
    monkeypatch.setenv("NG_PILOT_DIR", str(uploads))
    (tmp_path / ".env").write_text("API_KEY=private-test-key\n")
    (project / "config.yaml").write_text(
        (Path(__file__).parents[1] / "config.yaml").read_text()
    )
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3)
    )
    runner.configure(project, run, tmp_path)
    config = yaml.safe_load((project / "config.yaml").read_text())
    assert config["sqlite"]["sqlite_directory"] == str(run / "data/sqlite/neural.db")
    assert config["chroma"]["persist_directory"] == str(run / "data/chroma")
    assert config["pilot_dataset"]["path"] == str(project / "data/pilot")
    assert config["dashscope"]["llm_model"] == "qwen3.5-plus"
    assert not (project / ".env").exists()
    assert "private-test-key" not in (project / "config.yaml").read_text()
    assert sorted(p.name for p in (project / "data/pilot").iterdir()) == ["one.md"]
    assert list(runner.check_inputs(project, run)) == ["one.md"]
    (uploads / "one.md").write_text("later upload")
    assert (project / "data/pilot/one.md").read_text() == "article"
    assert list(runner.check_inputs(project, run)) == ["one.md"]
    with pytest.raises(RuntimeError, match="prepared input snapshots"):
        runner.configure(project, run, tmp_path)


@pytest.mark.parametrize("kind", ["missing", "empty", "blank", "symlink"])
def test_snapshot_rejects_invalid_uploads(tmp_path, kind):
    uploads = tmp_path / "uploads"
    if kind != "missing":
        uploads.mkdir()
    if kind == "blank":
        (uploads / "one.md").write_text("  \n")
    if kind == "symlink":
        (tmp_path / "original.md").write_text("article")
        (uploads / "one.md").symlink_to(tmp_path / "original.md")
    with pytest.raises(RuntimeError):
        runner.snapshot_pilot(tmp_path / "project", uploads)
    assert not (tmp_path / "project/data/pilot").exists()


def test_cli_failure_records_failed_status_and_redacts_key(sample, monkeypatch):
    monkeypatch.setenv("API_KEY", "private-test-key")
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "runner.py",
            "probe",
            "--project",
            str(sample.project),
            "--run",
            str(sample.run),
            "--base",
            str(sample.base),
        ],
    )

    def fail(*args):
        raise RuntimeError("request failed: private-test-key")

    monkeypatch.setattr(runner, "load_runtime", fail)
    assert runner.main() == 1
    status_text = (sample.run / "status.json").read_text()
    assert json.loads(status_text)["status"] == "failed"
    assert "private-test-key" not in status_text
    assert "private-test-key" not in (sample.run / "failure.json").read_text()
