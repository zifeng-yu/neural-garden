"""Isolated Ubuntu driver: no ES, exported secrets, or live-database reset."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import sqlite3
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, cast


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure(project, run, base):
    import yaml
    from dotenv import dotenv_values

    if not dotenv_values(base / ".env").get("API_KEY"):
        raise RuntimeError("The private .env must contain a non-empty API_KEY.")
    if (run / "data").exists():
        raise RuntimeError("Refusing to prepare over an existing dataset.")
    config_path = project / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["dashscope"]["llm_model"] = "qwen3.5-plus"
    config["dashscope"]["enable_thinking"] = False
    config["pilot_dataset"]["path"] = str(project / "data/pilot")
    config["sqlite"]["sqlite_directory"] = str(run / "data/sqlite/neural.db")
    config["chroma"]["persist_directory"] = str(run / "data/chroma")
    config["logging"]["directory"] = str(run / "logs")
    # Retain the whole run for offline evaluation, including LLM retry/timing logs.
    config["logging"]["level"] = "DEBUG"
    config["logging"]["max_bytes"] = 0
    # Only the exported run copy is changed; the Git source cache stays intact.
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    files = sorted((project / "data/pilot").glob("*.md"))
    if not files or any(not p.read_text().strip() for p in files):
        raise RuntimeError("pilot must contain non-empty UTF-8 Markdown articles.")
    if any(p.is_symlink() for p in files):
        raise RuntimeError("pilot must contain regular files, not symbolic links.")
    free = shutil.disk_usage(run).free
    if free < 5 * 1024**3:
        raise RuntimeError(
            "At least 5 GiB free disk space is required before indexing."
        )
    write_json(run / "input-manifest.json", {p.name: sha256(p) for p in files})
    write_json(run / "status.json", {"status": "prepared", "articles": len(files)})
    print(f"Prepared {len(files)} articles; all data paths are under {run}.")


def load_runtime(project, run, base):
    from dotenv import dotenv_values

    key = dotenv_values(base / ".env").get("API_KEY")
    if not key:
        raise RuntimeError("Missing API_KEY in the private .env.")
    os.environ["API_KEY"] = key
    sys.path.insert(0, str(project))
    from src.config.logging_config import setup_logging

    setup_logging()

    class RedactingFormatter(logging.Formatter):
        def __init__(self, original, secret: str):
            super().__init__()
            self.original = original or logging.Formatter()
            self.secret = secret

        def format(self, record):
            return self.original.format(record).replace(self.secret, "[REDACTED]")

    for handler in logging.getLogger().handlers:
        handler.setFormatter(RedactingFormatter(handler.formatter, key))

    import dashscope
    from dashscope import MultiModalConversation, TextEmbedding

    dashscope.api_key = key
    # DashScope's SDK consumes `timeout` as an HTTP setting, not a model parameter.
    for cls in (MultiModalConversation, TextEmbedding):
        original = cls.call

        def bounded_call(*args, _original=original, **kwargs):
            kwargs.setdefault("timeout", 180)
            return _original(*args, **kwargs)

        # This intentional runtime replacement preserves each SDK call's arguments;
        # the two SDK methods have different static signatures and return types.
        cast(Any, cls).call = staticmethod(bounded_call)


def embedding(text):
    """Fail visibly after bounded retries; never silently leave a missing vector."""
    from dashscope import TextEmbedding

    from src.config.config import EMBEDDING_MODEL

    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            response = TextEmbedding.call(model=EMBEDDING_MODEL, input=text)
            if response.status_code != 200:
                raise RuntimeError(f"Embedding HTTP {response.status_code}")
            vector = response.output["embeddings"][0]["embedding"]
            if not vector or not all(math.isfinite(float(x)) for x in vector):
                raise ValueError("Embedding returned an empty or non-finite vector")
            logging.info(
                "Embedding model=%s attempt=%d seconds=%.2f chars=%d",
                EMBEDDING_MODEL,
                attempt,
                time.perf_counter() - started,
                len(text),
            )
            return vector
        except Exception:
            logging.exception(
                "Embedding failed model=%s attempt=%d seconds=%.2f chars=%d",
                EMBEDDING_MODEL,
                attempt,
                time.perf_counter() - started,
                len(text),
            )
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError("Unreachable embedding retry state")


def probe(run):
    from src.util.callDashscopellm import generate

    started = time.perf_counter()
    reply = generate("You are a connection test. Reply only OK.", "Reply OK.")
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("LLM connectivity test returned no text")
    llm_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vector = embedding("Neural Garden connection test")
    write_json(
        run / "probe.json",
        {
            "status": "passed",
            "llm_model": "qwen3.5-plus",
            "llm_seconds": round(llm_seconds, 3),
            "embedding_seconds": round(time.perf_counter() - started, 3),
            "embedding_dimensions": len(vector),
        },
    )
    write_json(run / "status.json", {"status": "probe_passed"})
    print("PASS: qwen3.5-plus + embedding HTTP calls and responses.")


def install_strict_indexer(run):
    import src.indexer as indexer
    import src.knowledgeGraph.concepts as concepts
    import src.vector_store.query_dao as queries
    from src.knowledge.knowledge_extractor import ContentNotExtrator, extractor_by_llm
    from src.knowledge.knowledgeUnit import KnowledgeUnit
    from src.util.getHashValue import get_hash_value

    skips = []

    def extract(content, id):
        try:
            result = extractor_by_llm(content)
            return KnowledgeUnit(
                result.title, result.summary, result.keywords, content, id
            )
        except Exception as exc:
            # Match the normal indexer: keep the original chunk and continue.
            # Record the reason so export validation can distinguish known skips
            # from unexplained missing knowledge units. Never serialize SDK errors
            # verbatim here: they may contain credentials or request contents.
            skips.append(
                {
                    "source": id,
                    "content_hash": get_hash_value(content),
                    "reason": "invalid_content"
                    if isinstance(exc, ContentNotExtrator)
                    else "extraction_failed",
                    "error_type": type(exc).__name__,
                }
            )
            write_json(run / "skipped-chunks.json", skips)
            logging.exception(
                "Knowledge extraction failed; keeping chunk and continuing: %s", id
            )
            return None

    original_relations = indexer.extract_relations_from_text

    def relations(*args, **kwargs):
        result = original_relations(*args, **kwargs)
        if result is None:
            raise ValueError("Relation extraction returned invalid data")
        return result

    # These are the actual imported aliases used in the indexing call chain.
    indexer.extract_knowledge_unit = extract
    indexer.extract_relations_from_text = relations
    indexer.get_embedding = embedding
    concepts.get_embedding = embedding
    queries.get_embedding = embedding
    write_json(run / "skipped-chunks.json", skips)
    return indexer, skips


def check_inputs(project, run):
    expected = json.loads((run / "input-manifest.json").read_text())
    actual = {p.name: sha256(p) for p in sorted((project / "data/pilot").glob("*.md"))}
    if actual != expected:
        raise RuntimeError("pilot files changed during this run")
    return expected


def build(project, run):
    if not (run / "probe.json").exists():
        raise RuntimeError("Network probe must pass before indexing")
    if (run / "data").exists():
        raise RuntimeError("A partial run cannot be resumed via file-hash skipping.")
    files = check_inputs(project, run)
    indexer, skips = install_strict_indexer(run)
    indexer.sqlite_table_create_if_not()
    # Include an empty insight collection for a complete, portable local dataset.
    from src.config.config import (
        CHROMA_CONCEPT_TABLE_NAME,
        CHROMA_INSIGHT_TABLE_NAME,
        CHROMA_KNOWLEDGE_TABLE_NAME,
    )
    from src.get_chroma_collection import get_collection

    # All chunks may legitimately lack knowledge, leaving empty collections.
    for name in (
        CHROMA_INSIGHT_TABLE_NAME,
        CHROMA_KNOWLEDGE_TABLE_NAME,
        CHROMA_CONCEPT_TABLE_NAME,
    ):
        get_collection(name)
    completed: list[str] = []
    for name in files:
        write_json(
            run / "status.json",
            {
                "status": "indexing",
                "current": name,
                "completed": completed,
                "total": len(files),
                "skipped_chunks": len(skips),
            },
        )
        started = time.perf_counter()
        indexer.index_file(str(project / "data/pilot" / name))
        completed.append(name)
        logging.info(
            "Document complete: %s seconds=%.2f", name, time.perf_counter() - started
        )
    write_json(
        run / "status.json",
        {
            "status": "indexed",
            "completed": completed,
            "total": len(files),
            "skipped_chunks": len(skips),
        },
    )


def validate_sqlite(project, run):
    from src.document.splitter import markdown_spilt
    from src.util.getHashValue import get_hash_value

    expected = check_inputs(project, run)
    path = run / "data/sqlite/neural.db"
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity_check failed")
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("SQLite contains broken foreign keys")
        tables = [
            "documents",
            "document_chunks",
            "document_chunk_knowledge_units",
            "document_chunk_concepts",
            "concept_relations",
            "relation_evidence",
        ]
        data = {
            t: [dict(r) for r in conn.execute("SELECT * FROM " + t)] for t in tables
        }
    finally:
        conn.close()
    docs = data["documents"]
    if len(docs) != len(expected) or {d["file_name"] for d in docs} != set(expected):
        raise RuntimeError("SQLite document set differs from pilot")
    chunks = {c["id"]: c for c in data["document_chunks"]}
    for doc in docs:
        text = (project / "data/pilot" / doc["file_name"]).read_text()
        actual = sorted(
            (c for c in chunks.values() if c["document_id"] == doc["id"]),
            key=lambda c: c["split_no"],
        )
        split = markdown_spilt(text)
        if doc["content_hash"] != get_hash_value(text) or len(split) != len(actual):
            raise RuntimeError(f"Document/chunk mismatch: {doc['file_name']}")
        for number, (stored, content) in enumerate(zip(actual, split, strict=True), 1):
            if (
                stored["split_no"] != number
                or stored["content"] != content
                or stored["content_hash"] != get_hash_value(content)
            ):
                raise RuntimeError(f"Chunk mismatch: {doc['file_name']} #{number}")
    for table in (
        "document_chunk_knowledge_units",
        "document_chunk_concepts",
        "relation_evidence",
    ):
        for row in data[table]:
            chunk = chunks.get(row["document_chunk_id"])
            if not chunk or chunk["document_id"] != row["document_id"]:
                raise RuntimeError(f"Broken document/chunk identity in {table}")
    skips = json.loads((run / "skipped-chunks.json").read_text())
    rejected = {(r["source"], r["content_hash"]) for r in skips}
    by_doc = {d["id"]: d["file_name"] for d in docs}
    units = data["document_chunk_knowledge_units"]
    covered = {k["document_chunk_id"] for k in units}
    missing = {
        (
            f"filename:{by_doc[c['document_id']]},splitNo:{c['split_no']}",
            c["content_hash"],
        )
        for c in chunks.values()
        if c["id"] not in covered
    }
    if missing != rejected:
        raise RuntimeError(
            "Missing knowledge units do not match recorded extraction skips"
        )
    for unit in units:
        if any(not unit[k].strip() for k in ("title", "summary", "embedding_text")):
            raise RuntimeError("Empty knowledge-unit fields")
        if not isinstance(json.loads(unit["keywords"]), list):
            raise RuntimeError("Invalid knowledge-unit keywords")
    concepts = data["document_chunk_concepts"]
    for concept in concepts:
        if not concept["normalized_concept"].strip() or concept[
            "normalized_concept_hash"
        ] != get_hash_value(concept["normalized_concept"]):
            raise RuntimeError("Invalid concept name/hash")
    relation_by_id = {r["id"]: r for r in data["concept_relations"]}
    for evidence in data["relation_evidence"]:
        relation = relation_by_id[evidence["relation_id"]]
        names = {
            c["normalized_concept"]
            for c in concepts
            if c["document_chunk_id"] == evidence["document_chunk_id"]
        }
        if (
            evidence["evidence_role"] != "both"
            or not {relation["source_concept"], relation["target_concept"]} <= names
        ):
            raise RuntimeError("Relation evidence does not match chunk concepts")
    if set(relation_by_id) != {e["relation_id"] for e in data["relation_evidence"]}:
        raise RuntimeError("Relations without evidence")
    return data, skips


def verify(project, run):
    import chromadb

    from src.config.config import CHROMA_CONCEPT_TABLE_NAME, CHROMA_KNOWLEDGE_TABLE_NAME

    status = json.loads((run / "status.json").read_text())
    if status["status"] != "indexed":
        raise RuntimeError("Cannot verify a run that did not finish indexing")
    data, skips = validate_sqlite(project, run)
    client = chromadb.PersistentClient(path=str(run / "data/chroma"))
    for collection_name, table, concept in [
        (CHROMA_KNOWLEDGE_TABLE_NAME, "document_chunk_knowledge_units", False),
        (CHROMA_CONCEPT_TABLE_NAME, "document_chunk_concepts", True),
    ]:
        result = client.get_collection(collection_name).get(
            include=["documents", "metadatas", "embeddings"]
        )
        expected = {
            str(r["normalized_concept_hash"] if concept else r["id"]): r
            for r in data[table]
        }
        if set(result["ids"]) != set(expected):
            raise RuntimeError(f"SQLite/Chroma ID mismatch: {collection_name}")
        documents = result["documents"]
        metadatas = result["metadatas"]
        embeddings = result["embeddings"]
        # Chroma types these fields as optional even when explicitly requested.
        # Empty collections need no payload; populated collections must supply it.
        if not result["ids"]:
            continue
        if documents is None or metadatas is None or embeddings is None:
            raise RuntimeError(f"Missing Chroma validation payload: {collection_name}")
        if any(
            len(values) != len(result["ids"])
            for values in (documents, metadatas, embeddings)
        ):
            raise RuntimeError(
                f"Incomplete Chroma validation payload: {collection_name}"
            )
        for i, key in enumerate(result["ids"]):
            row = expected[key]
            if (
                documents[i]
                != row["normalized_concept" if concept else "embedding_text"]
            ):
                raise RuntimeError(f"Chroma text mismatch: {collection_name}/{key}")
            if not concept:
                meta = metadatas[i]
                if meta is None or any(
                    meta.get(k) != row[k] for k in ("document_id", "document_chunk_id")
                ):
                    raise RuntimeError("Chroma knowledge identity mismatch")
            vector = embeddings[i]
            if len(vector) == 0 or not all(math.isfinite(float(x)) for x in vector):
                raise RuntimeError("Invalid stored embedding")
    report = {
        "status": "verified_with_skips" if skips else "verified",
        "counts": {k: len(v) for k, v in data.items()},
        "skipped_chunks": skips,
        "commit": (run / "commit.txt").read_text().strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    }
    write_json(run / "validation.json", report)
    write_json(run / "status.json", {"status": report["status"]})
    print(f"Verification passed; recorded knowledge skips: {len(skips)}")


def package(project, run, base):
    report = json.loads((run / "validation.json").read_text())
    if report["status"] not in ("verified", "verified_with_skips"):
        raise RuntimeError("Refusing to export an unverified dataset")
    check_inputs(project, run)
    # No writer remains: index and verify ran as separate processes.
    # SQLite backup captures any WAL content in a standalone portable database.
    export = run / "export"
    export.mkdir()
    (export / "data/sqlite").mkdir(parents=True)
    source = sqlite3.connect(
        (run / "data/sqlite/neural.db").resolve().as_uri() + "?mode=ro", uri=True
    )
    target = sqlite3.connect(export / "data/sqlite/neural.db")
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    shutil.copytree(run / "data/chroma", export / "data/chroma")
    shutil.copytree(project / "data/pilot", export / "data/pilot")
    # Index/verify have exited, so their redacted logs are closed and stable.
    if (run / "logs").is_dir():
        shutil.copytree(run / "logs", export / "logs")
    for name in (
        "validation.json",
        "input-manifest.json",
        "skipped-chunks.json",
        "probe.json",
        "commit.txt",
    ):
        shutil.copy2(run / name, export / name)
    shutil.copy2(project / "uv.lock", export / "uv.lock")
    shutil.copy2(project / "config.yaml", export / "server-config.yaml")
    files = sorted(p for p in export.rglob("*") if p.is_file())
    (export / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(export)}\n" for p in files)
    )
    suffix = "-with-skips" if report["skipped_chunks"] else ""
    archive = base / "exports" / f"neural-garden-{run.name}{suffix}.tar.gz"
    temporary = archive.with_suffix(archive.suffix + ".partial")
    with tarfile.open(temporary, "w:gz") as tar:
        for path in sorted(export.iterdir()):
            tar.add(path, arcname=path.name)
    temporary.replace(archive)
    checksum = sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{checksum}  {archive.name}\n"
    )
    (run / "SUCCESS.txt").write_text(
        f"Archive: {archive}\nSHA256: {checksum}\n"
        f"Recorded knowledge skips: {len(report['skipped_chunks'])}; "
        "see skipped-chunks.json.\n"
        "This is a NEW corpus. Back up local data, replace SQLite and the entire "
        "Chroma directory, "
        "then RESET the local ES index; do not merge directories or use ES upsert.\n"
    )
    write_json(
        run / "status.json",
        {
            "status": "complete",
            "archive": str(archive),
            "skipped_chunks": len(report["skipped_chunks"]),
        },
    )
    print((run / "SUCCESS.txt").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "probe", "index", "verify", "package"]
    )
    for name in ("project", "run", "base"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    project, run, base = args.project.resolve(), args.run.resolve(), args.base.resolve()
    try:
        if args.action == "prepare":
            configure(project, run, base)
        elif args.action == "package":
            package(project, run, base)
        else:
            load_runtime(project, run, base)
            {
                "probe": lambda: probe(run),
                "index": lambda: build(project, run),
                "verify": lambda: verify(project, run),
            }[args.action]()
    except Exception as exc:
        key = os.environ.get("API_KEY", "")
        message = f"{type(exc).__name__}: {exc}"
        if key:
            message = message.replace(key, "[REDACTED]")
        write_json(run / "failure.json", {"action": args.action, "error": message})
        write_json(
            run / "status.json",
            {
                "status": "failed",
                "action": args.action,
                "error": message,
            },
        )
        print(f"FAILED [{args.action}]: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
