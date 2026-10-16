"""Two-phase indexing: concurrent document preparation, then serial publication.

The original src.indexer entry point is unchanged. Stage 1 never opens SQLite or
Chroma. Stage 2 reads immutable prepared files and publishes documents in filename
order, so each document sees the standard concepts from previous documents.
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from pydantic import TypeAdapter

from src import indexer as pipeline
from src.knowledge.knowledge_extractor import ContentNotExtrator, extractor_by_llm
from src.knowledge.knowledgeUnit import KnowledgeUnit
from src.repository.document_insert_domain import (
    InsertChunk,
    InsertConcepts,
    InsertDoc,
    InsertDocumentDomain,
    InsertKnowledgeUnit,
)
from src.util.getHashValue import get_hash_value
from src.util.index_logging import index_stage

logger = logging.getLogger(__name__)
_DOMAIN = TypeAdapter(InsertDocumentDomain)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def inputs(project: Path, run: Path) -> dict[str, str]:
    expected = json.loads((run / "input-manifest.json").read_text())
    actual = {p.name: digest(p) for p in sorted((project / "data/pilot").glob("*.md"))}
    if not expected or actual != expected:
        raise RuntimeError("Pilot inputs changed or are empty")
    return dict(sorted(expected.items()))


def prepare_document(path: Path) -> dict:
    """Each worker owns its chunks/skips; no shared database or JSON-file writes."""
    started = perf_counter()
    text = pipeline.load_document(str(path))
    document = InsertDoc(path.name, get_hash_value(path.name), get_hash_value(text))
    chunks = []
    skips: list[dict] = []
    with index_stage(logger, "phase1_document", source=path.name) as stats:
        for number, content in enumerate(pipeline.markdown_spilt(text), 1):
            chunk = InsertChunk(number, content, get_hash_value(content))
            source = f"filename:{path.name},splitNo:{number}"
            with index_stage(
                logger, "knowledge_extract", chunk_no=number
            ) as extraction:
                try:
                    result = extractor_by_llm(content)
                    knowledge = KnowledgeUnit(
                        result.title, result.summary, result.keywords, content, source
                    )
                    chunk.know_unit = InsertKnowledgeUnit(
                        result.title,
                        result.summary,
                        result.keywords,
                        knowledge.to_embedding_text(),
                    )
                except Exception as exc:
                    extraction["outcome"] = "skipped"
                    skips.append(
                        {
                            "source": source,
                            "content_hash": chunk.content_hash,
                            "reason": "invalid_content"
                            if isinstance(exc, ContentNotExtrator)
                            else "extraction_failed",
                            "error_type": type(exc).__name__,
                        }
                    )
                    logger.exception(
                        "Knowledge extraction failed; retaining chunk: %s", source
                    )
            if chunk.know_unit is not None:
                with index_stage(
                    logger, "concept_extract", chunk_no=number
                ) as concept_stats:
                    names = pipeline.documents_to_concepts(
                        chunk.know_unit.title, content
                    )
                    chunk.concepts = [
                        InsertConcepts(n, n, get_hash_value(n)) for n in (names or [])
                    ]
                    concept_stats["concept_count"] = len(chunk.concepts)
            chunks.append(chunk)
        with index_stage(logger, "normalize_internal"):
            pipeline.concept_clustering_in_document(chunks)
        stats.update(
            chunks=len(chunks),
            knowledge_skipped=len(skips),
            knowledge_count=sum(c.know_unit is not None for c in chunks),
            normalized_concept_count=len(
                {c.normalized_concept for chunk in chunks for c in chunk.concepts}
            ),
        )
    return {
        "schema_version": 1,
        "source_sha256": digest(path),
        "document": asdict(InsertDocumentDomain(document, chunks)),
        "skips": skips,
        "seconds": round(perf_counter() - started, 3),
    }


def validate_prepared(path: Path, source: Path) -> tuple[InsertDocumentDomain, list]:
    value = json.loads(path.read_text())
    if value["schema_version"] != 1 or value["source_sha256"] != digest(source):
        raise RuntimeError(f"Prepared source mismatch: {source.name}")
    domain = _DOMAIN.validate_python(value["document"])
    text = pipeline.load_document(str(source))
    expected_doc = InsertDoc(
        source.name, get_hash_value(source.name), get_hash_value(text)
    )
    if domain.insert_doc != expected_doc:
        raise RuntimeError(f"Prepared document identity mismatch: {source.name}")
    raw_chunks = pipeline.markdown_spilt(text)
    if len(raw_chunks) != len(domain.chunks):
        raise RuntimeError(f"Prepared chunk count mismatch: {source.name}")
    missing = set()
    for number, (chunk, raw) in enumerate(
        zip(domain.chunks, raw_chunks, strict=True), 1
    ):
        if (
            chunk.split_no != number
            or chunk.content != raw
            or chunk.content_hash != get_hash_value(raw)
        ):
            raise RuntimeError(f"Prepared chunk mismatch: {source.name}/{number}")
        if chunk.know_unit is None:
            if chunk.concepts:
                raise RuntimeError("A skipped chunk must not have concepts")
            missing.add(
                (f"filename:{source.name},splitNo:{number}", chunk.content_hash)
            )
        for concept in chunk.concepts:
            if (
                not concept.normalized_concept.strip()
                or concept.normalized_concept_hash
                != get_hash_value(concept.normalized_concept)
            ):
                raise RuntimeError("Prepared concept hash/name mismatch")
    skips = value["skips"]
    if missing != {(s["source"], s["content_hash"]) for s in skips} or len(
        skips
    ) != len(missing):
        raise RuntimeError("Prepared missing knowledge does not match skip records")
    return domain, skips


def phase1(project: Path, run: Path, workers: int = 2) -> None:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if not (run / "probe.json").exists():
        raise RuntimeError("Network probe must pass before phase 1")
    if (run / "data").exists() or (run / "phase1").exists():
        raise RuntimeError(
            "Phase 1 requires a fresh run; existing results are preserved"
        )
    names = inputs(project, run)
    directory = run / "phase1"
    directory.mkdir()
    completed: list[str] = []
    failed: list[dict] = []
    skips: list[dict] = []
    artifacts = {}
    summaries = []
    started = perf_counter()

    def progress():
        write_json(
            run / "status.json",
            {
                "status": "phase1_running",
                "workers": workers,
                "completed": sorted(completed),
                "failed": failed,
                "total": len(names),
                "skipped_chunks": len(skips),
            },
        )

    progress()
    # Bounded document concurrency also bounds simultaneous SDK requests. No worker
    # submits another job or writes the shared progress/skip files.
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="ng-phase1"
    ) as pool:
        jobs = {
            pool.submit(prepare_document, project / "data/pilot" / name): (i, name)
            for i, name in enumerate(names, 1)
        }
        for future in as_completed(jobs):
            number, name = jobs[future]
            try:
                result = future.result()
                path = directory / f"{number:04d}.json"
                write_json(path, result)
                domain, document_skips = validate_prepared(
                    path, project / "data/pilot" / name
                )
                artifacts[name] = digest(path)
                skips.extend(document_skips)
                completed.append(name)
                summaries.append(
                    {
                        "source": name,
                        "chunks": len(domain.chunks),
                        "knowledge": sum(
                            c.know_unit is not None for c in domain.chunks
                        ),
                        "concepts": len(
                            {
                                c.normalized_concept
                                for x in domain.chunks
                                for c in x.concepts
                            }
                        ),
                        "skipped_chunks": len(document_skips),
                        "seconds": result["seconds"],
                    }
                )
            except Exception as exc:
                failed.append({"source": name, "error_type": type(exc).__name__})
                logger.exception("Phase 1 document failed: %s", name)
            write_json(
                run / "skipped-chunks.json", sorted(skips, key=lambda s: s["source"])
            )
            progress()
    if (run / "data").exists():
        raise RuntimeError("Phase 1 unexpectedly created database data")
    report = {
        "schema_version": 1,
        "status": "failed" if failed else "ready_for_review",
        "workers": workers,
        "order": list(names),
        "artifacts": artifacts,
        "documents": sorted(summaries, key=lambda x: x["source"]),
        "failed": failed,
        "seconds": round(perf_counter() - started, 3),
        "commit": (run / "commit.txt").read_text().strip(),
        "config_sha256": digest(project / "config.yaml"),
        "lock_sha256": digest(project / "uv.lock"),
    }
    write_json(run / "phase1-report.json", report)
    if failed:
        raise RuntimeError(
            f"Phase 1 failed for {len(failed)} document(s); phase 2 is blocked"
        )
    write_json(
        run / "status.json",
        {
            "status": "phase1_ready_for_review",
            "completed": list(names),
            "total": len(names),
            "skipped_chunks": len(skips),
        },
    )


def phase1_results(project: Path, run: Path) -> list[InsertDocumentDomain]:
    """Validate all artifacts before phase 2 performs any model calls or writes."""
    names = inputs(project, run)
    report = json.loads((run / "phase1-report.json").read_text())
    if report["status"] != "ready_for_review" or report["order"] != list(names):
        raise RuntimeError("Phase 1 is incomplete or its document order changed")
    if (
        report["commit"] != (run / "commit.txt").read_text().strip()
        or report["config_sha256"] != digest(project / "config.yaml")
        or report["lock_sha256"] != digest(project / "uv.lock")
    ):
        raise RuntimeError("Run code/configuration metadata changed since phase 1")
    domains = []
    skips: list[dict] = []
    for number, name in enumerate(names, 1):
        path = run / "phase1" / f"{number:04d}.json"
        if report["artifacts"].get(name) != digest(path):
            raise RuntimeError(f"Prepared artifact modified: {name}")
        domain, rejected = validate_prepared(path, project / "data/pilot" / name)
        domains.append(domain)
        skips.extend(rejected)
    if sorted(skips, key=lambda s: s["source"]) != json.loads(
        (run / "skipped-chunks.json").read_text()
    ):
        raise RuntimeError("Skip manifest changed since phase 1")
    return domains


def publish_document(domain: InsertDocumentDomain) -> None:
    """Publish prepared chunks in the original indexer order, without re-extraction."""
    source = domain.insert_doc.file_name
    with index_stage(logger, "phase2_document", source=source):
        with index_stage(logger, "normalize_external"):
            pipeline.normalized_concept(domain.chunks)
        with index_stage(logger, "sqlite_save"):
            saved = pipeline.save_documents_domain(domain)
        with index_stage(logger, "knowledge_vectors"):
            for unit in pipeline.query_by_ids_knowunit(saved["knowledge_unit_ids"]):
                vector = pipeline.get_embedding(unit.embedding_text)
                if not vector:
                    raise RuntimeError(f"Missing knowledge embedding: {unit.id}")
                pipeline.save_knowlege(
                    pipeline.KnowledgeUnitDTO(
                        str(unit.id),
                        vector,
                        unit.embedding_text,
                        pipeline.KnowledgeUnitMetadata(
                            source,
                            unit.document_id,
                            unit.document_chunk_id,
                            unit.title,
                            unit.keywords,
                        ),
                    )
                )
        # Publish the canonical concepts BEFORE consuming the next document.
        with index_stage(logger, "concept_vectors"):
            for concept in pipeline.query_by_document_id_chunk_concetps(
                saved["document_id"]
            ):
                vector = pipeline.get_embedding(concept.normalized_concept)
                if not vector:
                    raise RuntimeError(
                        f"Missing concept embedding: {concept.normalized_concept_hash}"
                    )
                pipeline.save_concept(
                    pipeline.ConceptDTO(
                        concept.normalized_concept_hash,
                        vector,
                        concept.normalized_concept,
                    )
                )
        for chunk in pipeline.query_by_id_chunks(saved["chunk_ids"]):
            concepts = [
                c.normalized_concept for c in pipeline.query_by_chunk_id(chunk.id)
            ]
            if not concepts:
                continue
            with index_stage(logger, "relations", chunk_id=chunk.id) as relation_stats:
                relations = pipeline.extract_relations_from_text(
                    chunk.content, concepts
                )
                if relations is None:
                    raise RuntimeError(f"Invalid relation output: {chunk.id}")
                relation_stats["extracted_relation_count"] = len(relations)
                relation_stats["invalid_endpoint_count"] = sum(
                    head not in concepts or tail not in concepts
                    for head, _, tail in relations
                )
                for head, relation, tail in relations:
                    if head in concepts and tail in concepts:
                        relation_id = pipeline.insert_concept_relation(
                            head,
                            tail,
                            relation,
                            pipeline.RelationType.DOCUMENT_INTERNAL,
                        )
                        if relation_id is not None:
                            pipeline.insert_relation_evidence(
                                relation_id,
                                chunk.document_id,
                                chunk.id,
                                pipeline.EvidenceRoleEnum.BOTH,
                            )


def phase2(project: Path, run: Path) -> None:
    if (run / "data").exists():
        raise RuntimeError(
            "Phase 2 data already exists; refusing to append to an existing database"
        )
    domains = phase1_results(project, run)
    from src.config.config import (
        CHROMA_CONCEPT_TABLE_NAME,
        CHROMA_INSIGHT_TABLE_NAME,
        CHROMA_KNOWLEDGE_TABLE_NAME,
    )
    from src.get_chroma_collection import get_collection

    pipeline.sqlite_table_create_if_not()
    for name in (
        CHROMA_CONCEPT_TABLE_NAME,
        CHROMA_INSIGHT_TABLE_NAME,
        CHROMA_KNOWLEDGE_TABLE_NAME,
    ):
        get_collection(name)
    completed: list[str] = []
    for domain in domains:
        write_json(
            run / "status.json",
            {
                "status": "phase2_running",
                "current": domain.insert_doc.file_name,
                "completed": completed,
                "total": len(domains),
            },
        )
        publish_document(domain)
        completed.append(domain.insert_doc.file_name)
    write_json(
        run / "status.json",
        {"status": "indexed", "completed": completed, "total": len(domains)},
    )
