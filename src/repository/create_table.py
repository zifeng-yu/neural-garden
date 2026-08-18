from src.get_sqlite_connection import get_sqlite_connection


def create_table_init():
    with get_sqlite_connection() as conn:

        # 文档表
        sql_documents = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_name_hash TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT (datetime('now', '+8 hours')),
            updated_at DATETIME DEFAULT (datetime('now', '+8 hours')),
            UNIQUE(file_name_hash, content_hash)
        );
        """
        conn.execute(sql_documents)

        # 文档分块表
        sql_document_chunks = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            split_no INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT (datetime('now', '+8 hours')),
            updated_at DATETIME DEFAULT (datetime('now', '+8 hours')),

            UNIQUE(document_id, split_no),

            FOREIGN KEY (document_id)
                REFERENCES documents(id)
        );
        """
        conn.execute(sql_document_chunks)

        # 文档分块知识单元
        sql_document_chunk_knowledge_units = """
        CREATE TABLE IF NOT EXISTS document_chunk_knowledge_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            document_chunk_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            keywords TEXT NOT NULL,
            embedding_text TEXT NOT NULL,
            created_at DATETIME DEFAULT (datetime('now', '+8 hours')),
            updated_at DATETIME DEFAULT (datetime('now', '+8 hours')),

            FOREIGN KEY (document_id)
                REFERENCES documents(id),

            FOREIGN KEY (document_chunk_id)
                REFERENCES document_chunks(id),

            UNIQUE(document_chunk_id)
        );
        """
        conn.execute(sql_document_chunk_knowledge_units)

        # 文档分块概念表
        sql_document_chunk_concepts = """
        CREATE TABLE IF NOT EXISTS document_chunk_concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            document_chunk_id INTEGER NOT NULL,
            concept TEXT NOT NULL,
            normalized_concept TEXT NOT NULL,
            normalized_concept_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT (datetime('now', '+8 hours')),
            updated_at DATETIME DEFAULT (datetime('now', '+8 hours')),

            FOREIGN KEY (document_id)
                REFERENCES documents(id),

            FOREIGN KEY (document_chunk_id)
                REFERENCES document_chunks(id),

            UNIQUE(document_chunk_id, concept)
        );
        """
        conn.execute(sql_document_chunk_concepts)

        # 索引
        index_sql_idx_chunk_concepts_normalized = """
        CREATE INDEX IF NOT EXISTS idx_chunk_concepts_normalized
            ON document_chunk_concepts(normalized_concept);
        """
        conn.execute(index_sql_idx_chunk_concepts_normalized)

        sql_concept_relations = """
        CREATE TABLE IF NOT EXISTS concept_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_concept TEXT NOT NULL,
            target_concept TEXT NOT NULL,
            relation TEXT NOT NULL,
            relation_source TEXT NOT NULL,
            confidence REAL,
            created_at DATETIME DEFAULT (datetime('now', '+8 hours')),
            updated_at DATETIME DEFAULT (datetime('now', '+8 hours')),
            UNIQUE(
                source_concept,
                target_concept,
                relation,
                relation_source
            )
        );
        """
        conn.execute(sql_concept_relations)

        conn.commit()


def drop_table():
    with get_sqlite_connection() as conn:

        sql = """
            DROP TABLE IF EXISTS document_chunk_concepts;
            DROP TABLE IF EXISTS document_chunk_knowledge_units;
            DROP TABLE IF EXISTS document_chunks;
            DROP TABLE IF EXISTS documents;
            """
        conn.executescript(sql)
        conn.commit()
