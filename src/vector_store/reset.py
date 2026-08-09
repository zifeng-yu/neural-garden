import logging

import chromadb

import src.config.logging_config
from src.config.config import (
    CHROMA_CONCEPT_TABLE_NAME,
    CHROMA_KNOWLEDGE_TABLE_NAME,
    PERSIST_DIRECTORY,
)

logger = logging.getLogger(__name__)


def resetDB_KNOWLEDGE():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    client.delete_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    logger.info("reset chromaDB KNOWLEDGE finish")


def resetDB_CONCEPT():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    client.delete_collection(CHROMA_CONCEPT_TABLE_NAME)
    logger.info("reset chromaDB CONCEPT finish")
