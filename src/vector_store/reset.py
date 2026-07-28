import logging

import chromadb

import src.config.logging_config
from src.config.config import CHROMA_TABLE_NAME, PERSIST_DIRECTORY


def reset():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    client.delete_collection(CHROMA_TABLE_NAME)
    logging.info("reset chromaDB finish")
