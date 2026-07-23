import chromadb

from config import CHROMA_TABLE_NAME, PERSIST_DIRECTORY


def reset():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    client.delete_collection(CHROMA_TABLE_NAME)
    print("reset chromaDB finish")
