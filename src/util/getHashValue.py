import hashlib


def get_hash_value(content: str) -> str:
    return hashlib.md5(f"{content}".encode()).hexdigest()
