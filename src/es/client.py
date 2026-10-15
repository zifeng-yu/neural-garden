"""ES客户端配置；创建客户端不执行索引或同步。"""

from elasticsearch import Elasticsearch

from src.config.config import ES_REQUEST_TIMEOUT, ES_URL


def get_es_client() -> Elasticsearch:
    return Elasticsearch(ES_URL, request_timeout=ES_REQUEST_TIMEOUT)
