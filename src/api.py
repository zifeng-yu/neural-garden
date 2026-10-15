import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    logger.info("Neural Garden API 启动")
    try:
        yield
    finally:
        logger.info("Neural Garden API 停止")


app = FastAPI(title="Neural Garden", lifespan=lifespan)


def run_answer(question: str, top_k: int):
    from src.rag_answer import answer_question

    return answer_question(question, top_k=top_k)


@app.get("/health")
def health():
    return {"status": "ok"}


class AnswersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=200)
    top_k: int = Field(strict=True, ge=1, le=5, default=3)


@app.post("/answers")
def answers(requestBody: AnswersRequest):
    try:
        return run_answer(requestBody.question, requestBody.top_k)
    except Exception as e:
        logger.exception("RAG 回答请求失败，top_k=%s", requestBody.top_k)
        raise HTTPException(500, "服务错误") from e
