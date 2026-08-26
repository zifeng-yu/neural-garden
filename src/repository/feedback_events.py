from dataclasses import dataclass
from enum import Enum

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO


class EventTypeEnum(str, Enum):
    CLICK = "click"
    DWELL = "dwell"
    COPY = "copy"
    LIKE = "like"
    DISLIKE = "dislike"


@dataclass
class FeedbackEventsDO(BaseDO):
    session_id: str
    event_type: EventTypeEnum
    document_id: int
    rank: int
    dwell_time: int


def save_feedback_events(
    session_id: str,
    event_type: EventTypeEnum,
    document_id: int | None,
    rank: int | None,
    dwell_time: int | None,
) -> int:
    with get_sqlite_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback_events
            (
                session_id,
                event_type,
                document_id,
                rank,
                dwell_time
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event_type.value,
                document_id,
                rank,
                dwell_time,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("insert feedback_events failed")
        return cursor.lastrowid
