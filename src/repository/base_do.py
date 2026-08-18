from dataclasses import dataclass
from datetime import datetime


@dataclass
class BaseDO:
    id: int
    created_at: datetime
    updated_at: datetime
