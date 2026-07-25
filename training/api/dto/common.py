from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Pagination:
    page: int = 1
    limit: int = 50
    total_items: Optional[int] = None
    total_pages: Optional[int] = None

@dataclass(frozen=True)
class Filter:
    field: str
    operator: str
    value: str
