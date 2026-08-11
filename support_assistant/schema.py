from typing import List

from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
