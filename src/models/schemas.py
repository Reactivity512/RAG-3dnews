from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

class NewsItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    summary: Optional[str] = None
    url: str
    published_at: datetime
    source: str  # "hardware" или "software"
    
    class Config:
        from_attributes = True

class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]

class ArticlePoint(BaseModel):
    id: str
    payload: dict
    vector: List[float]  # dense
    sparse_vector: SparseVector  # sparse (BM25)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=50)
    use_hybrid: bool = True  # Dense + Sparse fusion

class QueryResponse(BaseModel):
    answer: str
    sources: List[NewsItem]
    query_time_ms: float

class LoadNewsRequest(BaseModel):
    urls: Optional[List[str]] = None  # если не указано - все из config
    limit: Optional[int] = None  # лимит новостей на фид

class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    ollama: bool
    collection_exists: bool