from pydantic_settings import BaseSettings
from typing import List, Optional
from pydantic import field_validator

class Settings(BaseSettings):
    # Ollama
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "evilfreelancer/enbeddrus"
    llm_model: str = "qwen2.5:1.5b"
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "3dnews_articles"
    dense_vector_size: int = 768
    dense_distance: str = "Cosine"
    
    # RSS
    rss_feeds: List[str] = [
        "https://3dnews.ru/hardware-news/rss",
        "https://3dnews.ru/software-news/rss"
    ]

    # валидатор для парсинга строки в список
    @field_validator('rss_feeds', mode='before')
    @classmethod
    def parse_rss_feeds(cls, v):
        if isinstance(v, str):
            return [url.strip() for url in v.split(',') if url.strip()]
        return v
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # RAG
    top_k_dense: int = 10
    top_k_sparse: int = 10
    fusion_k: int = 60  # для RRF (Reciprocal Rank Fusion)
    max_context_tokens: int = 2000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()