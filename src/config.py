from pydantic_settings import BaseSettings
from typing import List, Union
from pydantic import field_validator, model_validator
import json


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
    
    # RSS - используем str чтобы избежать автоматического парсинга как JSON
    rss_feeds: str = "https://3dnews.ru/hardware-news/rss,https://3dnews.ru/software-news/rss"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # RAG
    top_k_dense: int = 10
    top_k_sparse: int = 10
    fusion_k: int = 60
    max_context_tokens: int = 2000
    
    @field_validator('rss_feeds', mode='before')
    @classmethod
    def parse_rss_feeds(cls, v):
        if isinstance(v, str):
            # Пробуем сначала распарсить как JSON
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                # Если не JSON, значит это comma-separated строка
                return v
        return v
    
    @model_validator(mode='after')
    def convert_rss_feeds_to_list(self):
        # Конвертируем rss_feeds в список если это строка
        if isinstance(self.rss_feeds, str):
            self.rss_feeds = [url.strip() for url in self.rss_feeds.split(',') if url.strip()]
        return self
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Дополнительные атрибуты для обратной совместимости
OLLAMA_BASE_URL = settings.ollama_host
QDRANT_URL = f"{settings.qdrant_host}:{settings.qdrant_port}"
QDRANT_COLLECTION_NAME = settings.qdrant_collection
DENSE_VECTOR_SIZE = settings.dense_vector_size
TOP_K_DENSE = settings.top_k_dense
TOP_K_SPARSE = settings.top_k_sparse
FUSION_K = settings.fusion_k