import time
import asyncio
from typing import List
from src.config import settings
from src.models.schemas import NewsItem, QueryRequest, QueryResponse
from src.embeddings.ollama_client import OllamaEmbeddingClient
from src.qdrant.client import QdrantManager

class RAGService:
    """Основной сервис RAG"""
    
    def __init__(self):
        self.ollama = OllamaEmbeddingClient()
        self.qdrant = QdrantManager()
        self.qdrant.create_collection()
    
    async def search_and_answer(self, request: QueryRequest) -> QueryResponse:
        """Полный пайплайн: запрос → поиск → генерация ответа"""
        start_time = time.time()
        
        # 1. Получаем эмбеддинги запроса
        # ВАЖНО: передаем список с одним элементом!
        query_dense_list = await self.ollama.get_dense_embeddings([request.query])
        query_dense = query_dense_list[0]  # Берем первый (единственный) вектор
        
        query_sparse = await self.ollama.get_sparse_vector(request.query)
        
        # 2. Гибридный поиск
        if request.use_hybrid:
            results = self.qdrant.hybrid_search(
                query_dense=query_dense,  # Теперь это плоский список
                query_sparse=query_sparse,
                top_k=request.top_k,
                fusion_k=settings.fusion_k
            )
            scored_points = [rp for rp, _ in results]
        else:
            # Только dense для сравнения
            scored_points = self.qdrant.client.search(
                collection_name=settings.qdrant_collection,
                query_vector=models.NamedVector(  # Для консистентности тоже используем named vector
                    name="dense",
                    vector=query_dense
                ),
                limit=request.top_k,
                with_payload=True
            )
        
        # 3. Извлекаем новости
        sources: List[NewsItem] = []
        contexts = []
        for sp in scored_points:
            payload = sp.payload
            news = NewsItem(
                id=sp.id,
                title=payload["title"],
                content=payload["content"],
                url=payload["url"],
                published_at=payload["published_at"],
                source=payload["source"]
            )
            sources.append(news)
            # Формируем контекст: заголовок + первые 500 символов
            ctx = f"[{news.title}] {news.content[:500]}..."
            contexts.append(ctx)
        
        # 4. Генерируем ответ
        context_text = "\n\n".join(contexts[:5])  # лимит контекста
        answer = await self.ollama.generate_response(
            query=request.query,
            context=context_text,
            max_tokens=300
        )
        
        query_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query_time_ms=round(query_time, 2)
        )
    
    async def health_check(self) -> dict:
        """Проверка здоровья всех компонентов"""
        return {
            "qdrant": self.qdrant.collection_exists(),
            "ollama": await self.ollama.check_health(),
            "collection_stats": self.qdrant.get_stats() if self.qdrant.collection_exists() else None
        }