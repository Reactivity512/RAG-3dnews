from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.config import settings
from src.models.schemas import (
    QueryRequest, QueryResponse, LoadNewsRequest, 
    NewsItem, HealthResponse
)
from src.rss.parser import load_all_feeds
from src.embeddings.ollama_client import OllamaEmbeddingClient
from src.qdrant.client import QdrantManager
from src.rag.service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные инстансы
rag_service: RAGService = None
qdrant_manager: QdrantManager = None
ollama_client: OllamaEmbeddingClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте"""
    global rag_service, qdrant_manager, ollama_client
    
    logger.info("Starting RAG service...")
    
    # Инициализируем компоненты
    ollama_client = OllamaEmbeddingClient()
    qdrant_manager = QdrantManager()
    qdrant_manager.create_collection()
    rag_service = RAGService()
    
    logger.info("Services initialized")
    yield

    logger.info("Shutting down...")

app = FastAPI(
    title="3DNews RAG API",
    description="RAG-система для поиска по новостям 3DNews",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    """Проверка здоровья сервиса"""
    stats = await rag_service.health_check()
    return HealthResponse(
        status="ok" if all([stats["qdrant"], stats["ollama"]]) else "degraded",
        qdrant=stats["qdrant"],
        ollama=stats["ollama"],
        collection_exists=stats["collection_stats"] is not None
    )

@app.post("/query", response_model=QueryResponse)
async def query_news(request: QueryRequest):
    """Поиск новости и генерация ответа"""
    try:
        return await rag_service.search_and_answer(request)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-news")
async def load_news(request: LoadNewsRequest = None, background_tasks: BackgroundTasks = None):
    """
    Загрузка новостей из RSS в Qdrant.
    Выполняется асинхронно в фоне.
    """
    if request and request.urls:
        feeds = request.urls
    else:
        feeds = settings.rss_feeds
    
    limit = request.limit if request else None
    
    async def _load():
        # Гарантируем создание коллекции (метод идемпотентный)
        qdrant_manager.create_collection()

        logger.info("Starting news load...")

        existing_urls = qdrant_manager.get_existing_urls()
        
        # 1. Загружаем новости
        items = load_all_feeds()

        # 2. Фильтруем уже обработанные
        new_items = [i for i in items if i.url not in existing_urls]

        if not new_items:
            logger.info("No new items")
            return

        # 3. Генерируем векторы ТОЛЬКО для новых
        logger.info(f"Generating embeddings for {len(new_items)} items...")
        texts = [f"{i.title}. {i.content}" for i in new_items]
        dense_embeddings = await ollama_client.get_dense_embeddings(texts)
        sparse_vectors = [await ollama_client.get_sparse_vector(t) for t in texts]


        # 4. Загружаем в Qdrant
        logger.info("Upserting to Qdrant...")
        qdrant_manager.upsert_news(new_items, dense_embeddings, sparse_vectors)
        logger.info("Load to Qdrant complete")

        logger.info(f"Loaded {len(new_items)} new items")
    
    if background_tasks:
        background_tasks.add_task(_load)
        return {"status": "started", "message": "Loading in background"}
    else:
        await _load()
        return {"status": "completed", "loaded": len(items)}

@app.get("/news/{news_id}")
async def get_news(news_id: str):
    """Получение новости по ID"""
    items = qdrant_manager.get_by_ids([news_id])
    if not items:
        raise HTTPException(status_code=404, detail="News not found")
    return items[0]

@app.get("/stats")
async def get_stats():
    """Статистика коллекции"""
    if not qdrant_manager.collection_exists():
        return {"error": "Collection not found"}
    return qdrant_manager.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )