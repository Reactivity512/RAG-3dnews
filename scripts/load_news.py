#!/usr/bin/env python3
"""CLI-скрипт для загрузки новостей в Qdrant"""

import asyncio
import sys
import logging
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.rss.parser import load_all_feeds
from src.embeddings.ollama_client import OllamaEmbeddingClient
from src.qdrant.client import QdrantManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting news loader...")
    
    # Инициализация
    ollama = OllamaEmbeddingClient()
    qdrant = QdrantManager()
    qdrant.create_collection()
    
    # Загрузка новостей
    logger.info("Loading RSS feeds...")
    items = load_all_feeds()
    logger.info(f"Loaded {len(items)} news items")
    
    if not items:
        logger.error("No items to load!")
        return
    
    # Генерация эмбеддингов
    logger.info("Generating embeddings...")
    dense_embeddings = []
    sparse_vectors = []
    
    for i, item in enumerate(items, 1):
        # 1. Подготовьте тексты
        texts = [f"{item.title}. {item.content}" for item in items]

        # 2. Получите dense-эмбеддинги одним запросом (батч)
        dense_embeddings = await ollama.get_dense_embeddings(texts)

        # 3. Сгенерируйте sparse-векторы (локально, быстро)
        sparse_vectors = [await ollama.get_sparse_vector(t) for t in texts]

        # 4. Загрузите в Qdrant
        logger.info(f"Upserting {len(items)} items to Qdrant...")
        qdrant_manager.upsert_news(items, dense_embeddings, sparse_vectors)
    
    # Загрузка в Qdrant
    logger.info("Upserting to Qdrant...")
    qdrant.upsert_news(items, dense_embeddings, sparse_vectors)
    
    # Статистика
    stats = qdrant.get_stats()
    logger.info(f"Done! Collection stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())