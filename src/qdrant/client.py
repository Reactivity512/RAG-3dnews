from qdrant_client import QdrantClient, models
from typing import List, Optional, Tuple, Set
import logging

from src.config import settings
from src.models.schemas import NewsItem, SparseVector

logger = logging.getLogger(__name__)

class QdrantManager:
    """Управление коллекцией в Qdrant с поддержкой hybrid search"""
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            prefer_grpc=True
        )
        self.collection_name = settings.qdrant_collection
        self.dense_size = settings.dense_vector_size
        self.dense_distance = settings.dense_distance
        
    def create_collection(self):
        """Создаёт коллекцию с dense + sparse векторами"""
        if self.client.collection_exists(self.collection_name):
            logger.info(f"Collection {self.collection_name} already exists")
            return
        
        # Создаем коллекцию с именованными векторами
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.dense_size,
                    distance=models.Distance.COSINE,
                    on_disk=True
                )
            },
            sparse_vectors_config={
                "text": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,
                        full_scan_threshold=1000
                    )
                )
            },
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=2,
                memmap_threshold=10000
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            )
        )
        logger.info(f"Created collection {self.collection_name}")
        
        # Создаём индексы для фильтрации
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="published_at",
            field_schema=models.PayloadSchemaType.DATETIME
        )
    
    def upsert_news(self, items: List[NewsItem], embeddings: List[List[float]], 
                   sparse_vectors: List[dict]):
        """Добавляет новости в коллекцию"""
        points = []
        for item, dense, sparse in zip(items, embeddings, sparse_vectors):
            # Создаем sparse вектор в правильном формате
            sparse_vector = models.SparseVector(
                indices=sparse["indices"], 
                values=sparse["values"]
            )
            
            point = models.PointStruct(
                id=item.id,
                vector={
                    "dense": dense,
                    "text": sparse_vector  # Имя должно совпадать с sparse_vectors_config
                },
                payload={
                    "title": item.title,
                    "content": item.content,
                    "summary": item.summary,
                    "url": item.url,
                    "published_at": item.published_at.isoformat(),
                    "source": item.source
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        logger.info(f"Upserted {len(points)} points")
    
    def hybrid_search(
        self, 
        query_dense: List[float], 
        query_sparse: dict,
        top_k: int = 10,
        fusion_k: int = 60
    ) -> List[Tuple[models.ScoredPoint, float]]:
        """
        Выполняет гибридный поиск с Reciprocal Rank Fusion (RRF).
        Возвращает скорированные результаты.
        """
        # 1. Dense search - используем именованный вектор
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedVector(
                name="dense",
                vector=query_dense
            ),
            limit=top_k,
            with_payload=True
        )
        
        # 2. Sparse search - используем NamedSparseVector
        sparse_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedSparseVector(
                name="text", 
                vector=models.SparseVector(
                    indices=query_sparse["indices"], 
                    values=query_sparse["values"]
                )
            ),
            limit=top_k,
            with_payload=True
        )
        
        # 3. Reciprocal Rank Fusion
        rrf_scores = {}
        for rank, result in enumerate(dense_results, 1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + 1 / (rank + fusion_k)
        for rank, result in enumerate(sparse_results, 1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + 1 / (rank + fusion_k)
        
        # Сортируем по итоговому RRF-скор
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Возвращаем с объединёнными данными
        all_results = {r.id: r for r in dense_results + sparse_results}
        final = []
        for pid in sorted_ids[:top_k]:
            if pid in all_results:
                final.append((all_results[pid], rrf_scores[pid]))
        
        return final
    
    def get_by_ids(self, ids: List[str]) -> List[NewsItem]:
        """Получает новости по списку ID"""
        from src.models.schemas import NewsItem
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False
        )
        items = []
        for p in points:
            payload = p.payload
            items.append(NewsItem(
                id=p.id,
                title=payload["title"],
                content=payload["content"],
                summary=payload.get("summary"),
                url=payload["url"],
                published_at=payload["published_at"],
                source=payload["source"]
            ))
        return items
    
    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)
    
    def get_stats(self) -> dict:
        """Возвращает статистику коллекции"""
        info = self.client.get_collection(self.collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count
        }

    def get_existing_urls(self) -> Set[str]:
        """Получает все URL из коллекции через постраничный scroll"""
        existing = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=["url"],
                offset=offset
            )
            existing.update(r.payload.get("url") for r in records if r.payload.get("url"))
            if offset is None:
                break
        return existing