import httpx
import logging
from typing import List, Tuple
from src.config import settings

logger = logging.getLogger(__name__)

class OllamaEmbeddingClient:
    """Клиент для получения эмбеддингов через Ollama API"""
    
    def __init__(self, host: str = None):
        self.host = host or settings.ollama_host
        self.embedding_model = settings.embedding_model
        self.llm_model = settings.llm_model
        self.timeout = 300
        
    async def get_dense_embeddings(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Batch embedding generation с чанками и увеличенным таймаутом"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.host}/api/embed",
                    json={"model": self.embedding_model, "input": batch, "truncate": True}
                )
                response.raise_for_status()
                all_embeddings.extend(response.json()["embeddings"])
        return all_embeddings
    
    def _text_to_sparse_tokens(self, text: str, max_tokens: int = 256) -> Tuple[List[int], List[float]]:
        """Создаёт sparse-вектор с уникальными отсортированными индексами"""
        import re
        from collections import Counter, defaultdict
        
        # Токенизация
        tokens = re.findall(r'[a-zA-Zа-яА-ЯёЁ]{3,}', text.lower())
        if not tokens:
            return [], []
        
        # TF через Counter (имеет most_common)
        tf = Counter(tokens)
        
        # Агрегация весов по индексам (решает коллизии хешей)
        index_weights = defaultdict(float)
        for token, count in tf.most_common(max_tokens):  # ← Counter.most_common()
            idx = abs(hash(token)) % 100000
            weight = 1 + (count ** 0.5)
            index_weights[idx] += weight  # суммируем при коллизиях
        
        # Сортировка по индексам (требование Qdrant)
        sorted_items = sorted(index_weights.items(), key=lambda x: x[0])
        indices = [idx for idx, _ in sorted_items]
        values = [w for _, w in sorted_items]
        
        return indices, values
    
    async def get_sparse_vector(self, text: str) -> dict:
        """Возвращает sparse-вектор в формате Qdrant"""
        indices, values = self._text_to_sparse_tokens(text)
        return {"indices": indices, "values": values}
    
    async def generate_response(
        self, 
        query: str, 
        context: str, 
        max_tokens: int = 512
    ) -> str:
        """Генерирует ответ"""
        prompt = f"""Ты — помощник по новостям технологий. Ответь на вопрос, используя только предоставленный контекст.
Если в контексте нет ответа — скажи об этом.

Контекст:
{context}

Вопрос: {query}

Ответ:"""
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{settings.ollama_host}/api/generate",
                json={
                    "model": settings.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Нет ответа").strip()
    
    async def check_health(self) -> bool:
        """Проверяет доступность Ollama и моделей"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Проверка сервера
                r = await client.get(f"{self.host}/api/tags")
                if r.status_code != 200:
                    return False
                models = r.json().get("models", [])
                # Проверка наличия нужных моделей
                model_names = [m["name"] for m in models]
                return (self.embedding_model in model_names and 
                        self.llm_model in model_names)
        except:
            return False