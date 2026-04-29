# 🗞️ 3DNews RAG System

RAG-система для поиска и ответов по новостям 3DNews с использованием:
- **LLM**: `qwen3.5:4b` через Ollama
- **Embeddings**: `nomic-embed-text` через Ollama  
- **Vector DB**: Qdrant с гибридным поиском (Dense + Sparse)
- **API**: FastAPI

## 🚀 Быстрый старт

### 1. Подготовка
```bash
# Клонируйте репозиторий
git clone <repo>
cd rag-3dnews

# Создайте .env
cp .env.example .env
# Отредактируйте под ваши нужды

# Запустите сервисы
docker-compose up -d

# Дождитесь загрузки моделей в Ollama:
docker-compose exec ollama ollama pull evilfreelancer/enbeddrus
docker-compose exec ollama ollama pull qwen2.5:1.5b
```

### 2. Загрузка новостей
```bash
# Через API
curl -X POST http://localhost:8000/load-news

# Или напрямую скриптом
python scripts/load_news.py
```

### 3. Использование API
```bash
# Поиск с ответом
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Какие новые видеокарты вышли в 2026?",
    "top_k": 5,
    "use_hybrid": true
  }'

# Проверка здоровья
curl http://localhost:8000/health
```

## 🔧 Конфигурация гибридного поиска

В `src/config.py` можно настроить:
* `top_k_dense` / `top_k_sparse`: сколько результатов брать от каждого поиска
* `fusion_k`: параметр RRF (меньше = больший вес топ-результатам)
* `dense_vector_size`: 768 для nomic-embed-text

## 🧪 Тестирование

```bash
# Запуск тестов
pytest tests/

# Локальный запуск API
uvicorn src.main:app --reload
```

## 📊 Мониторинг

* Swagger UI: http://localhost:8000/docs
* Stats endpoint: GET /stats
* Health: GET /health

## ⚠️ Примечания

1. Sparse векторы: В данном проекте используется упрощённый подход на основе частоты токенов.
Для продакшена рекомендуется использовать SPLADE или fine-tuned sparse encoder.
2. Производительность:
	- Эмбеддинги генерируются последовательно — для ускорения добавить батчинг
	- При >10к документов настроить hnsw_config в Qdrant
3. Безопасность: Добавьте аутентификацию и rate limiting перед деплоем.

📄 Лицензия

MIT

