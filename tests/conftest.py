import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_ollama_client():
    """Mock для Ollama клиента"""
    mock = MagicMock()
    mock.get_dense_embeddings = AsyncMock(return_value=[
        [0.1] * 768 for _ in range(2)
    ])
    mock.get_sparse_vector = AsyncMock(return_value={
        "indices": [1, 2, 3],
        "values": [0.5, 0.3, 0.2]
    })
    mock.check_health = AsyncMock(return_value=True)
    mock.client = MagicMock()
    mock.client.get = MagicMock()
    return mock


@pytest.fixture
def mock_qdrant_manager():
    """Mock для Qdrant менеджера"""
    mock = MagicMock()
    mock.collection_exists = MagicMock(return_value=True)
    mock.get_stats = MagicMock(return_value={
        "points_count": 100,
        "segments_count": 5
    })
    mock.get_existing_urls = MagicMock(return_value=[])
    mock.create_collection = MagicMock()
    mock.upsert_news = MagicMock()
    mock.get_by_ids = MagicMock(return_value=[])
    mock.client = MagicMock()
    mock.client.collection_exists = MagicMock(return_value=True)
    mock.collection_name = "test_collection"
    return mock


@pytest.fixture
def mock_rag_service():
    """Mock для RAG сервиса"""
    mock = MagicMock()
    mock.health_check = AsyncMock(return_value={
        "qdrant": True,
        "ollama": True,
        "collection_stats": {"points_count": 100}
    })
    mock.search_and_answer = AsyncMock(return_value={
        "answer": "Test answer",
        "results": []
    })
    return mock


@pytest.fixture
def test_app(mock_ollama_client, mock_qdrant_manager, mock_rag_service):
    """Создание тестового FastAPI приложения с моками"""
    # Применяем моки ДО импорта приложения
    with patch('src.main.ollama_client', mock_ollama_client), \
         patch('src.main.qdrant_manager', mock_qdrant_manager), \
         patch('src.main.rag_service', mock_rag_service):
        
        from src.main import app
        yield app


@pytest.fixture
def api_client(test_app):
    """Фикстура для тестирования API эндпоинтов"""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def sample_rss_items():
    """Пример RSS данных для тестов"""
    return [
        {
            "title": "Test News 1",
            "content": "Content of test news 1",
            "url": "http://example.com/1",
            "published": "2024-01-01"
        },
        {
            "title": "Test News 2",
            "content": "Content of test news 2",
            "url": "http://example.com/2",
            "published": "2024-01-02"
        }
    ]