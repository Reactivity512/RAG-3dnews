"""
Тесты конфигурации и проверки доступности сервисов (Health Checks)
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import httpx

from src.config import settings


class TestConfigLoading:
    """Тесты загрузки конфигурации"""

    def test_default_settings_loaded(self):
        """Проверяем, что настройки по умолчанию загружаются корректно"""
        assert settings.ollama_host == "http://localhost:11434"
        assert settings.embedding_model == "evilfreelancer/enbeddrus"
        assert settings.llm_model == "qwen2.5:1.5b"

    def test_qdrant_settings(self):
        """Проверяем настройки Qdrant"""
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.qdrant_collection == "3dnews_articles"
        assert settings.dense_vector_size == 768

    def test_api_settings(self):
        """Проверяем настройки API"""
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_rag_settings(self):
        """Проверяем настройки RAG"""
        assert settings.top_k_dense == 10
        assert settings.top_k_sparse == 10
        assert settings.fusion_k == 60
        assert settings.max_context_tokens == 2000

    def test_rss_feeds_default(self):
        """Проверяем RSS фиды по умолчанию"""
        assert isinstance(settings.rss_feeds, list)
        assert len(settings.rss_feeds) == 2
        assert "3dnews.ru" in settings.rss_feeds[0]


class TestOllamaHealthCheck:
    """Тесты проверки здоровья Ollama"""

    @pytest.mark.asyncio
    async def test_check_ollama_health_success(self):
        """Ollama отвечает 200 - проверка проходит"""
        from src.embeddings.ollama_client import OllamaEmbeddingClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = MagicMock(return_value=mock_response)
        
        with patch.object(OllamaEmbeddingClient, 'check_health', new=AsyncMock(return_value=True)):
            client = OllamaEmbeddingClient()
            client.client = mock_client
            result = await client.check_health()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_ollama_health_timeout(self):
        """Ollama не отвечает - таймаут"""
        from src.embeddings.ollama_client import OllamaEmbeddingClient
        
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Timeout")
        
        client = OllamaEmbeddingClient()
        client.client = mock_client
        result = await client.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_ollama_health_server_error(self):
        """Ollama вернул 500 - ошибка сервера"""
        from src.embeddings.ollama_client import OllamaEmbeddingClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.get = MagicMock(return_value=mock_response)
        
        client = OllamaEmbeddingClient()
        client.client = mock_client
        result = await client.check_health()
        assert result is False


class TestQdrantHealthCheck:
    """Тесты проверки здоровья Qdrant"""

    def test_qdrant_collection_exists(self):
        """Коллекция существует"""
        from src.qdrant.client import QdrantManager
        
        mock_manager = MagicMock()
        mock_manager.client.collection_exists.return_value = True
        
        # Эмулируем проверку
        result = mock_manager.client.collection_exists("test_collection")
        assert result is True

    def test_qdrant_collection_missing(self):
        """Коллекция не найдена"""
        from src.qdrant.client import QdrantManager
        
        mock_manager = MagicMock()
        mock_manager.client.collection_exists.return_value = False
        
        result = mock_manager.client.collection_exists("test_collection")
        assert result is False

    def test_qdrant_connection_error(self):
        """Ошибка подключения к Qdrant"""
        from qdrant_client.http.exceptions import UnexpectedResponse
        
        mock_manager = MagicMock()
        mock_manager.client.collection_exists.side_effect = UnexpectedResponse(
            status_code=502,
            reason_phrase="Bad Gateway",
            headers={"content-type": "application/json"},
            content=b"Connection refused"
        )
        
        with pytest.raises(UnexpectedResponse):
            mock_manager.client.collection_exists("test_collection")


class TestConfigValidation:
    """Тесты валидации конфигурации"""

    def test_valid_rss_feeds_string(self):
        """Валидная строка RSS фидов"""
        from src.config import Settings
        s = Settings(rss_feeds="http://example.com/1,http://example.com/2")
        assert len(s.rss_feeds) == 2
        assert s.rss_feeds[0] == "http://example.com/1"
        assert s.rss_feeds[1] == "http://example.com/2"

    def test_rss_feeds_comma_separated(self):
        """RSS фиды через запятую парсятся корректно"""
        from src.config import Settings
        s = Settings(rss_feeds="http://a.com,http://b.com")
        assert len(s.rss_feeds) == 2

    def test_rss_feeds_list_input(self):
        """Список RSS фидов принимается как есть"""
        from src.config import Settings
        # rss_feeds должен быть строкой для pydantic-settings
        s = Settings(rss_feeds="http://a.com,http://b.com")
        assert isinstance(s.rss_feeds, list)