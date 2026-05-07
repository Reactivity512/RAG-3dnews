"""
Unit-тесты клиента Ollama для работы с эмбеддингами
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import httpx

from src.embeddings.ollama_client import OllamaEmbeddingClient
from src.config import settings


class TestOllamaEmbeddingClient:
    """Тесты клиента для получения эмбеддингов"""

    @pytest.fixture
    def client(self):
        """Фикстура для создания клиента"""
        return OllamaEmbeddingClient()

    @pytest.mark.asyncio
    async def test_get_dense_embeddings_returns_correct_dimensions(self, client, mock_ollama_client):
        """Метод get_dense_embeddings возвращает векторы правильной размерности (768)"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1] * settings.dense_vector_size for _ in range(2)]
        }
        
        with patch('httpx.AsyncClient.post', new=AsyncMock(return_value=mock_response)):
            texts = ["test text 1", "test text 2"]
            embeddings = await client.get_dense_embeddings(texts)
            
            assert len(embeddings) == 2
            assert len(embeddings[0]) == settings.dense_vector_size

    @pytest.mark.asyncio
    async def test_get_dense_embeddings_with_empty_text(self, client):
        """Метод работает с пустым текстом (не падает)"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": []}
        
        with patch('httpx.AsyncClient.post', new=AsyncMock(return_value=mock_response)):
            embeddings = await client.get_dense_embeddings([])
            assert embeddings == []

    @pytest.mark.asyncio
    async def test_get_dense_embeddings_handles_timeout(self, client):
        """При таймауте Ollama не падает, а возвращает пустой список или обрабатывает ошибку"""
        with patch('httpx.AsyncClient.post', new=AsyncMock(side_effect=httpx.TimeoutException("Timeout"))):
            with pytest.raises(httpx.TimeoutException):
                await client.get_dense_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_get_dense_embeddings_handles_500_error(self, client):
        """При ошибке 500 от Ollama не падает, а выбрасывает исключение"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = Exception("500 Server Error")
        
        with patch('httpx.AsyncClient.post', new=AsyncMock(return_value=mock_response)):
            with pytest.raises(Exception):
                await client.get_dense_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_get_dense_embeddings_with_very_long_text(self, client):
        """Метод работает с очень длинным текстом"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.5] * settings.dense_vector_size]
        }
        
        with patch('httpx.AsyncClient.post', new=AsyncMock(return_value=mock_response)):
            long_text = "word " * 10000  # Очень длинный текст
            embeddings = await client.get_dense_embeddings([long_text])
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == settings.dense_vector_size


class TestSparseVector:
    """Тесты для sparse векторов"""

    @pytest.fixture
    def client(self):
        return OllamaEmbeddingClient()

    def test_sparse_vector_structure(self, client):
        """Метод _text_to_sparse_tokens возвращает корректную структуру"""
        indices, values = client._text_to_sparse_tokens("test text for vector")
        
        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values)
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(v, float) for v in values)

    def test_sparse_vector_sorted_by_index(self, client):
        """Sparse векторы отсортированы по индексу"""
        text = "test word test word example example test"
        indices, values = client._text_to_sparse_tokens(text)
        
        # Индексы должны быть отсортированы по возрастанию
        assert indices == sorted(indices)

    def test_sparse_vector_empty_text(self, client):
        """Пустой текст возвращает пустые списки"""
        indices, values = client._text_to_sparse_tokens("")
        
        assert indices == []
        assert values == []

    def test_sparse_vector_short_text(self, client):
        """Короткий текст (менее 3 символов) игнорируется"""
        indices, values = client._text_to_sparse_tokens("ab cd ef")
        
        # "ab" и "cd" короче 3 символов, должны игнорироваться
        # "ef" тоже короче 3 символов
        assert len(indices) == 0

    @pytest.mark.asyncio
    async def test_get_sparse_vector_returns_dict(self, client):
        """Метод get_sparse_vector возвращает словарь с ключами indices и values"""
        result = await client.get_sparse_vector("test text")
        
        assert isinstance(result, dict)
        assert "indices" in result
        assert "values" in result


class TestClientConfiguration:
    """Тесты конфигурации клиента"""

    def test_client_uses_settings_host(self):
        """Клиент использует хост из настроек"""
        client = OllamaEmbeddingClient()
        assert client.host == settings.ollama_host

    def test_client_uses_settings_embedding_model(self):
        """Клиент использует модель эмбеддинга из настроек"""
        client = OllamaEmbeddingClient()
        assert client.embedding_model == settings.embedding_model

    def test_client_uses_settings_llm_model(self):
        """Клиент использует LLM модель из настроек"""
        client = OllamaEmbeddingClient()
        assert client.llm_model == settings.llm_model

    def test_client_custom_host(self):
        """Клиент может использовать кастомный хост"""
        custom_host = "http://custom:8080"
        client = OllamaEmbeddingClient(host=custom_host)
        assert client.host == custom_host

    def test_client_default_timeout(self):
        """Клиент имеет дефолтный таймаут"""
        client = OllamaEmbeddingClient()
        assert client.timeout == 300


class TestHealthCheck:
    """Тесты проверки здоровья клиента"""

    @pytest.fixture
    def client(self):
        return OllamaEmbeddingClient()

    @pytest.mark.asyncio
    async def test_check_health_success(self, client):
        """Успешная проверка здоровья"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": settings.embedding_model},
                {"name": settings.llm_model}
            ]
        }
        
        with patch('httpx.AsyncClient.get', new=AsyncMock(return_value=mock_response)):
            result = await client.check_health()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_health_models_missing(self, client):
        """Проверка падает когда нужные модели отсутствуют"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        
        with patch('httpx.AsyncClient.get', new=AsyncMock(return_value=mock_response)):
            result = await client.check_health()
            assert result is False

    @pytest.mark.asyncio
    async def test_check_health_timeout(self, client):
        """Таймаут при проверке здоровья"""
        with patch('httpx.AsyncClient.get', new=AsyncMock(side_effect=httpx.TimeoutException)):
            result = await client.check_health()
            assert result is False
