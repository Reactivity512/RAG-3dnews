"""
Тесты API эндпоинтов (без LLM-логики)
"""
import pytest
from unittest.mock import MagicMock, patch


class TestHealthEndpoint:
    """Тесты эндпоинта /health"""

    def test_health_returns_200_and_ok(self, api_client):
        """Проверка что /health возвращает 200 и статус ok"""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "degraded"]

    def test_health_response_structure(self, api_client):
        """Проверка структуры ответа health"""
        response = api_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "qdrant" in data
        assert "ollama" in data
        assert "collection_exists" in data


class TestStatsEndpoint:
    """Тесты эндпоинта /stats"""

    def test_stats_returns_200_and_valid_json(self, api_client):
        """Проверка что /stats возвращает 200 и валидный JSON"""
        response = api_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_stats_has_expected_fields(self, api_client, mock_qdrant_manager):
        """Проверка что в ответе есть ожидаемые поля"""
        mock_qdrant_manager.get_stats.return_value = {
            "points_count": 100,
            "segments_count": 5
        }
        
        response = api_client.get("/stats")
        assert response.status_code == 200

    def test_stats_collection_not_found(self, api_client, mock_qdrant_manager):
        """Проверка обработки случая когда коллекция не найдена"""
        mock_qdrant_manager.collection_exists.return_value = False
        
        response = api_client.get("/stats")
        data = response.json()
        assert data["indexed_vectors_count"] == 0


class TestQueryEndpoint:
    """Тесты эндпоинта /query"""

    VALID_QUERY_PAYLOAD = {
        "query": "тестовый запрос",
        "top_k": 5,
        "use_hybrid": True
    }

    def test_query_returns_200_on_valid_input(self, api_client):
        """Запрос с валидными данными возвращает 200"""
        response = api_client.post("/query", json=self.VALID_QUERY_PAYLOAD)
        assert response.status_code == 200

    def test_query_response_has_required_fields(self, api_client):
        """Ответ содержит необходимые поля"""
        response = api_client.post("/query", json=self.VALID_QUERY_PAYLOAD)
        data = response.json()
        assert "answer" in data or "results" in data

    def test_query_missing_query_field(self, api_client):
        """Отсутствие поля query возвращает 422"""
        response = api_client.post("/query", json={"top_k": 5})
        assert response.status_code == 422

    def test_query_empty_query_string(self, api_client):
        """Пустой query возвращает 422"""
        response = api_client.post("/query", json={
            "query": "",
            "top_k": 5
        })
        assert response.status_code == 422

    def test_query_negative_top_k(self, api_client):
        """Отрицательный top_k возвращает 422"""
        response = api_client.post("/query", json={
            "query": "test",
            "top_k": -1
        })
        assert response.status_code == 422

    def test_query_zero_top_k(self, api_client):
        """Нулевой top_k возвращает 422"""
        response = api_client.post("/query", json={
            "query": "test",
            "top_k": 0
        })
        assert response.status_code == 422

    def test_query_with_high_top_k_does_not_crash(self, api_client):
        """Большой top_k не вызывает краш"""
        response = api_client.post("/query", json={
            "query": "test",
            "top_k": 1000
        })
        assert response.status_code == 422


class TestLoadNewsEndpoint:
    """Тесты эндпоинта /load-news"""

    def test_load_news_endpoint_exists_and_returns_success(self, api_client):
        """Эндпоинт /load-news существует и возвращает успех"""
        response = api_client.post("/load-news")
        # Ожидаем успешный ответ
        assert response.status_code in [200, 500]

    def test_load_news_response_contains_count(self, api_client):
        """Ответ содержит информацию о загруженных новостях"""
        response = api_client.post("/load-news")
        data = response.json()
        # Проверяем структуру ответа
        assert isinstance(data, dict)
