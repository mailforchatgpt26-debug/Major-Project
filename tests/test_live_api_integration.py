import os
from typing import Any

import pytest
import requests


BASE_URL = os.getenv("TEST_API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT_S = float(os.getenv("TEST_API_TIMEOUT_S", "70"))


def _get(path: str) -> requests.Response:
    return requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT_S)


@pytest.fixture(scope="session")
def api_reachable() -> bool:
    try:
        r = _get("/health")
        return r.status_code == 200
    except requests.RequestException:
        return False


def _assert_prediction_shape(item: dict[str, Any]) -> None:
    required = [
        "partnerCode",
        "partner",
        "export_forecast",
        "export_change",
        "import_change",
        "confidence",
        "risk_level",
    ]
    for key in required:
        assert key in item, f"missing key: {key}"

    assert isinstance(item["partnerCode"], str) and item["partnerCode"]
    assert isinstance(item["partner"], str) and item["partner"]
    assert isinstance(item["export_forecast"], (int, float))
    assert isinstance(item["export_change"], (int, float))
    assert isinstance(item["import_change"], (int, float))
    assert isinstance(item["confidence"], (int, float))
    assert item["risk_level"] in {"low", "medium", "high"}


@pytest.mark.integration
def test_health_endpoint_contract(api_reachable: bool):
    if not api_reachable:
        pytest.skip(f"API not reachable at {BASE_URL}")

    r = _get("/health")
    assert r.status_code == 200
    payload = r.json()

    assert payload["status"] in {"healthy", "degraded"}
    assert isinstance(payload.get("model_loaded"), bool)
    assert isinstance(payload.get("data_loaded"), bool)
    assert "timestamp" in payload


@pytest.mark.integration
def test_predictions_endpoint_returns_business_fields(api_reachable: bool):
    if not api_reachable:
        pytest.skip(f"API not reachable at {BASE_URL}")

    r = _get("/api/predictions?sector=pharma&month=2025-01")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    assert len(payload) > 0

    _assert_prediction_shape(payload[0])


@pytest.mark.integration
def test_news_endpoint_returns_articles(api_reachable: bool):
    if not api_reachable:
        pytest.skip(f"API not reachable at {BASE_URL}")

    try:
        r = _get("/api/news?sector=pharma&month=2025-01")
    except requests.ReadTimeout:
        pytest.skip("Live news endpoint timed out in current environment")

    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)

    if payload:
        first = payload[0]
        for key in ["id", "title", "source", "url", "date", "sentiment", "relevance_score"]:
            assert key in first, f"missing key: {key}"
        assert isinstance(first["title"], str) and first["title"]
        assert isinstance(first["url"], str)
        assert isinstance(first["sentiment"], (int, float))

