import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests


BASE_URL = os.getenv("TEST_API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT_S = float(os.getenv("TEST_API_TIMEOUT_S", "70"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get(path: str) -> requests.Response:
    return requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT_S)


@pytest.fixture(scope="session")
def api_ready() -> bool:
    try:
        r = _get("/health")
        return r.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# 1) Functional correctness
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_functional_predictions_and_actuals_contract(api_ready: bool):
    if not api_ready:
        pytest.skip(f"API not reachable at {BASE_URL}")

    r = _get("/api/predictions?sector=pharma&month=2025-01")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list) and payload, "predictions payload should be non-empty list"

    sample = payload[0]
    for key in [
        "partnerCode",
        "partner",
        "export_forecast",
        "export_actual",
        "import_forecast",
        "import_actual",
        "export_change",
        "import_change",
        "confidence",
        "risk_level",
    ]:
        assert key in sample, f"missing prediction field: {key}"

    assert isinstance(sample["export_forecast"], (int, float))
    assert isinstance(sample["import_forecast"], (int, float))
    assert isinstance(sample["confidence"], (int, float))
    assert sample["risk_level"] in {"low", "medium", "high"}


@pytest.mark.integration
def test_functional_explainability_and_simulation_contract(api_ready: bool):
    if not api_ready:
        pytest.skip(f"API not reachable at {BASE_URL}")

    # Explainability
    ex = _get("/api/explainability?partner=DEU")
    assert ex.status_code == 200
    ex_payload = ex.json()
    assert "features" in ex_payload and isinstance(ex_payload["features"], list)
    assert "blurb" in ex_payload and isinstance(ex_payload["blurb"], str)

    # Simulation
    sim = requests.post(
        f"{BASE_URL}/api/v1/simulate",
        json={
            "target_country": "DEU",
            "feature": "sentiment",
            "change_percent": 5.0,
            "sector": "pharma",
            "month": "2025-01",
        },
        timeout=TIMEOUT_S,
    )
    assert sim.status_code == 200
    sim_payload = sim.json()
    for key in ["baseline", "counterfactual", "delta", "pct_impact", "global_impact", "explanation"]:
        assert key in sim_payload, f"missing simulation field: {key}"


@pytest.mark.integration
def test_functional_news_endpoint_returns_valid_items(api_ready: bool):
    if not api_ready:
        pytest.skip(f"API not reachable at {BASE_URL}")

    try:
        r = _get("/api/news?sector=pharma&month=2025-01")
    except requests.ReadTimeout:
        pytest.skip("News endpoint timed out in current environment")

    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    if payload:
        item = payload[0]
        for key in ["id", "title", "source", "url", "date", "sentiment", "relevance_score"]:
            assert key in item, f"missing news field: {key}"


# ---------------------------------------------------------------------------
# 2) Data integrity and consistency
# ---------------------------------------------------------------------------
def test_data_integrity_edges_schema_and_non_empty():
    edges_path = PROJECT_ROOT / "data/processed/edges.csv"
    assert edges_path.exists(), "data/processed/edges.csv must exist"

    df = pd.read_csv(edges_path)
    assert len(df) > 0, "edges.csv should not be empty"

    required_cols = {
        "source_iso3",
        "target_iso3",
        "year",
        "month",
        "sector",
        "trade_value_usd",
        "sentiment_norm",
    }
    assert required_cols.issubset(df.columns), f"missing columns: {required_cols - set(df.columns)}"


def test_data_integrity_valid_2025_partners_have_both_trade_directions():
    partners_path = PROJECT_ROOT / "data/processed/valid_2025_partners.json"
    edges_path = PROJECT_ROOT / "data/processed/edges.csv"
    if not partners_path.exists():
        pytest.skip("valid_2025_partners.json not found")

    with partners_path.open() as f:
        partners = set(json.load(f))
    assert partners, "valid_2025_partners.json should contain partner codes"

    df = pd.read_csv(edges_path)
    e2025 = df[df["year"] == 2025]

    missing = []
    for p in partners:
        has_export = not e2025[(e2025["source_iso3"] == "IND") & (e2025["target_iso3"] == p)].empty
        has_import = not e2025[(e2025["source_iso3"] == p) & (e2025["target_iso3"] == "IND")].empty
        if not (has_export and has_import):
            missing.append(p)

    assert not missing, f"partners missing 2025 bidirectional trade rows: {missing[:10]}"


# ---------------------------------------------------------------------------
# 3) Operational reliability
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_operational_health_and_graceful_degradation_flags(api_ready: bool):
    if not api_ready:
        pytest.skip(f"API not reachable at {BASE_URL}")

    r = _get("/health")
    assert r.status_code == 200
    payload = r.json()

    assert payload["status"] in {"healthy", "degraded"}
    # Service should still answer health regardless of cache/db availability.
    assert isinstance(payload.get("redis_available"), bool)
    assert isinstance(payload.get("postgres_available"), bool)
    assert isinstance(payload.get("model_loaded"), bool)
    assert isinstance(payload.get("data_loaded"), bool)


@pytest.mark.integration
def test_operational_predictions_still_work_without_cache_or_db(api_ready: bool):
    if not api_ready:
        pytest.skip(f"API not reachable at {BASE_URL}")

    health = _get("/health").json()
    # Even when Redis/Postgres are down, predictions should still be served.
    if (not health.get("redis_available", True)) or (not health.get("postgres_available", True)):
        pred = _get("/api/predictions?sector=pharma&month=2025-01")
        assert pred.status_code == 200
        assert isinstance(pred.json(), list)

