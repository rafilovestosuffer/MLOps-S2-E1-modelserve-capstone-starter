import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_model_mock():
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.85, 0.15]])
    return model


def _make_store_mock():
    return MagicMock()


@pytest.fixture(scope="module")
def client():
    """Build a TestClient with all heavy deps mocked out for the entire module."""
    from fastapi.testclient import TestClient

    features_mock = {
        "amt": 100.0, "lat": 33.9, "long": -117.4,
        "city_pop": 50000, "unix_time": 1325376018,
        "merch_lat": 33.99, "merch_long": -117.1,
    }

    with patch("app.main.load_production_model", return_value=(_make_model_mock(), "test-v1")), \
         patch("app.main.get_feature_store", return_value=_make_store_mock()), \
         patch("app.main.get_online_features", return_value=features_mock):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_predict_valid(client):
    resp = client.post("/predict", json={"entity_id": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert "model_version" in data
    assert "timestamp" in data


def test_predict_invalid_input(client):
    # Missing entity_id → 422 Unprocessable Entity
    resp = client.post("/predict", json={"wrong_field": "abc"})
    assert resp.status_code == 422


def test_predict_explain(client):
    resp = client.get("/predict/0?explain=true")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "features_used" in data


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"prediction_requests_total" in resp.content
