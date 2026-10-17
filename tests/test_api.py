"""Automated tests for the Iris prediction API."""
import pytest
from fastapi.testclient import TestClient

from src.main import MAX_BATCH_SIZE, app

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VIRGINICA = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}


@pytest.fixture(scope="module")
def client():
    # Using "with" runs the startup code, so the model is loaded like in production
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "/predict" in response.json()["endpoints"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["model_type"] == "DecisionTreeClassifier"
    assert info["classes"] == ["setosa", "versicolor", "virginica"]
    assert info["test_accuracy"] > 0.9


def test_predict_setosa(client):
    response = client.post("/predict", json=SETOSA)
    assert response.status_code == 200
    result = response.json()
    assert result["species"] == "setosa"
    assert result["class_id"] == 0
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-3


def test_predict_virginica(client):
    result = client.post("/predict", json=VIRGINICA).json()
    assert result["species"] == "virginica"


def test_negative_value_rejected(client):
    bad = {**SETOSA, "petal_width": -1}
    assert client.post("/predict", json=bad).status_code == 422


def test_unrealistic_value_rejected(client):
    bad = {**SETOSA, "sepal_length": 50}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_rejected(client):
    bad = {k: v for k, v in SETOSA.items() if k != "sepal_width"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_predict(client):
    response = client.post("/predict/batch", json={"samples": [SETOSA, VIRGINICA]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [p["species"] for p in body["predictions"]] == ["setosa", "virginica"]


def test_batch_empty_rejected(client):
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422


def test_batch_too_large_rejected(client):
    samples = [SETOSA] * (MAX_BATCH_SIZE + 1)
    assert client.post("/predict/batch", json={"samples": samples}).status_code == 422
