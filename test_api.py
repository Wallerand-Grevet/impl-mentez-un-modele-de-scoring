import pytest
import json
from app import app, preprocess_features, selected_features
import pandas as pd

# Client de test Flask
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

# Test de la route GET /
def test_home(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "API MLflow active" in res.json["message"]

# Test de preprocessing minimal
def test_preprocess_features():
    sample = pd.DataFrame([{
        "AMT_INCOME_TOTAL": 100000,
        "AMT_CREDIT": 200000,
        "AMT_ANNUITY": 10000,
        "CNT_FAM_MEMBERS": 2,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -4000,
        "DAYS_REGISTRATION": -4000,
        "DAYS_ID_PUBLISH": -1000
    }])
    processed = preprocess_features(sample)
    assert isinstance(processed, pd.DataFrame)
    assert set(selected_features).issubset(processed.columns)

# Test complet sur /predict
def test_predict_valid(client):
    payload = {
        "features": [{
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000,
            "AMT_ANNUITY": 10000,
            "CNT_FAM_MEMBERS": 2,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -4000,
            "DAYS_REGISTRATION": -4000,
            "DAYS_ID_PUBLISH": -1000
        }]
    }
    res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert "probability" in data
    assert "prediction" in data
    assert "decision" in data

#  Test avec JSON invalide
def test_predict_invalid_json(client):
    res = client.post("/predict", data=json.dumps({"bad": "structure"}), content_type="application/json")
    assert res.status_code == 400
    assert "error" in res.json
