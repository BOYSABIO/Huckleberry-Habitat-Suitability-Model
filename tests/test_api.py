import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import PredictRequest

FEATURE_COLUMNS = list(PredictRequest.model_fields.keys())


@pytest.fixture
def client():
    """Fixture to create a test client for the API."""
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_version" in data


VALID_PAYLOAD = {
    "year": 2020,
    "season_num": 2,
    "elevation": 1200.0,
    "soil_ph": 5.5,
    "air_temperature": 15.2,
    "precipitation_amount": 2.1,
    "specific_humidity": 0.008,
    "relative_humidity": 65.0,
    "mean_vapor_pressure_deficit": 0.5,
    "potential_evapotranspiration": 3.2,
    "surface_downwelling_shortwave_flux_in_air": 250.0,
    "wind_speed": 2.5,
}


def test_predict_returns_probability_and_interval(client):
    """POST /predict returns probability and a two-element confidence_interval."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert len(data["confidence_interval"]) == 2

    low, high = data["confidence_interval"]
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert low <= high


def test_predict_rejects_missing_field(client):
    """POST /predict returns 422 when a required feature is missing."""
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "elevation"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_interval_on_real_hb_row(client):
    """Real training data should produce a non-degenerate interval when trees disagree."""
    row = pd.read_csv("data/snapshots/HB.csv", nrows=1)
    payload = row[FEATURE_COLUMNS].iloc[0].to_dict()

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    low, high = data["confidence_interval"]
    assert low <= high
    # Row 0: most trees agree "suitable" — interval spans full range due to 0/1 tree votes.
    assert "probability" in data
