import pytest
from fastapi.testclient import TestClient

from src.api.main import app


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

def test_predict_returns_probability(client):
    """Test the predict endpoint returns a probability."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_rejects_missing_field(client):
    """Test the predict endpoint rejects a missing field."""
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "elevation"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
