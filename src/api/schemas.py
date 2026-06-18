from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    """Request body for the prediction endpoint."""
    year: int
    season_num: int = Field(ge=0, le=3, description="0=winter, 1=spring, 2=summer, 3=fall")
    elevation: float
    soil_ph: float
    air_temperature: float
    precipitation_amount: float
    specific_humidity: float
    relative_humidity: float
    mean_vapor_pressure_deficit: float
    potential_evapotranspiration: float
    surface_downwelling_shortwave_flux_in_air: float
    wind_speed: float

class PredictResponse(BaseModel):
    """Response body for the prediction endpoint."""
    probability: float = Field(ge=0.0, le=1.0, description="Probability of suitability")
    confidence_interval: List[float] = Field(
        min_length=2,
        max_length=2,
        description=(
            "2.5th and 97.5th percentile of per-tree suitability probabilities. "
            "Wide intervals (e.g. [0, 1]) mean trees disagree; not a classical CI."
        ),
    )