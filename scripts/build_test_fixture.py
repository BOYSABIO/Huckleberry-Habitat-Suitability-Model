"""Build tests/fixtures/test_model.joblib for CI (small RF on HB sample)."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "year",
    "season_num",
    "elevation",
    "soil_ph",
    "air_temperature",
    "precipitation_amount",
    "specific_humidity",
    "relative_humidity",
    "mean_vapor_pressure_deficit",
    "potential_evapotranspiration",
    "surface_downwelling_shortwave_flux_in_air",
    "wind_speed",
]

df = pd.read_csv("data/snapshots/HB.csv", nrows=300)
X = df[FEATURES].fillna(0)
y = df["occurrence"]
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(Xs, y)

payload = {
    "model": {"estimator": rf, "scaler": scaler, "feature_names": FEATURES},
    "feature_names": FEATURES,
    "model_type": "random_forest",
    "parameters": {"n_estimators": 10},
    "version_id": "test_fixture",
}

out = "tests/fixtures/test_model.joblib"
joblib.dump(payload, out)
print(f"Wrote {out}")
