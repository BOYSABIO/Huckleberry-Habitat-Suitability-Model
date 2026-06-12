import pandas as pd
from src.model.predictor import load_predictor_from_path

predictor = load_predictor_from_path("models/huckleberry_model_v13_20260612_111519.joblib")

# 1 Wat is the API contract?
print(predictor.feature_names)

# 2 One row - all zeros is ok for the smoke test
row = {name: 0.0 for name in predictor.feature_names}
df = pd.DataFrame([row])
df_test = df.drop(columns=["elevation"])
print(len(list(df.columns)))

# 3 Point estimate
proba = predictor.predict_proba(df)
try:
    proba_test = predictor.predict_proba(df_test)
    print("P(suitable) without elevation:", proba_test[0, 1])
except ValueError as e:
    print(f"Error: {e}")
print("P(suitable):", proba[0, 1])

# What the API will return (Phase 3)
result = predictor.predict_with_interval(df)
try:
    result_test = predictor.predict_with_interval(df_test)
    print("TEST:", result_test.probabilities[0], result_test.confidence_intervals[0])
except ValueError as e:
    print(f"Error: {e}")
print(result.probabilities[0], result.confidence_intervals[0])

# Real Data Test
real_data_test = pd.read_csv("data/snapshots/HB.csv", nrows=5)
row = real_data_test.iloc[0:1]
print(predictor.predict_proba(row)[0, 1])
print(predictor.predict_with_interval(row).confidence_intervals[0])