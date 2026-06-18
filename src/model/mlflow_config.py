"""MLflow naming constants (no mlflow import — safe for API/tests)."""

REGISTERED_MODEL_NAME = "huckleberry-habitat"
MLFLOW_EXPERIMENT_NAME = "huckleberry-training"
DEFAULT_MLFLOW_MODEL_ALIAS = "production"
DEFAULT_MLFLOW_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{DEFAULT_MLFLOW_MODEL_ALIAS}"
