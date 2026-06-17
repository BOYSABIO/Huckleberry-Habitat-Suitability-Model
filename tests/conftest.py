"""Shared pytest configuration."""

import os
from pathlib import Path

# Small committed model for CI and fresh clones (same 12-feature API schema).
TEST_MODEL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "test_model.joblib"


def pytest_configure(config):
    """Set MODEL_PATH before test modules import the FastAPI app."""
    if not os.getenv("MODEL_PATH") and TEST_MODEL_FIXTURE.exists():
        os.environ["MODEL_PATH"] = str(TEST_MODEL_FIXTURE)
