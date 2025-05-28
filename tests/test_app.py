# =============================================================================
# test_app.py
#
# Purpose:
#   This script contains unit tests for the FastAPI-based RCA LLM Assistant API.
#   It verifies that the health check endpoint and the RCA query endpoint both function as expected.
#   The tests use FastAPI's TestClient to simulate HTTP requests and validate API responses.
# =============================================================================

import warnings

warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated and has been renamed to numpy._core._multiarray_umath",
    category=DeprecationWarning,
)

import sys
import os
# Add the project root to sys.path to allow absolute imports from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check():
    # Test the root ("/") health check endpoint to ensure the API is running.
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "RCA LLM API is running"}

def test_query_endpoint():
    # Test the RCA query endpoint with a sample payload.
    # Checks that the response contains a "results" list of length 1.
    payload = {
        "query": "disk space error",
        "top_k": 2
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "results" in json_data
    assert isinstance(json_data["results"], list)
    assert len(json_data["results"]) == 1
