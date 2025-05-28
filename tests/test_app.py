import warnings

warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated and has been renamed to numpy._core._multiarray_umath",
    category=DeprecationWarning,
)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "RCA LLM API is running"}

def test_query_endpoint():
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
