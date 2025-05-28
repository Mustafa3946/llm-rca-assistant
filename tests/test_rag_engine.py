import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import rca_pipeline

def test_rca_pipeline():
    query = "network timeout error"
    response = rca_pipeline(query)
    assert isinstance(response, str)
    assert "Mocked LLM Response" in response

if __name__ == "__main__":
    test_rca_pipeline()
    print("test_rca_pipeline passed!")
