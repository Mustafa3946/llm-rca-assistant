import warnings

warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated and has been renamed to numpy._core._multiarray_umath",
    category=DeprecationWarning,
)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import rca_pipeline

def test_rca_pipeline():
    query = "network timeout error"
    response = rca_pipeline(query)
    
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], str)
    assert "Mocked LLM Response" in response[0]

if __name__ == "__main__":
    test_rca_pipeline()
    print("test_rca_pipeline passed!")
