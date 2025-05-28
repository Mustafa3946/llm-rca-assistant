# =============================================================================
# test_rag_engine.py
#
# Purpose:
#   This script contains unit tests for the core Retrieval-Augmented Generation (RAG) pipeline
#   used in the RCA LLM Assistant. It verifies that the rca_pipeline function returns a valid
#   response for a sample query and that the mocked LLM response is present in the output.
#   The tests help ensure the RAG pipeline is functioning as expected for downstream tasks.
# =============================================================================

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
    # Test that the RAG pipeline returns a valid response for a sample query.
    query = "network timeout error"
    response = rca_pipeline(query)
    
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], str)
    assert "Mocked LLM Response" in response[0]

if __name__ == "__main__":
    test_rca_pipeline()
    print("test_rca_pipeline passed!")
