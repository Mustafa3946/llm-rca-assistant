# =============================================================================
# test_retriever.py
#
# Purpose:
#   This script contains unit tests for the log retrieval function used in the RCA LLM Assistant.
#   It verifies that the get_top_k_logs function returns the expected number of results and that
#   each result contains a valid distance and log message. The tests help ensure the retrieval
#   component of the RAG pipeline is functioning correctly.
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

from src.retriever import get_top_k_logs

def test_top_k_logs():
    # Test that get_top_k_logs returns a list of (distance, log_message) tuples for a sample query.
    query = "connection timeout error"
    results = get_top_k_logs(query, k=3)

    assert isinstance(results, list)
    assert len(results) == 3
    for dist, log in results:
        assert isinstance(dist, float)
        assert isinstance(log, str)
        assert len(log) > 0

if __name__ == "__main__":
    test_top_k_logs()
    print("test_top_k_logs passed!")
