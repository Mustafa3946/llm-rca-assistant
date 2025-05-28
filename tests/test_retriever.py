import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import get_top_k_logs

def test_top_k_logs():
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
