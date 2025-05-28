# =============================================================================
# retriever.py
#
# Purpose:
#   This module provides a function to retrieve the top-k most semantically similar log messages
#   to a user query using a FAISS index and SentenceTransformer embeddings. It is used as part of
#   the RAG pipeline to efficiently find relevant logs for root cause analysis.
# =============================================================================

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def get_top_k_logs(query: str, k: int = 3):
    # Encodes the user query into a dense vector using SentenceTransformer.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vec = model.encode([query])

    # Loads the FAISS index from disk and searches for the top-k closest embeddings.
    index = faiss.read_index("outputs/faiss.index")
    D, I = index.search(query_vec, k)

    # Loads the log messages from the embeddings CSV file.
    logs_df = pd.read_csv("outputs/embeddings.csv")
    logs = logs_df["cleaned_message"]
    
    # Returns a list of tuples: (distance, log_message) for the top-k results.
    return [(float(d), logs[i]) for d, i in zip(D[0], I[0])]