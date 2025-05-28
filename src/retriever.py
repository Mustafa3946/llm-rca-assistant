import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def get_top_k_logs(query: str, k: int = 3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vec = model.encode([query])

    index = faiss.read_index("outputs/faiss.index")
    D, I = index.search(query_vec, k)

    logs_df = pd.read_csv("outputs/embeddings.csv")
    logs = logs_df["cleaned_message"]
    
    return [(float(d), logs[i]) for d, i in zip(D[0], I[0])]