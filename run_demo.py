# run_demo.py

"""
End-to-end pipeline:
1. Load and clean logs
2. Create feature store
3. Embed cleaned logs using SBERT
4. Build and save FAISS index
5. Accept user query and return top-K most similar logs
"""

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.preprocess import load_logs, clean_logs, create_feature_store

# Paths
RAW_LOG_PATH = "data/sample_logs.json"
FEATURE_STORE_PATH = "outputs/feature_store.csv"
EMBEDDING_PATH = "outputs/embeddings.csv"
FAISS_INDEX_PATH = "outputs/faiss.index"

# Step 1: Preprocess
def preprocess_logs():
    df = load_logs(RAW_LOG_PATH)
    cleaned_df = clean_logs(df)
    create_feature_store(cleaned_df)
    return cleaned_df

# Step 2: Embed logs
def embed_logs(df: pd.DataFrame, out_path: str = EMBEDDING_PATH):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["cleaned_message"].tolist(), show_progress_bar=True)
    embedding_df = df.copy()
    embedding_cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
    embedding_df[embedding_cols] = embeddings
    embedding_df.to_csv(out_path, index=False)
    print(f"Embeddings saved to: {out_path}")
    return embeddings

# Step 3: Build FAISS index
def build_faiss_index(embeddings: np.ndarray, dim: int, index_path: str = FAISS_INDEX_PATH):
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")
    return index

# Step 4: Query
def query_logs(index, logs_df, query_text: str, k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query_text]).astype("float32")
    distances, indices = index.search(query_vec, k)

    print(f"\nTop {k} logs similar to: \"{query_text}\"")
    for dist, idx in zip(distances[0], indices[0]):
        print(f"Distance: {dist:.4f} | Log: {logs_df.iloc[idx]['cleaned_message']}")

if __name__ == "__main__":
    print(">> Running full RCA-LLM pipeline...\n")
    df = preprocess_logs()
    embeddings = embed_logs(df)
    index = build_faiss_index(embeddings.astype("float32"), embeddings.shape[1])

    # Demo query
    query_logs(index, df, query_text="connection timeout error", k=3)
