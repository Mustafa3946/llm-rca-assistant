'''
embed.py

Purpose:
    This script loads cleaned log messages from the feature store (outputs/feature_store.csv),
    generates dense vector embeddings for each message using a SentenceTransformer model,
    and saves the resulting embeddings in a format compatible with vector databases (e.g., CSV for FAISS).
    These embeddings are used for efficient similarity search and retrieval in the RCA assistant pipeline.
'''

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

def load_feature_store(path: str) -> pd.DataFrame:
    # Loads the feature store CSV containing cleaned log messages and features.
    return pd.read_csv(path)

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    # Uses SentenceTransformer to generate dense semantic embeddings for a list of texts.
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def save_embeddings(df: pd.DataFrame, embeddings: np.ndarray, out_path: str = "outputs/embeddings.csv"):
    # Saves the original DataFrame and the generated embeddings as columns in a single CSV file.
    # This output can be loaded into FAISS or other vector search engines.
    embedding_df = pd.DataFrame(embeddings)
    result_df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)
    result_df.to_csv(out_path, index=False)
    print(f"Embeddings saved to: {out_path}")

if __name__ == "__main__":
    # Main execution flow:
    # 1. Load cleaned log messages from the feature store.
    # 2. Generate vector embeddings for each message.
    # 3. Save the embeddings alongside the original data for downstream use.
    df = load_feature_store("outputs/feature_store.csv")
    texts = df["cleaned_message"].tolist()
    embeddings = generate_embeddings(texts)
    save_embeddings(df, embeddings)
