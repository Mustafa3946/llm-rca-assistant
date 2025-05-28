'''
Load cleaned logs from the Feature Store (outputs/feature_store.csv)
Generate vector embeddings for each cleaned log message
Save those embeddings in a Vector Database-compatible format (CSV, FAISS, or similar)
'''

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

def load_feature_store(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    # SentenceTransformer	Generates dense semantic vectors from messages.
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def save_embeddings(df: pd.DataFrame, embeddings: np.ndarray, out_path: str = "outputs/embeddings.csv"):
    # Embedding CSV Output	These can be loaded into FAISS or OpenSearch k-NN.
    embedding_df = pd.DataFrame(embeddings)
    result_df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)
    result_df.to_csv(out_path, index=False)
    print(f"Embeddings saved to: {out_path}")

if __name__ == "__main__":
    df = load_feature_store("outputs/feature_store.csv")
    texts = df["cleaned_message"].tolist()
    embeddings = generate_embeddings(texts)
    save_embeddings(df, embeddings)
