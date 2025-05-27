'''
Loads your embeddings from the CSV file (skipping metadata columns).
Builds a FAISS index with L2 distance.
Saves the index file locally.
Loads the feature store logs so you can reference the original messages.
Runs a demo query using the first embedding vector as the query.
Prints out the top 3 closest logs by similarity with their distances.
'''

import faiss
import numpy as np
import pandas as pd

EMBEDDINGS_PATH = "outputs/embeddings.csv"
FAISS_INDEX_PATH = "outputs/faiss.index"
LOGS_PATH = "outputs/feature_store.csv"

def build_faiss_index(embeddings: np.ndarray, dim: int, index_path: str):
    # Create a FAISS index (Flat L2)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")
    return index

def load_faiss_index(index_path: str):
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from: {index_path}")
    return index

def query_index(index, query_embedding: np.ndarray, k=5):
    # Search the FAISS index for top-k closest vectors
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def main():
    # Load embeddings
    df = pd.read_csv(EMBEDDINGS_PATH)
    # The first columns are metadata, embeddings start from column 4 (index 3)
    embeddings = df.iloc[:, 4:].to_numpy().astype('float32')
    dim = embeddings.shape[1]

    # Build and save FAISS index
    index = build_faiss_index(embeddings, dim, FAISS_INDEX_PATH)

    # Load feature store logs
    logs_df = pd.read_csv(LOGS_PATH)

    # Example: Query by embedding of a sample text (for demo)
    sample_text_embedding = embeddings[0].reshape(1, -1)  # Using first embedding as query

    distances, indices = query_index(index, sample_text_embedding, k=3)
    print("Query results:")
    for dist, idx in zip(distances[0], indices[0]):
        print(f"Distance: {dist:.4f} | Log: {logs_df.iloc[idx]['cleaned_message']}")

if __name__ == "__main__":
    main()
