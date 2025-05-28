'''
faiss_index.py

Purpose:
    This script loads dense vector embeddings from a CSV file, builds a FAISS index for efficient similarity search,
    and saves the index locally. It also demonstrates how to query the index using a sample embedding and prints
    the top closest log messages with their distances. The script is intended for use in the RCA assistant pipeline
    to enable fast retrieval of relevant log entries based on semantic similarity.
'''

import faiss
import numpy as np
import pandas as pd

EMBEDDINGS_PATH = "outputs/embeddings.csv"
FAISS_INDEX_PATH = "outputs/faiss.index"
LOGS_PATH = "outputs/feature_store.csv"

def build_faiss_index(embeddings: np.ndarray, dim: int, index_path: str):
    # Creates a FAISS index using L2 (Euclidean) distance and saves it to disk.
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")
    return index

def load_faiss_index(index_path: str):
    # Loads a FAISS index from the specified file path.
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from: {index_path}")
    return index

def query_index(index, query_embedding: np.ndarray, k=5):
    # Searches the FAISS index for the top-k closest vectors to the query embedding.
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def main():
    # Main execution flow:
    # 1. Load embeddings from the CSV file.
    # 2. Build and save the FAISS index.
    # 3. Load the feature store logs for reference.
    # 4. Run a demo query using the first embedding as the query vector.
    # 5. Print the top 3 closest log messages and their distances.

    # Load embeddings (metadata columns are first 4 columns; embeddings start from column 5)
    df = pd.read_csv(EMBEDDINGS_PATH)
    embeddings = df.iloc[:, 4:].to_numpy().astype('float32')
    dim = embeddings.shape[1]

    # Build and save FAISS index
    index = build_faiss_index(embeddings, dim, FAISS_INDEX_PATH)

    # Load feature store logs for reference to original messages
    logs_df = pd.read_csv(LOGS_PATH)

    # Example: Query using the embedding of the first log message
    sample_text_embedding = embeddings[0].reshape(1, -1)  # Use first embedding as query

    distances, indices = query_index(index, sample_text_embedding, k=3)
    print("Query results:")
    for dist, idx in zip(distances[0], indices[0]):
        print(f"Distance: {dist:.4f} | Log: {logs_df.iloc[idx]['cleaned_message']}")

if __name__ == "__main__":
    main()
