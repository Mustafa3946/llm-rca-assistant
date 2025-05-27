# ingest raw logs, clean and normalize them, and extract structured features that can be used for downstream tasks like embedding, similarity search, and model inference.

import pandas as pd
from pathlib import Path


def load_logs(file_path: str) -> pd.DataFrame:
    # Reads logs from .csv or .json formats, simulating ingestion from systems like CloudWatch, Jira, etc.
    """
    Loads structured ETL logs from a JSON file.
    """
    if not file_path.endswith(".json"):
        raise ValueError("Expected a .json log file")
    df = pd.read_json(file_path)
    return df


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    # Normalizes message text by removing special characters, lowering case, and ensuring timestamps are parsed correctly.
    """
    Cleans and normalizes message field for downstream processing.
    """
    df = df.dropna(subset=["message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cleaned_message"] = df["message"].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True).str.lower()
    return df


def create_feature_store(df: pd.DataFrame, out_path: str = "outputs/feature_store.csv"):
    # Extracts key elements (e.g., error type) from logs and saves them to outputs/feature_store.csv — a simplified "ML Feature Store".
    """
    Extracts simple features from log messages and saves them to CSV.
    """
    df["error_type"] = df["cleaned_message"].apply(lambda x: x.split(" ")[0])
    features = df[["timestamp", "error_code", "error_type", "cleaned_message"]]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"Feature store saved to: {out_path}")


if __name__ == "__main__":
    input_log_path = "data/sample_logs.json"
    logs_df = load_logs(input_log_path)
    cleaned_df = clean_logs(logs_df)
    create_feature_store(cleaned_df)
