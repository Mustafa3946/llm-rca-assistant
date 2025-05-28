# =============================================================================
# preprocess.py
#
# Purpose:
#   This script ingests raw ETL log data, cleans and normalizes the log messages,
#   and extracts structured features for downstream tasks such as embedding,
#   similarity search, and model inference. The processed features are saved to
#   a CSV file, serving as a simple "ML Feature Store" for the RCA assistant.
# =============================================================================

import pandas as pd
from pathlib import Path


def load_logs(file_path: str) -> pd.DataFrame:
    # Reads logs from a JSON file, simulating ingestion from systems like CloudWatch, Jira, etc.
    """
    Loads structured ETL logs from a JSON file.
    Raises an error if the file is not in JSON format.
    """
    if not file_path.endswith(".json"):
        raise ValueError("Expected a .json log file")
    df = pd.read_json(file_path)
    return df


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    # Cleans and normalizes the 'message' field for downstream processing.
    # - Drops rows with missing messages.
    # - Parses the 'timestamp' column to datetime.
    # - Removes special characters and lowercases the message text.
    """
    Cleans and normalizes message field for downstream processing.
    """
    df = df.dropna(subset=["message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cleaned_message"] = df["message"].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True).str.lower()
    return df


def create_feature_store(df: pd.DataFrame, out_path: str = "outputs/feature_store.csv"):
    # Extracts key elements (e.g., error type) from logs and saves them to a CSV file.
    # - Derives 'error_type' as the first word of the cleaned message.
    # - Saves selected columns to the specified output path.
    # - Ensures the output directory exists.
    """
    Extracts simple features from log messages and saves them to CSV.
    """
    df["error_type"] = df["cleaned_message"].apply(lambda x: x.split(" ")[0])
    features = df[["timestamp", "error_code", "error_type", "cleaned_message"]]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"Feature store saved to: {out_path}")


if __name__ == "__main__":
    # Main execution flow:
    # 1. Load raw logs from the sample JSON file.
    # 2. Clean and normalize the logs.
    # 3. Extract features and save them to the feature store CSV.
    input_log_path = "data/sample_logs.json"
    logs_df = load_logs(input_log_path)
    cleaned_df = clean_logs(logs_df)
    create_feature_store(cleaned_df)
