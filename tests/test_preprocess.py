# =============================================================================
# test_preprocess.py
#
# Purpose:
#   This script contains unit tests for the data preprocessing functions used in the RCA LLM Assistant.
#   It verifies that log loading, cleaning, and feature store creation work as expected.
#   The tests ensure that the preprocessing pipeline produces valid, well-structured outputs for downstream tasks.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src import preprocess

TEST_INPUT_FILE = "data/sample_logs.json"
TEST_OUTPUT_FILE = "outputs/test_feature_store.csv"

def test_load_logs():
    # Test that logs are loaded from the JSON file into a DataFrame with required columns.
    df = preprocess.load_logs(TEST_INPUT_FILE)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "message" in df.columns

def test_clean_logs():
    # Test that log cleaning produces a DataFrame with normalized messages and parsed timestamps.
    df = preprocess.load_logs(TEST_INPUT_FILE)
    cleaned_df = preprocess.clean_logs(df)
    
    assert "cleaned_message" in cleaned_df.columns
    # Ensure cleaned_message contains only lowercase alphanumeric characters and spaces.
    assert cleaned_df["cleaned_message"].str.contains("[^a-z0-9 ]", regex=True).sum() == 0
    # Ensure timestamp column is of datetime type.
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["timestamp"])

def test_create_feature_store():
    # Test that the feature store is created and contains the expected columns.
    df = preprocess.load_logs(TEST_INPUT_FILE)
    cleaned_df = preprocess.clean_logs(df)
    preprocess.create_feature_store(cleaned_df, out_path=TEST_OUTPUT_FILE)

    assert os.path.exists(TEST_OUTPUT_FILE)

    out_df = pd.read_csv(TEST_OUTPUT_FILE)
    expected_cols = ["timestamp", "error_code", "error_type", "cleaned_message"]
    for col in expected_cols:
        assert col in out_df.columns

if __name__ == "__main__":
    test_load_logs()
    print("test_load_logs passed")

    test_clean_logs()
    print("test_clean_logs passed")

    test_create_feature_store()
    print("test_create_feature_store passed")
