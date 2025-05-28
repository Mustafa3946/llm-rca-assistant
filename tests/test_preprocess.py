import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src import preprocess

TEST_INPUT_FILE = "data/sample_logs.json"
TEST_OUTPUT_FILE = "outputs/test_feature_store.csv"


def test_load_logs():
    df = preprocess.load_logs(TEST_INPUT_FILE)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "message" in df.columns


def test_clean_logs():
    df = preprocess.load_logs(TEST_INPUT_FILE)
    cleaned_df = preprocess.clean_logs(df)
    
    assert "cleaned_message" in cleaned_df.columns
    assert cleaned_df["cleaned_message"].str.contains("[^a-z0-9 ]", regex=True).sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["timestamp"])


def test_create_feature_store():
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
