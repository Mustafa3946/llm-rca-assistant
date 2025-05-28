# =============================================================================
# app_streamlit.py
#
# Purpose:
#   This script launches a Streamlit-based web interface for the RCA LLM Assistant.
#   Users can enter natural language queries related to ETL failures or log analysis,
#   and the app will use a Retrieval-Augmented Generation (RAG) pipeline to return
#   relevant root cause analysis results based on local log data and embeddings.
# =============================================================================

import os
import sys

# Disable Streamlit's file watcher to improve compatibility and performance on some systems
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"

# Add the project root to sys.path to allow absolute imports from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Streamlit and the main RAG pipeline function
import streamlit as st
from src.rag_engine import rca_pipeline

# Configure Streamlit page settings
st.set_page_config(page_title="RCA LLM Assistant", page_icon="🛠️")
st.title("RCA LLM Assistant")

# User input: query text area and top_k slider for number of results
query = st.text_area("Enter your RCA query:", height=150)
top_k = st.slider("Number of top logs to retrieve (top_k):", min_value=1, max_value=10, value=3)

# Run RCA pipeline when button is clicked
if st.button("Run RCA"):
    if not query.strip():
        st.warning("Please enter a query to run the RCA.")
    else:
        with st.spinner("Running RCA pipeline..."):
            results = rca_pipeline(query, top_k)
            st.subheader("RCA Results:")
            for idx, result in enumerate(results, 1):
                st.markdown(f"**Result {idx}:**")
                st.write(result)
