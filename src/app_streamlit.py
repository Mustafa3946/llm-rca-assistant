import os
import sys

# Disable file watcher (for future compatibility)
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"

# Set up sys.path to allow absolute imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Streamlit app
import streamlit as st
from src.rag_engine import rca_pipeline

st.set_page_config(page_title="RCA LLM Assistant", page_icon="🛠️")
st.title("RCA LLM Assistant")

query = st.text_area("Enter your RCA query:", height=150)
top_k = st.slider("Number of top logs to retrieve (top_k):", min_value=1, max_value=10, value=3)

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
