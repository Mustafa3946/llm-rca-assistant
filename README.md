# LLM-RCA Assistant

A lightweight demo of an LLM-powered Root Cause Analysis (RCA) system for analyzing cloud ETL failures. This project simulates log ingestion, processing, semantic search (RAG), and natural language root cause summaries using open-source tools and minimal cloud usage.

---

## Features

- Log ingestion & preprocessing (simulated)
- ML Feature Store integration for failure metadata
- Embedding generation via open-source models (e.g., `all-MiniLM`)
- Vector store powered by FAISS (local, low cost)
- Retrieval-Augmented Generation (RAG) using LangChain
- RCA prompt templates for failure analysis
- Streamlit UI for interacting with the assistant

---

## Architecture

```text
+---------------------+
|  Simulated Logs     |
|  (CSV, JSON, etc)   |
+----------+----------+
           |
     +-----v------+
     | Preprocess |
     | Clean logs |
     +-----+------+
           |
+----------v-----------+
| Feature Store (CSV)  |
| Simulated metadata   |
+----------+-----------+
           |
+----------v----------+        +------------------------+
| Embedding Generator | -----> | Vector Store (FAISS)   |
| (e.g. SentenceTransformers)  +-----------+------------+
                                             |
                                   +---------v---------+
                                   | LangChain RAG     |
                                   | (query+context)   |
                                   +---------+---------+
                                             |
                                   +---------v---------+
                                   | LLM (e.g. Claude, |
                                   | GPT via API)     |
                                   +---------+---------+
                                             |
                                   +---------v---------+
                                   | Streamlit UI      |
                                   | RCA Answer + Fix  |
                                   +-------------------+
