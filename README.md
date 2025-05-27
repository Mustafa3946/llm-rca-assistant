# RCA-LLM-Demo

A lightweight, cost-free, local prototype of a Root Cause Analysis (RCA) assistant using LLMs and Retrieval-Augmented Generation (RAG). This demo avoids cloud costs by running locally with open-source tools.

---

## Architecture Overview

```ascii
+-------------------------------------------------------------+
|                        Data Sources                         |
|  Local Logs | ETL Metadata | Simulated Failure Tickets      |
+------------------+------------------+------------------------+
           |                |                  |
           +----------- Local Preprocessing --------------+
                            |
                 +----------v-----------+
                 |     Cleaned Logs     |
                 |   (Saved to CSV)     |
                 +----------+-----------+
                            |
          +----------------+-------------------+
          |                                    |
+---------v----------+           +-------------v---------------+
| ML Feature Store   |           | Embedding Generator (SBERT) |
| (CSV file)         |           +-------------+---------------+
+--------------------+                         |
                                   +----------v------------+
                                   | Vector DB (FAISS)     |
                                   +----------+------------+
                                              |
                                +-------------v--------------+
                                |  RAG Engine (LangChain)     |
                                | - Query Logs & Docs         |
                                | - Construct Prompt          |
                                +-------------+---------------+
                                              |
                                +-------------v---------------+
                                | LLM (Local Model or API)     |
                                | - Natural Language RCA       |
                                +-------------+----------------+
                                              |
                              +---------------v----------------+
                              |   Streamlit / FastAPI Frontend  |
                              | - RCA Output + Recommendations  |
                              +----------------+----------------+
                                               |
                         +---------------------v-------------------+
                         |     Observability (Local logs only)     |
                         +------------------------------------------+
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing

```bash
python src/preprocess.py
```

This will create a CSV-based feature store from a sample log file.

---

## Folder Structure

```
├── data/
│   ├── sample_logs.csv
│   └── feature_store.csv
├── src/
│   └── preprocess.py
├── README.md
└── requirements.txt
```

---

## Next Steps

- Step 3: Generate embeddings (SBERT)
- Step 4: Store in FAISS
- Step 5: Implement RAG + UI

---
