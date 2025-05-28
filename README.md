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

### 3. Run unit tests

```bash
pytest tests/test_preprocess.py
pytest tests/test_retriever.py
pytest tests/test_rag_engine.py
pytest tests/test_app.py
```


### 4. Run the Streamlit UI

```bash
streamlit run src/app_streamlit.py
```

---

## Folder Structure

```
├── data/
│   └── sample_logs.json
├── outputs/
│   ├── embeddings.csv
│   ├── faiss.index
│   └── feature_store.csv
├── src/
│   ├── app.py
│   ├── app_streamlit.py
│   ├── embed.py
│   ├── faiss_index.py
│   ├── preprocess.py
│   ├── rag_engine.py
│   └── retriever.py
├── models/
│   └── llama-2-7b.Q4_K_M.gguf
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Sample Queries You Can Try
You can enter any of the following queries into the Streamlit interface to see how the RCA pipeline responds:
What caused the ETL job to timeout while connecting to the source database?
What is causing the data validation error with missing customer IDs?
Why can't the system write data to the S3 bucket?
Why did the ETL job fail to complete successfully?

---