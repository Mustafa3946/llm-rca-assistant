# =============================================================================
# app.py
#
# Purpose:
#   This script provides a FastAPI-based REST API for the RCA LLM Assistant.
#   It exposes endpoints for health checks and for submitting RCA queries.
#   The API receives natural language queries and returns root cause analysis
#   results using a Retrieval-Augmented Generation (RAG) pipeline over local log data.
# =============================================================================

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from src.rag_engine import rca_pipeline

app = FastAPI(title="RCA LLM Assistant API")

class RCAQuery(BaseModel):
    # Request model for RCA queries: includes the query string and top_k parameter
    query: str
    top_k: int = 3

class RCAResponse(BaseModel):
    # Response model: returns a list of RCA results as strings
    results: List[str]

@app.get("/", tags=["Health"])
def read_root():
    # Health check endpoint to verify the API is running
    return {"status": "RCA LLM API is running"}

@app.post("/query", response_model=RCAResponse, tags=["RCA"])
def query_rca(query_data: RCAQuery):
    # Main RCA endpoint: receives a query and returns root cause analysis results
    results = rca_pipeline(query_data.query, query_data.top_k)
    return {"results": results}
