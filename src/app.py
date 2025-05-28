from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from src.rag_engine import rca_pipeline

app = FastAPI(title="RCA LLM Assistant API")

class RCAQuery(BaseModel):
    query: str
    top_k: int = 3

class RCAResponse(BaseModel):
    results: List[str]

@app.get("/", tags=["Health"])
def read_root():
    return {"status": "RCA LLM API is running"}

@app.post("/query", response_model=RCAResponse, tags=["RCA"])
def query_rca(query_data: RCAQuery):
    results = rca_pipeline(query_data.query, query_data.top_k)
    return {"results": results}
