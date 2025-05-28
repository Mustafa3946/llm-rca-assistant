# src/rag_engine.py

from src.retriever import get_top_k_logs
from src.mock_bedrock import query_llm

def generate_prompt(query: str, retrieved_logs: list) -> str:
    logs_text = "\n\n".join([f"- {log}" for _, log in retrieved_logs])
    return (
        f"You are an RCA assistant. Given the following system logs and the user query, "
        f"identify the most likely root cause in a concise and technical explanation.\n\n"
        f"User Query:\n{query}\n\nRelevant Logs:\n{logs_text}"
    )

def rca_pipeline(query: str, k: int = 3) -> str:
    retrieved_logs = get_top_k_logs(query, k=k)
    prompt = generate_prompt(query, retrieved_logs)
    response = query_llm(prompt)
    return response
