# src/mock_bedrock.py

def query_llm(prompt: str) -> str:
    # Simulate a response from an LLM (e.g., Bedrock Claude or Titan)
    return f"[Mocked LLM Response]\nBased on the logs, the most likely root cause is: network instability or firewall blocking outbound connections.\n\nPrompt Received:\n{prompt[:300]}..."
