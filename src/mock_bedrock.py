# =============================================================================
# mock_bedrock.py
#
# Purpose:
#   This module simulates responses from a large language model (LLM) such as Bedrock Claude or Titan.
#   It is used for local development and testing of the RCA assistant pipeline without requiring access
#   to an actual LLM API. The mock function returns a fixed response and echoes part of the prompt.
# =============================================================================

def query_llm(prompt: str) -> str:
    # Simulates a response from an LLM (e.g., Bedrock Claude or Titan) for testing purposes.
    # Returns a fixed mock response and includes the first 300 characters of the prompt for reference.
    return f"[Mocked LLM Response]\nBased on the logs, the most likely root cause is: network instability or firewall blocking outbound connections.\n\nPrompt Received:\n{prompt[:300]}..."
