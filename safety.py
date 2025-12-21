"""
Safety guardrails for responsible usage of the RAG system.
Implements rule-based content filtering to prevent unsafe queries.
"""

BLOCKED_TERMS = [
    "illegal",
    "violence",
    "hate",
    "self-harm",
    "explicit"
]

def validate_query(query: str):
    if not query or not query.strip():
        raise ValueError("Empty queries are not allowed.")

    query_lower = query.lower()
    for term in BLOCKED_TERMS:
        if term in query_lower:
            raise ValueError("Query violates safety and ethical guidelines.")
