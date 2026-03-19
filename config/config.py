"""
Central configuration for the Placement Preparation AI Assistant.
"""

DATA_DIR = "data"
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 100
RAG_TOP_K = 3

WEB_MAX_RESULTS = 3

GROQ_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT_SECONDS = 30.0
