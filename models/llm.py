"""
LLM Module: Handles communication with the Language Model
This module abstracts away the API details from the rest of the app
"""

import logging
import os
import httpx
from groq import Groq
from dotenv import load_dotenv
from config.config import GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_groq_client() -> Groq:
    """Create Groq client lazily to avoid import-time crashes."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please create a .env file with your API key.")

    http_client = httpx.Client(timeout=LLM_TIMEOUT_SECONDS)
    return Groq(api_key=api_key, http_client=http_client)


def create_llm_response(user_message: str, system_prompt: str = None, conversation_history: list = None) -> str:
    """
    Send a message to the LLM and get a response.
    
    Args:
        user_message (str): The user's question or input
        system_prompt (str): The system prompt that defines the LLM's behavior/role
        conversation_history (list): Previous messages (will implement later for multi-turn conversations)
    
    Returns:
        str: The LLM's response
    """

    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant for placement preparation."

    if conversation_history is None:
        conversation_history = []

    if not user_message or not user_message.strip():
        return "Please enter a valid question."
    
    try:
        client = _get_groq_client()

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

        llm_response = response.choices[0].message.content
        if not llm_response:
            logger.warning("LLM returned empty content.")
            return "I couldn't generate a response. Please try rephrasing your question."

        return llm_response
    
    except Exception as e:
        logger.exception("Error while calling Groq API: %s", e)
        return f"Error calling LLM: {str(e)}"
