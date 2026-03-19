"""
Prompting Utilities: Build structured prompts for consistent LLM behavior.
"""


def build_system_prompt() -> str:
    """
    Define the assistant role and global behavior rules.
    """
    return (
        "You are a Placement Preparation AI Assistant. "
        "Your job is to help students with DSA, interview preparation, resume guidance, and company-specific advice. "
        "Be clear, practical, and honest. If context is missing or uncertain, say so explicitly."
    )


def build_user_prompt(
    user_question: str,
    response_mode: str,
    context_source: str,
    context_text: str,
) -> str:
    """
    Build a structured user prompt with role inputs: mode, context, and question.

    Args:
        user_question (str): Original user query.
        response_mode (str): "Concise" or "Detailed".
        context_source (str): "local_rag", "web_search", or "none".
        context_text (str): Retrieved context text.

    Returns:
        str: Structured prompt payload for the model.
    """
    mode_instructions = (
        "Respond in 2-3 sentences. Be direct and avoid long explanations."
        if response_mode == "Concise"
        else "Provide a detailed explanation with examples, practical tips, and clear structure."
    )

    if not context_text:
        context_text = "No external context available. Use general placement knowledge."

    return (
        "TASK INPUTS:\n"
        f"- Response Mode: {response_mode}\n"
        f"- Context Source: {context_source}\n\n"
        "CONTEXT:\n"
        f"{context_text}\n\n"
        "USER QUESTION:\n"
        f"{user_question}\n\n"
        "INSTRUCTIONS:\n"
        f"{mode_instructions}\n"
        "If context is provided and relevant, prioritize it. "
        "If context is insufficient, answer with best effort and mention uncertainty briefly."
    )
