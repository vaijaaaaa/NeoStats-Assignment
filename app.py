"""
Placement Preparation AI Assistant
Main Streamlit app that provides a chatbot interface for placement prep
"""

import logging
import streamlit as st
from config.config import DATA_DIR, RAG_TOP_K, WEB_MAX_RESULTS
from models.llm import create_llm_response
from utils.rag import setup_rag, retrieve_relevant_chunks, build_context_from_chunks
from utils.search import web_search, build_web_context
from utils.prompting import build_system_prompt, build_user_prompt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Placement Prep AI",
    layout="centered"
)

st.title("Placement Preparation AI Assistant")
st.write("Ask me anything about DSA, interviews, companies, or your resume!")


@st.cache_resource
def load_rag_resources():
    """Load and cache RAG resources once."""
    try:
        index, chunk_records = setup_rag(data_dir=DATA_DIR)
        logger.info("RAG setup complete. Chunks loaded: %s", len(chunk_records))
        return index, chunk_records
    except Exception as error:
        logger.exception("RAG setup failed: %s", error)
        return None, []


index, chunk_records = load_rag_resources()

with st.sidebar:
    st.header("Settings")

    response_mode = st.radio(
        "Response Mode:",
        options=["Concise", "Detailed"],
        help="Concise: Quick answers | Detailed: In-depth explanations"
    )

    st.divider()
    st.caption("Mode affects how the AI structures its answers")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask your placement prep question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Thinking..."):
        response = ""
        try:
            retrieved_chunks = retrieve_relevant_chunks(
                query=user_input,
                index=index,
                chunk_records=chunk_records,
                top_k=RAG_TOP_K
            )
            context_text = build_context_from_chunks(retrieved_chunks)
            context_source = "local_rag"

            if context_text:
                context_source = "local_rag"
            else:
                web_results = web_search(user_input, max_results=WEB_MAX_RESULTS)
                web_context = build_web_context(web_results)

                if web_context:
                    context_source = "web_search"
                    context_text = web_context
                else:
                    context_source = "none"
                    context_text = ""

            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(
                user_question=user_input,
                response_mode=response_mode,
                context_source=context_source,
                context_text=context_text,
            )

            response = create_llm_response(
                user_message=user_prompt,
                system_prompt=system_prompt
            )
            logger.info("Response generated. Context source: %s", context_source)
        except Exception as error:
            logger.exception("Failed to process user request: %s", error)
            response = "Something went wrong while processing your request. Please try again."

        if not response or not response.strip():
            response = "I couldn't generate a response right now. Please try rephrasing your question."

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
