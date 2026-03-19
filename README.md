# Placement Preparation AI Assistant

A Streamlit chatbot for placement prep with:
- local RAG over your notes (`data/*.txt`)
- web-search fallback when local context is weak
- concise/detailed response modes

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your Groq key in `.env`:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py` - Streamlit UI + orchestration
- `models/llm.py` - Groq LLM client
- `models/embeddings.py` - embedding model wrapper
- `utils/rag.py` - document loading, chunking, FAISS retrieval
- `utils/search.py` - web search fallback
- `utils/prompting.py` - final prompt construction
- `data/` - local knowledge files (`dsa.txt`, `companies.txt`, `resume.txt`)

## Notes

- Keep `.env` private (already ignored by `.gitignore`).
- Update files in `data/` to personalize responses.
