"""
Web Search Utilities: Fetch fresh web results when local RAG context is insufficient.
"""

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 3) -> list[dict]:
	"""
	Search the web for latest/public information.

	Args:
		query (str): User query to search.
		max_results (int): Number of search results to fetch.

	Returns:
		list[dict]: Each result has {"title", "url", "snippet"}
	"""
	if not query:
		return []

	try:
		results = []
		with DDGS() as ddgs:
			search_results = ddgs.text(query, max_results=max_results)
			for item in search_results:
				results.append(
					{
						"title": item.get("title", ""),
						"url": item.get("href", ""),
						"snippet": item.get("body", ""),
					}
				)
		return results
	except Exception:
		return []


def build_web_context(results: list[dict]) -> str:
	"""
	Build prompt context text from web results.
	"""
	if not results:
		return ""

	context_parts = []
	for result in results:
		title = result.get("title", "")
		url = result.get("url", "")
		snippet = result.get("snippet", "")
		context_parts.append(f"[Web: {title}]\nURL: {url}\n{snippet}")

	return "\n\n".join(context_parts)
