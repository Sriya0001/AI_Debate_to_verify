import os
from tavily import TavilyClient


def search(query: str, max_results: int = 3) -> list[dict]:
    """
    Run a web search via Tavily and return a list of results.

    Each result dict contains:
        - url:     source URL
        - content: extracted page snippet
        - score:   Tavily relevance score (0-1)

    Raises on API errors so the calling agent can catch and handle.
    """
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",       # "advanced" is slower/more expensive
        include_answer=False,       # we want raw results, not a Tavily summary
    )

    return response.get("results", [])


def format_results_for_prompt(results: list[dict]) -> str:
    """
    Format Tavily results into a readable block for injection into a prompt.
    Keeps it tight — agents don't need full page content, just enough to argue.
    """
    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['url']}")
        lines.append(r["content"].strip())
        lines.append("")

    return "\n".join(lines)