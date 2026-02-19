import re
import textwrap
import requests
from bs4 import BeautifulSoup
from utils.core.log import get_logger
from utils.vault import secrets

"""
pip install requests bs4
Config from Vault
"""

WHITESPACE_RE = re.compile(r"\s+")

GOOGLE_API_KEY = (secrets.get("GOOGLE_API_KEY", default="") or "")
GOOGLE_CX = (secrets.get("GOOGLE_CX", default="") or "")


def _search_configured() -> bool:
    return bool(GOOGLE_API_KEY and GOOGLE_CX)


def _clean_text(text: str, max_chars: int = 800) -> str:
    text = WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    if len(text) > max_chars:
        text = textwrap.shorten(text, width=max_chars, placeholder=" ...")
    return text


def google_search(query: str) -> str:
    logger = get_logger()
    if not _search_configured():
        msg = "Google search not configured (set GOOGLE_API_KEY and GOOGLE_CX in Vault or env)."
        print(msg)
        logger.debug(msg)
        return msg
    logger.debug(f"Performing Google search with query: {query}")

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
    except requests.exceptions.RequestException as e:
        logger.debug(f"Error during Google search: {e}")
        return f"Error: {e}"

    if "error" in search_results:
        error_message = f"Error: {search_results['error']['message']}"
        logger.debug(error_message)
        return error_message

    items = search_results.get("items", [])
    if not items:
        logger.debug("No results found.")
        return "No results found."

    aggregated_results = []
    max_sources = 5

    for index, item in enumerate(items[:max_sources], start=1):
        link = item.get("link")
        title = item.get("title", "Untitled")
        if not link:
            continue

        logger.debug(f"Fetching content from ({index}): {link}")
        snippet_text = ""
        try:
            page_response = requests.get(link, timeout=10)
            page_response.raise_for_status()
            soup = BeautifulSoup(page_response.text, "html.parser")

            # Get reasonably clean text
            page_text = soup.get_text(separator=" ", strip=True)
            snippet_text = _clean_text(page_text, max_chars=2000)

            logger.debug(
                f"Fetched content from {link} (first 200 chars): {snippet_text[:200]}..."
            )
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout error fetching {link}. Skipping this source...")
            continue
        except requests.exceptions.RequestException as e:
            logger.debug(f"Error fetching {link}: {e}. Skipping this source...")
            continue

        aggregated_results.append(
            {
                "source_index": index,
                "title": title,
                "url": link,
                "snippet": snippet_text,
            }
        )

    if not aggregated_results:
        return "Failed to retrieve content from available results."

    return "\n".join(
        f"Source {r['source_index']}:\n"
        f"Title: {r['title']}\n"
        f"URL: {r['url']}\n"
        f"Snippet: {r['snippet']}\n"
        for r in aggregated_results
    )


def main():
    from utils.core.log import pid_tool_logger, set_logger

    package_id = "system_check"
    logger = pid_tool_logger(package_id, "web_search")
    set_logger(logger)

    test_query = "OpenAI ChatGPT latest updates"
    print("WEB_SEARCH TEST START")

    try:
        results = google_search(test_query)
        print("WEB_SEARCH OK")
        return True
    except Exception:
        print("WEB_SEARCH ERROR")
        raise


if __name__ == "__main__":
    main()
