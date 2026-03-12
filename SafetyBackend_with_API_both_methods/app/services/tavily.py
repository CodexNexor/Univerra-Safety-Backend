import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Iterable, Optional
from dotenv import load_dotenv

# ------------------------------ #
# Fast, low-RAM, connection-pooled
# Tavily + Brave search utilities
# ------------------------------ #

# Load environment variables once
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

TAVILY_ENDPOINT = "https://api.tavily.com/search"
BRAVE_WEB_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
BRAVE_VIDEO_ENDPOINT = "https://api.search.brave.com/res/v1/videos/search"

# --- Tunables ---
HTTP_TIMEOUT = 10  # seconds per request
BRAVE_COUNT = 5    # results per vertical
MAX_WORKERS = 4    # small, avoids RAM spikes but improves latency via parallel I/O

# Global shared session (connection pooling = faster + lower overhead)
_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    """Return a global pooled session (thread-safe for simple GET/POST)."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        # Keep headers tiny and static to lower per-request overhead
        s.headers.update({"Accept": "application/json"})
        _SESSION = s
    return _SESSION


# ------------------------------
# Helpers
# ------------------------------
def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        try:
            # As a fallback, try to parse text if JSON content-type was lying
            return json.loads(resp.text)
        except Exception:
            return {}

def _nonempty_str(s: Any) -> str:
    return s if isinstance(s, str) else ""


# ------------------------------
# Tavily General Search
# ------------------------------
def tavily_general_search(query: str) -> Dict[str, Any]:
    try:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
        }
        r = _session().post(TAVILY_ENDPOINT, json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return _safe_json(r)
    except Exception as e:
        print("Tavily Error (General):", e)
        return {}


# ------------------------------
# Tavily Location Prompt Search (one-by-one as requested)
#  - internally fast but sequential across prompts
# ------------------------------
def _tavily_single_prompt(prompt: str) -> Dict[str, Any]:
    """Fast single Tavily call for a prompt, minimal parsing."""
    try:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": prompt,
            "search_depth": "advanced",
            "include_answer": True,
        }
        r = _session().post(TAVILY_ENDPOINT, json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = _safe_json(r)
        return {
            "prompt": prompt,
            "answer": data.get("answer"),
            "sources": data.get("sources", []),
        }
    except Exception as e:
        print(f"Tavily Error for prompt: {prompt} → {e}")
        return {"prompt": prompt, "answer": None, "sources": []}


def tavily_location_search(prompts: List[str]) -> List[Dict[str, Any]]:
    # Strictly sequential across prompts (per your requirement)
    out: List[Dict[str, Any]] = []
    for p in prompts:
        out.append(_tavily_single_prompt(p))
    return out


# ------------------------------
# Brave Search Helpers (Web, News, Videos)
# ------------------------------
_BRAVE_HEADERS = {
    "Accept": "application/json",
    "X-Subscription-Token": BRAVE_SEARCH_API_KEY or "",
}

def brave_fetch(endpoint: str, query: str) -> Tuple[List[str], List[str]]:
    try:
        params = {"q": query, "count": BRAVE_COUNT}
        r = _session().get(endpoint, headers=_BRAVE_HEADERS, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = _safe_json(r)

        if endpoint.endswith("/web/search"):
            results = data.get("web", {}).get("results", [])
        elif endpoint.endswith("/news/search"):
            results = data.get("news", {}).get("results", [])
        elif endpoint.endswith("/videos/search"):
            results = data.get("videos", {}).get("results", [])
        else:
            results = []

        # Low RAM: list comprehension over dicts; trim to count above
        desc = [_nonempty_str(r.get("description")) for r in results]
        links = [_nonempty_str(r.get("url")) for r in results]
        return desc, links
    except Exception as e:
        print(f"Brave Error ({endpoint}):", e)
        return [], []


def brave_full_search(query: str) -> Dict[str, Any]:
    """
    Fetch web, news, and video in parallel (I/O-bound),
    then merge minimal slices to keep memory small.
    """
    endpoints = (BRAVE_WEB_ENDPOINT, BRAVE_NEWS_ENDPOINT, BRAVE_VIDEO_ENDPOINT)

    # Parallelize the three Brave verticals
    with ThreadPoolExecutor(max_workers=min(3, MAX_WORKERS)) as ex:
        futures = {ex.submit(brave_fetch, ep, query): ep for ep in endpoints}
        results: Dict[str, Tuple[List[str], List[str]]] = {}
        for fut in as_completed(futures):
            ep = futures[fut]
            try:
                results[ep] = fut.result()
            except Exception as e:
                print(f"Brave parallel fetch error for {ep}: {e}")
                results[ep] = ([], [])

    web_desc, web_links = results.get(BRAVE_WEB_ENDPOINT, ([], []))
    news_desc, news_links = results.get(BRAVE_NEWS_ENDPOINT, ([], []))
    video_desc, video_links = results.get(BRAVE_VIDEO_ENDPOINT, ([], []))

    # Keep response compact (top 2 each, joined once)
    combined_answer = " ".join(
        (web_desc[:2] + news_desc[:2] + video_desc[:2])
    ).strip()

    return {
        "brave_answer": combined_answer,
        "brave_sources": {
            "web": web_links,
            "news": news_links,
            "videos": video_links,
        },
    }


# ------------------------------
# Combine Tavily + Brave for general search
#  - Run both providers in parallel for speed
# ------------------------------
def combined_general_search(query: str) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_tavily = ex.submit(tavily_general_search, query)
        f_brave = ex.submit(brave_full_search, query)
        tavily_data = f_tavily.result()
        brave_data = f_brave.result()

    return {
        "query": query,
        "tavily_answer": tavily_data.get("answer"),
        "brave_answer": brave_data.get("brave_answer"),
        "sources": {
            "tavily": tavily_data.get("sources", []),
            "brave": brave_data.get("brave_sources", {}),
        },
    }


# ------------------------------
# Combine Tavily + Brave for location prompts
#  - Process prompts strictly one-by-one (as you asked)
#  - Inside each prompt, call Tavily and Brave in parallel
#    to reduce per-prompt latency without extra RAM bloat
# ------------------------------
def _combined_single_prompt(prompt: str) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_tav = ex.submit(_tavily_single_prompt, prompt)
        f_brv = ex.submit(brave_full_search, prompt)
        tavily_result = f_tav.result()   # {"prompt","answer","sources"}
        brave_result = f_brv.result()    # {"brave_answer","brave_sources"}

    return {
        "prompt": prompt,
        "tavily_answer": tavily_result.get("answer"),
        "brave_answer": brave_result.get("brave_answer"),
        "sources": {
            "tavily": tavily_result.get("sources", []),
            "brave": brave_result.get("brave_sources", {}),
        },
    }


def combined_location_search(prompts: List[str]) -> Dict[str, Any]:
    combined: List[Dict[str, Any]] = []
    # STRICT SEQUENCE: process prompt 1, then 2, ... until done
    for p in prompts:
        combined.append(_combined_single_prompt(p))
    return {"results": combined}


# ------------------------------
# Unified Entry Point for Query Processing
# ------------------------------
def search_with_tavily(query_or_prompts, is_location_query: bool):
    """
    Public method for use in query.py or other services.
    Routes request to correct method.
    - Location flow: prompts processed one-by-one; each prompt runs Tavily+Brave in parallel.
    - General flow: Tavily+Brave in parallel for a single query.
    """
    if is_location_query:
        if isinstance(query_or_prompts, list):
            return combined_location_search(query_or_prompts)
        else:
            return {"error": "Expected a list of prompts for location queries."}
    else:
        return combined_general_search(query_or_prompts)


# ------------------------------
# ✅ Example test (manual)
# ------------------------------
if __name__ == "__main__":
    # Test Location Search (sequential prompts, fast per-prompt)
    prompts = [
        "Recent crime in Kamla Nagar",
        "Traffic alerts in Connaught Place",
    ]
    location_test = search_with_tavily(prompts, is_location_query=True)
    print("\n🔎 Location Result:\n", location_test)

    # Test General Search (parallel providers)
    general_test = search_with_tavily("How is rainfall in Rajasthan this week?", is_location_query=False)
    print("\n💡 General Result:\n", general_test)
