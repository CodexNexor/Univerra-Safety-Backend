# app/services/location_search.py

import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# ------------------------------
# Load environment variables once
# ------------------------------
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

# API Endpoints
TAVILY_ENDPOINT = "https://api.tavily.com/search"
BRAVE_WEB_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
BRAVE_VIDEO_ENDPOINT = "https://api.search.brave.com/res/v1/videos/search"

# ------------------------------
# Tunables
# ------------------------------
HTTP_TIMEOUT = 8   # seconds per request
MAX_WORKERS = 2    # low to reduce spikes
BRAVE_COUNT = 3    # fewer results = faster + less RAM

# Global pooled session (connection reuse)
_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    """Return a global pooled session (thread-safe)."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
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
            return json.loads(resp.text)
        except Exception:
            return {}


def _nonempty_str(s: Any) -> str:
    return s if isinstance(s, str) else ""


# ------------------------------
# Tavily Search
# ------------------------------
def _tavily_location_search(prompt: str) -> Dict[str, Any]:
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
            "tavily_answer": data.get("answer"),
            "tavily_sources": data.get("sources", []),
        }
    except Exception as e:
        print(f"Tavily Error for prompt: {prompt} → {e}")
        return {"prompt": prompt, "tavily_answer": None, "tavily_sources": []}


# ------------------------------
# Brave Search (Web + News + Videos)
# ------------------------------
_BRAVE_HEADERS = {
    "Accept": "application/json",
    "X-Subscription-Token": BRAVE_SEARCH_API_KEY or "",
}

def _brave_location_search(prompt: str) -> Dict[str, Any]:
    endpoints = [BRAVE_WEB_ENDPOINT, BRAVE_NEWS_ENDPOINT, BRAVE_VIDEO_ENDPOINT]

    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                _session().get, ep,
                headers=_BRAVE_HEADERS,
                params={"q": prompt, "count": BRAVE_COUNT},
                timeout=HTTP_TIMEOUT
            ): ep for ep in endpoints
        }
        for fut, ep in futures.items():
            try:
                resp = fut.result()
                resp.raise_for_status()
                data = _safe_json(resp)
                if ep.endswith("/web/search"):
                    items = data.get("web", {}).get("results", [])
                elif ep.endswith("/news/search"):
                    items = data.get("news", {}).get("results", [])
                elif ep.endswith("/videos/search"):
                    items = data.get("videos", {}).get("results", [])
                else:
                    items = []

                results[ep] = {
                    "desc": [_nonempty_str(i.get("description")) for i in items],
                    "links": [_nonempty_str(i.get("url")) for i in items],
                }
            except Exception as e:
                print(f"Brave Error ({ep}):", e)
                results[ep] = {"desc": [], "links": []}

    combined_answer = " ".join(
        results.get(BRAVE_WEB_ENDPOINT, {}).get("desc", [])[:1] +
        results.get(BRAVE_NEWS_ENDPOINT, {}).get("desc", [])[:1] +
        results.get(BRAVE_VIDEO_ENDPOINT, {}).get("desc", [])[:1]
    ).strip()

    return {
        "brave_answer": combined_answer,
        "brave_sources": {
            "web": results.get(BRAVE_WEB_ENDPOINT, {}).get("links", []),
            "news": results.get(BRAVE_NEWS_ENDPOINT, {}).get("links", []),
            "videos": results.get(BRAVE_VIDEO_ENDPOINT, {}).get("links", []),
        },
    }


# ------------------------------
# Unified Search (Single Prompt)
# ------------------------------
def search_location(prompt: str) -> Dict[str, Any]:
    """Run Tavily + Brave search for a single location prompt."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_tav = ex.submit(_tavily_location_search, prompt)
        f_brv = ex.submit(_brave_location_search, prompt)
        tav = f_tav.result()
        brv = f_brv.result()

    return {
        "prompt": prompt,
        "tavily_answer": tav.get("tavily_answer"),
        "brave_answer": brv.get("brave_answer"),
        "sources": {
            "tavily": tav.get("tavily_sources", []),
            "brave": brv.get("brave_sources", {}),
        },
    }


# ------------------------------
# Combined Location Search (Multi Prompt)
# ------------------------------
def combined_location_search(prompts: List[str]) -> Dict[str, Any]:
    results = []
    for p in prompts:
        results.append(search_location(p))
    return {"results": results}


# ------------------------------
# ✅ Manual Test
# ------------------------------
if __name__ == "__main__":
    test_prompts = [
        "Crime reports near Connaught Place, Delhi",
        "Traffic updates for Bandra, Mumbai",
    ]
    out = combined_location_search(test_prompts)
    print("\n🔎 Location Search Result:\n", json.dumps(out, indent=2))
