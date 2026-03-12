# app/services/location.py

import os
import logging
import requests
import google.generativeai as genai
from app.services import location_search

# -------------------------
# Config
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

logger = logging.getLogger(__name__)

# -------------------------
# Extract location from query
# -------------------------
def extract_location_with_gemini(query: str) -> str | None:
    """Use Gemini to detect a location from the query."""
    try:
        prompt = (
            f"You are Univerra AI, an advanced hyper-location engine. "
            f"From this query, extract the MAIN location (city, town, state, region, landmark, or village). "
            f"Return ONLY the location name, nothing else.\n\nQuery: {query}"
        )
        resp = gemini_model.generate_content(prompt)
        if resp and resp.text:
            location = resp.text.strip()
            return location if location else None
    except Exception as e:
        logger.error(f"Location extraction error: {e}")
    return None

# -------------------------
# Mapbox Nearby Finder
# -------------------------
def find_nearby_locations(location: str, limit: int = 3) -> list[str]:
    """Use Mapbox Geocoding API to get nearby places."""
    try:
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{location}.json"
        params = {"access_token": MAPBOX_TOKEN, "limit": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if not data.get("features"):
            return [location]

        coords = data["features"][0]["geometry"]["coordinates"]
        lon, lat = coords

        nearby_url = "https://api.mapbox.com/geocoding/v5/mapbox.places/poi.json"
        params = {"access_token": MAPBOX_TOKEN, "proximity": f"{lon},{lat}", "limit": limit}
        nr = requests.get(nearby_url, params=params, timeout=10)
        nr.raise_for_status()
        nearby_data = nr.json()

        nearby_places = [f["text"] for f in nearby_data.get("features", [])]
        return [location] + nearby_places
    except Exception as e:
        logger.error(f"Mapbox nearby error: {e}")
        return [location]

# -------------------------
# Prompt Generator
# -------------------------
def create_prompts(user_query: str, locations: list[str]) -> list[str]:
    """Generate 3 prompts tailored to the user query and location(s)."""
    try:
        prompt = (
            f"You are Univerra AI. The user asked: '{user_query}'. "
            f"Locations detected: {', '.join(locations)}. "
            f"Create 3 short search queries that match the user's intent. "
            f"Make them precise and relevant. Return them as bullet points."
        )
        resp = gemini_model.generate_content(prompt)
        if not resp or not resp.text:
            return [f"General information about {', '.join(locations)}"]
        prompts = [line.strip("-• ").strip() for line in resp.text.splitlines() if line.strip()]
        return prompts[:3]
    except Exception as e:
        logger.error(f"Prompt creation error: {e}")
        return [f"General information about {', '.join(locations)}"]

# -------------------------
# Humanize Final Answer
# -------------------------
def humanize_answer(user_query: str, raw_results: list[dict]) -> str:
    """Send raw search results back to Gemini to generate a clean human response in Markdown."""
    try:
        context = "\n".join([f"Prompt: {r['prompt']}\nResult: {r['result']}" for r in raw_results])
        final_prompt = (
            f"You are Univerra AI, a human-like assistant. "
            f"The user asked: {user_query}. "
            f"Here are the raw search results:\n{context}\n\n"
            f"Now, write a clear, helpful, natural response in Markdown. "
            f"Do NOT mention Gemini, Mapbox, AI, or system details. "
            f"Use tables for structured info if applicable."
        )
        resp = gemini_model.generate_content(final_prompt)
        if resp and resp.text:
            return resp.text.strip()
    except Exception as e:
        logger.error(f"Humanization error: {e}")
    return "Sorry, I couldn’t find enough details right now."

# -------------------------
# Main Processor
# -------------------------
def process_location_query(user_query: str, history=None) -> dict:
    """Handles the full location query pipeline."""
    location = extract_location_with_gemini(user_query)
    if not location:
        return {"answer": "Sorry, I couldn’t detect any location in your query."}

    all_locations = find_nearby_locations(location)
    prompts = create_prompts(user_query, all_locations)

    raw_results = []
    for p in prompts:
        try:
            result = location_search.search_location(p)
            if not isinstance(result, str):
                # Convert dict/list results to Markdown table if needed
                if isinstance(result, list):
                    md = "| Result |\n|--------|\n" + "\n".join([f"| {r} |" for r in result])
                    result = md
                else:
                    result = str(result)
        except Exception as e:
            result = f"Error: {e}"
        raw_results.append({"prompt": p, "result": result})

    final_answer = humanize_answer(user_query, raw_results)

    return {
        "main_location": location,
        "all_locations": all_locations,
        "prompts": prompts,
        "raw_results": raw_results,
        "answer": final_answer,
    }

# -------------------------
# API Entry Point
# -------------------------
def handle_location_query(user_query: str, history=None) -> dict:
    """Public entrypoint used in api.py for frontend requests."""
    result = process_location_query(user_query, history)
    return {
        "answer": result.get("answer", "Sorry, no answer."),
        "main_location": result.get("main_location", ""),
        "all_locations": result.get("all_locations", []),
        "prompts": result.get("prompts", []),
        "raw_results": result.get("raw_results", [])
    }
