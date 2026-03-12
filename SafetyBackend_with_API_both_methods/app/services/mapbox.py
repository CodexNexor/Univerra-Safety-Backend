# app/services/mapbox.py

import os
import requests
from typing import Tuple, List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")

# ------------------------------------------------------------
# Global Session (connection pooling → faster, lower RAM)
# ------------------------------------------------------------
_SESSION: Optional[requests.Session] = None

def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({"Accept": "application/json"})
        _SESSION = s
    return _SESSION


# ------------------------------------------------------------
# STEP 1: Get coordinates of a location using Mapbox
# ------------------------------------------------------------
def get_coordinates(location_name: str, country_code: str = "IN") -> Tuple[float, float]:
    """Get (lng, lat) for a location name via Mapbox."""
    try:
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{location_name}.json"
        params = {
            "access_token": MAPBOX_API_KEY,
            "limit": 1,
            "country": country_code
        }
        resp = _session().get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("features"):
            return tuple(data["features"][0]["center"])  # (lng, lat)
        return ()
    except Exception as e:
        print(f"[Mapbox] Error in get_coordinates: {e}")
        return ()


# ------------------------------------------------------------
# STEP 2: Get nearby villages/localities using Overpass API
# ------------------------------------------------------------
def get_nearby_locations(location_name: str, radius_m: int = 10000, max_nearby: int = 5) -> List[str]:
    """Find nearby villages, suburbs, hamlets, etc. around a location."""
    coords = get_coordinates(location_name)
    if not coords:
        return [location_name.strip()]

    lng, lat = coords
    try:
        query = f"""
        [out:json][timeout:25];
        (
          node(around:{radius_m},{lat},{lng})["place"~"village|hamlet|suburb|locality|town"];
          way(around:{radius_m},{lat},{lng})["place"~"village|hamlet|suburb|locality|town"];
          relation(around:{radius_m},{lat},{lng})["place"~"village|hamlet|suburb|locality|town"];
        );
        out center;
        """
        resp = _session().get("https://overpass-api.de/api/interpreter", params={"data": query}, timeout=30)
        data = resp.json().get("elements", [])

        names: List[str] = []
        for item in data:
            tags = item.get("tags", {})
            name = tags.get("name")
            if name:
                names.append(name.strip())

        # Deduplicate + remove self if similar
        unique_names = list(dict.fromkeys([n for n in names if n.lower() not in location_name.lower()]))

        # Return main + nearby
        return [location_name.strip()] + unique_names[:max_nearby]

    except Exception as e:
        print(f"[Overpass] Error in get_nearby_locations: {e}")
        return [location_name.strip()]


# ------------------------------------------------------------
# STEP 3: Get traffic congestion level using Mapbox Traffic API
# ------------------------------------------------------------
def get_traffic_conditions(location_name: str, radius_m: int = 1000) -> Dict[str, Any]:
    """Fetch traffic congestion info near a location using Mapbox Traffic API."""
    coords = get_coordinates(location_name)
    if not coords:
        return {"status": "error", "message": "Traffic data not available.", "summary": ""}

    lng, lat = coords
    tile_url = f"https://api.mapbox.com/v4/mapbox.mapbox-traffic-v1/tilequery/{lng},{lat}.json"

    try:
        resp = _session().get(tile_url, params={
            "access_token": MAPBOX_API_KEY,
            "radius": radius_m,
            "limit": 20
        }, timeout=15)

        data = resp.json()
        features = data.get("features", [])

        if not features:
            return {"status": "ok", "message": "No traffic data found nearby.", "summary": ""}

        congestion_levels = [f.get("properties", {}).get("congestion") for f in features if "properties" in f]
        congestion_levels = [c for c in congestion_levels if c]

        if not congestion_levels:
            return {"status": "ok", "message": "No congestion info available.", "summary": ""}

        summary = {
            "low": congestion_levels.count("low"),
            "moderate": congestion_levels.count("moderate"),
            "heavy": congestion_levels.count("heavy"),
            "severe": congestion_levels.count("severe")
        }

        # Human-readable summary
        parts = [f"{v} area(s) with {k} traffic" for k, v in summary.items() if v > 0]
        summary_text = ", ".join(parts) if parts else "No traffic congestion detected."

        return {
            "status": "ok",
            "message": "Traffic data retrieved successfully.",
            "summary": summary_text,
            "details": summary
        }

    except Exception as e:
        print(f"[Mapbox] Error in get_traffic_conditions: {e}")
        return {"status": "error", "message": "Failed to fetch traffic data.", "summary": ""}


# ------------------------------------------------------------
# ✅ Example test (manual run)
# ------------------------------------------------------------
if __name__ == "__main__":
    location = "Singhana jhunjhunu"

    nearby = get_nearby_locations(location)
    print("Nearby Locations (incl. main):", nearby)

    traffic = get_traffic_conditions(location)
    print("Traffic:", traffic)
