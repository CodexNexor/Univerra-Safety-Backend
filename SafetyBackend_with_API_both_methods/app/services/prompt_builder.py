import random
from typing import List, Sequence

# Templates to build various prompts (tuple is a touch lighter than list)
TEMPLATES: Sequence[str] = (
    "{locations} recent safety incidents alerts",
    "{locations} latest crime news updates",
    "{locations} current safety rating crime risk",
    "{locations} police alerts public safety advisories today",
    "{locations} live crime updates danger reports",
)

def build_prompts(main_location: str, nearby_locations: List[str], max_prompts: int = 5) -> List[str]:
    # Total pool size (main + nearby); we only ever need up to 4
    n_total = 1 + (len(nearby_locations) if nearby_locations else 0)
    k = 4 if n_total >= 4 else n_total

    # Sample indices to avoid copying all strings into a combined list
    idxs = random.sample(range(n_total), k)

    # Map sampled indices to strings without building an intermediate combined list
    def _loc(i: int) -> str:
        return main_location if i == 0 else nearby_locations[i - 1]

    joined_locations = " ".join(_loc(i) for i in idxs)

    # Clamp prompts to available templates
    limit = max_prompts if max_prompts < len(TEMPLATES) else len(TEMPLATES)

    # Render prompts with a single pass
    return [tpl.format(locations=joined_locations) for tpl in TEMPLATES[:limit]]

# ✅ Example test
if __name__ == "__main__":
    main_loc = "Anand Vihar"
    nearby_locs = ["Laxmi Nagar", "Preet Vihar", "Karkardooma"]
    output = build_prompts(main_loc, nearby_locs)
    print("Generated Prompts:")
    for p in output:
        print("-", p)
