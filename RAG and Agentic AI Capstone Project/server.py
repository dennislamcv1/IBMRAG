# Libraries to import to create our MCP server and handle data loading
from pathlib import Path
import json

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover
    from fastmcp import FastMCP

# Initializing our MCP server instance
mcp = FastMCP("Connoisseur-Server")

# Data paths
DATA_DIR = Path(__file__).parent
CULINARY_MAP_PATH = DATA_DIR / "California-Culinary-Map.txt"
RESTAURANT_DATA_PATH = DATA_DIR / "structured_restaurant_data.json"
REVIEW_DATA_PATH = DATA_DIR / "augmented_user_review.json"

# Helper functions
def load_restaurant_data() -> list[dict]:
    """Load the structured restaurant data produced in Module 1."""
    with open(RESTAURANT_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_review_data() -> list[dict]:
    """Load the augmented user reviews produced in Module 1."""
    with open(REVIEW_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_present(record: dict, *keys: str, default=None):
    """Return the first non-empty value found for the given keys."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default

# MCP Resource - Exposing the Raw Culinary Map data
@mcp.resource("culinary-map://california")
def get_culinary_map() -> str:
    """The full raw California Culinary Map text from Module 1.
    Contains detailed descriptions of 100+ restaurants across California
    including their vibes, cuisines, ratings, and price ranges."""
    return CULINARY_MAP_PATH.read_text()

# TOOL 1 — Get Restaurant Info (Structured Search)
@mcp.tool()
def get_restaurant_info(restaurant_name: str) -> str:
    """Search for a restaurant by name and return its structured details
    including cuisine, rating, price range, and signature dish."""
    restaurants = load_restaurant_data()
    query = restaurant_name.lower().strip()

    # Finding restuarants that match the query in the structured JSON data
    matches = []
    for restaurant in restaurants:
        name = restaurant["name"].lower()
        if query in name or name in query:
            matches.append(restaurant)

    # Return a not found message if no matches are found
    if not matches:
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No restaurant found matching '{restaurant_name}'.",
                "suggestion": "Try a partial name like 'Iron' or 'Sakura'.",
            },
            indent=2,
        )

    return json.dumps(
        {"status": "found", "count": len(matches), "results": matches},
        indent=2,
    )

# TOOL 2 — Recommend by Vibe (Semantic Search)
@mcp.tool()
def recommend_by_vibe(vibe: str) -> str:
    """Find restaurants that match a given vibe or atmosphere keyword.
    Searches both structured vibe tags and raw text descriptions.
    Examples of vibe keywords: "moody", "sun-drenched", "romantic"""
    restaurants = load_restaurant_data()
    vibe_lower = vibe.lower().strip()

    # Pass 1: Search structured vibe tags in JSON
    structured_matches = []
    for restaurant in restaurants:
        vibe_field = restaurant.get("vibe", "")
        vibes_value = restaurant.get("vibes")
        if isinstance(vibes_value, str):
            vibes_list = [vibes_value.lower()]
        else:
            vibes_list = [v.lower() for v in vibes_value or []]

        description = _first_present(
            restaurant,
            "vibe",
            "description",
            "environment",
            default="",
        ).lower()

        if vibe_lower in vibe_field.lower() or any(vibe_lower in v for v in vibes_list) or vibe_lower in description:
            structured_matches.append(
                {
                    "name": restaurant["name"],
                    "location": restaurant.get("location", restaurant.get("neighborhood", "N/A")),
                    "food_style": restaurant.get("food_style", restaurant.get("cuisine", "N/A")),
                    "rating": restaurant["rating"],
                    "vibe": restaurant.get("vibe", restaurant.get("vibes", [])),
                    "price_range": restaurant["price_range"],
                }
            )

    # Pass 2: Search the raw text for additional matches 
    raw_text = CULINARY_MAP_PATH.read_text()
    paragraphs = raw_text.split("\n\n")
    text_excerpts = []
    for para in paragraphs:
        if vibe_lower in para.lower() and para.strip():
            text_excerpts.append(para.strip()[:300])

    return json.dumps(
        {
            "vibe_searched": vibe,
            "structured_matches": structured_matches,
            "raw_text_excerpts": text_excerpts[:5],
        },
        indent=2,
    )

# TOOL 3 — Get Review (Returns Review Data for Lab 2 Demonstration)
@mcp.tool()
def get_review(restaurant_name: str) -> str:
    """Retrieve the full review for a restaurant."""
    reviews = load_review_data()
    query = restaurant_name.lower().strip()

    # Find the matching review
    matching_review = None
    for review in reviews:
        review_title = _first_present(review, "restaurant_name", "title", default="")
        review_text = _first_present(review, "review_text", "text", default="")
        if query in review_title.lower() or query in review_text.lower():
            matching_review = review
            break
    
    # Return a not found message if no review matches the query
    if not matching_review:
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No review found matching '{restaurant_name}'.",
            },
            indent=2,
        )

    return json.dumps(
        {
            "status": "found",
            "restaurant": _first_present(matching_review, "restaurant_name", "title", default="N/A"),
            "reviewer": _first_present(matching_review, "reviewer", "userId", default="N/A"),
            "rating": matching_review.get("rating", "N/A"),
            "review_text": _first_present(matching_review, "review_text", "text", default="N/A"),
            "image_description": _first_present(matching_review, "image_description", default="N/A"),
            "image_captions": matching_review.get("image_captions", []),
            "visit_date": _first_present(matching_review, "visit_date", "date", default="N/A"),
        },
        indent=2,
    )

# Run the Server
if __name__ == "__main__":
    mcp.run()
