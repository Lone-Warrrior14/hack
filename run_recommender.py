import json
from recommender import recommend_places

with open("jharkhand_places_clean.json", "r", encoding="utf-8") as f:
    all_places = json.load(f)

# Example profile: edit these to test
interests = ["Waterfall", "Photography", "Bird watching"]
budget = "Low"
days = 2
group = "Families, friends"
month = "Nov"  # or "winter", "Jan", etc.

recs = recommend_places(
    all_places=all_places,
    interests=interests,
    budget=budget,
    days=days,
    group_size_desc=group,
    month_hint=month,
    top_n=10
)

print(json.dumps({"recommendations": recs}, ensure_ascii=False, indent=2))
