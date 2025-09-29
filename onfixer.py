import json, re
from typing import List, Dict, Any

raw = open("allplaces.json","r", encoding="utf-8").read()

# Split concatenated JSON objects with a heuristic: boundaries "}{" -> "}\n{"
normalized = raw.replace("}{", "}\n{").strip()

objs = []
buf = ""
depth = 0
for ch in normalized:
    buf += ch
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            try:
                objs.append(json.loads(buf))
            except Exception:
                # attempt minor fix: remove trailing ",... " artifacts inside strings
                buf2 = re.sub(r",\s*\.\.\.", "", buf)
                # fix the Topchanchi established malformed value to a string
                buf2 = buf2.replace('1915 (lake excavation); sanctuary notified later",',
                                    '1915",')
                try:
                    objs.append(json.loads(buf2))
                except Exception:
                    pass
            buf = ""

def collect_places(objs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    places = []
    for obj in objs:
        for k, v in list(obj.items()):
            if isinstance(v, dict):
                if "tourist_spots" in v and isinstance(v["tourist_spots"], list):
                    places.extend(v["tourist_spots"])
            elif isinstance(v, list) and k.endswith("_tourist_spots"):
                places.extend(v)
            elif k.endswith("_tourist_spots") and isinstance(v, list):
                places.extend(v)
    return places

all_places = collect_places(objs)

def norm_budget(b: str) -> str:
    if not isinstance(b, str): return "Unknown"
    s = b.lower()
    if "low" in s: return "Low"
    if "moderate" in s or "mid" in s: return "Moderate"
    if "high" in s or "premium" in s: return "High"
    return "Unknown"

def ensure_list(x):
    if isinstance(x, list): return x
    if x is None: return []
    if isinstance(x, str):
        parts = [p.strip(" -•|") for p in re.split(r"[;•\n\|]", x) if p.strip()]
        return parts if parts else [x]
    return []

# Fix Topchanchi established parse issue and move note to highlights if needed
for p in all_places:
    if isinstance(p.get("name",""), str) and "Topchanchi" in p["name"]:
        est = p.get("established")
        if isinstance(est, str) and "1915" in est:
            p["established"] = 1915
            hl = ensure_list(p.get("highlights"))
            if "Sanctuary notified later" not in " ".join(hl):
                hl.append("Sanctuary notified later")
            p["highlights"] = hl

# Normalize fields and deduplicate
normed = []
seen = set()
for p in all_places:
    key = (p.get("name","").strip().lower(), p.get("location","").strip().lower())
    if key in seen:
        continue
    seen.add(key)
    q = dict(p)
    q["interest"] = ensure_list(q.get("interest"))
    q["highlights"] = ensure_list(q.get("highlights"))
    q["criticisms"] = ensure_list(q.get("criticisms"))
    q["budget_norm"] = norm_budget(q.get("budget",""))
    normed.append(q)

# Write consolidated places as a single JSON array
with open("jharkhand_places_clean.json","w", encoding="utf-8") as f:
    json.dump(normed, f, ensure_ascii=False, indent=2)

print(f"Total consolidated places: {len(normed)}")
