# recommender.py
import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_budget(budget_str: str) -> str:
    if not isinstance(budget_str, str):
        return "Unknown"
    s = budget_str.lower()
    if "low" in s:
        return "Low"
    if "moderate" in s or "mid" in s:
        return "Moderate"
    if "high" in s or "premium" in s:
        return "High"
    return "Unknown"


def parse_group_fit(ideal_group_size: str) -> List[str]:
    if not isinstance(ideal_group_size, str):
        return []
    s = ideal_group_size.lower()
    tokens = []
    for tag in [
        "solo", "couple", "couples", "friends", "small", "large", "family",
        "families", "seniors", "children", "kids", "groups", "trekkers",
        "photographers", "devotees", "students"
    ]:
        if tag in s:
            tokens.append(tag)
    return list(set(tokens))


def month_from_season_str(best_time: str) -> List[str]:
    if not isinstance(best_time, str):
        return []
    s = best_time.lower()
    season_map = {
        "october": "oct", "november": "nov", "december": "dec", "january": "jan",
        "february": "feb", "march": "mar", "april": "apr", "may": "may",
        "june": "jun", "july": "jul", "august": "aug", "september": "sep",
        "winter": "winter", "monsoon": "monsoon", "summer": "summer", "shravan": "shravan"
    }
    found = []
    for k, v in season_map.items():
        if k in s:
            found.append(v)
    return list(set(found))


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return " ".join([str(t) for t in x])
    return str(x)


def build_corpus(row: pd.Series) -> str:
    tokens = []
    for col in ["type", "best_time_to_visit"]:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            tokens.append(val)
        elif val is not None:
            tokens.append(_to_text(val))
    for col in ["interest", "highlights"]:
        vals = row.get(col, [])
        if isinstance(vals, list):
            tokens.extend([_to_text(v) for v in vals])
        elif isinstance(vals, str) and vals.strip():
            tokens.append(vals)
    tokens.append("budget_" + normalize_budget(row.get("budget", "")))
    group_tags = parse_group_fit(row.get("ideal_group_size", ""))
    tokens.extend(["group_" + g for g in group_tags])
    text = " ".join(tokens)
    text = re.sub(r"[^a-zA-Z0-9\s/_-]", " ", text)
    return text.lower().strip()


def prepare_dataframe(all_places: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(all_places)
    if "budget" not in df.columns:
        df["budget"] = "Unknown"
    if "ideal_group_size" not in df.columns:
        df["ideal_group_size"] = ""
    if "best_time_to_visit" not in df.columns:
        df["best_time_to_visit"] = ""
    if "interest" not in df.columns:
        df["interest"] = [[] for _ in range(len(df))]
    if "highlights" not in df.columns:
        df["highlights"] = [[] for _ in range(len(df))]

    df["budget_norm"] = df["budget"].apply(normalize_budget)
    df["group_tags"] = df["ideal_group_size"].apply(parse_group_fit)
    df["season_tags"] = df["best_time_to_visit"].apply(month_from_season_str)
    df["corpus"] = df.apply(build_corpus, axis=1)
    return df


def build_vectorizer(df: pd.DataFrame) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(df["corpus"].tolist())
    return vec, X


def make_user_query(interests: List[str], budget: str, group_desc: str, month_hint: str) -> str:
    toks = []
    toks.extend(interests or [])
    if budget:
        toks.append("budget_" + normalize_budget(budget))
    if group_desc:
        gtags = parse_group_fit(group_desc)
        toks.extend(["group_" + g for g in gtags])
    if month_hint:
        toks.append(month_hint.lower())
    return " ".join(toks)


def score_rules(row: pd.Series, user_budget: str, user_group: str, user_month_tag: str, days: int) -> float:
    score = 0.0
    if user_budget:
        if normalize_budget(user_budget) == row.get("budget_norm"):
            score += 0.3
        elif row.get("budget_norm") == "Unknown":
            score += 0.05
        else:
            score -= 0.1
    if user_group:
        ugtags = set(parse_group_fit(user_group))
        rgtags = set(row.get("group_tags", []))
        if ugtags and rgtags and (ugtags & rgtags):
            score += 0.2
    if user_month_tag:
        if user_month_tag in row.get("season_tags", []):
            score += 0.2
        elif any(seas in ["winter", "monsoon", "summer"] for seas in row.get("season_tags", [])) and user_month_tag in ["oct", "nov", "dec", "jan", "feb", "mar"]:
            score += 0.05
    place_type = _to_text(row.get("type", "")).lower()
    if days is not None:
        if days <= 1 and any(k in place_type for k in ["park", "temple", "religious", "city", "museum"]):
            score += 0.15
        if days >= 2 and any(k in place_type for k in ["waterfall", "dam", "lake", "hill", "wildlife", "nature"]):
            score += 0.15
    try:
        r = float(row.get("rating", 0))
    except Exception:
        r = 0.0
    score += (r - 3.5) * 0.05
    return score


def recommend_places(
    all_places: List[Dict[str, Any]],
    interests: List[str],
    budget: str,
    days: int,
    group_size_desc: str,
    month_hint: str,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    df = prepare_dataframe(all_places)
    vec, X = build_vectorizer(df)
    user_query = make_user_query(interests, budget, group_size_desc, month_hint)
    qv = vec.transform([user_query])
    sim = cosine_similarity(qv, X).ravel()

    hybrid = []
    month_tag = month_hint.lower() if isinstance(month_hint, str) and month_hint else ""
    for i, row in df.iterrows():
        rules = score_rules(row, budget, group_size_desc, month_tag, days)
        hybrid.append(0.7 * sim[i] + 0.3 * rules)
    df["score"] = hybrid
    df = df.sort_values("score", ascending=False).head(top_n)

    out = []
    keys = [
        "name", "location", "rating", "type", "entry_fee", "timings",
        "best_time_to_visit", "highlights", "criticisms", "interest",
        "budget", "ideal_group_size"
    ]
    for _, r in df.iterrows():
        item = {}
        for k in keys:
            val = r.get(k, None)
            if isinstance(val, (list, tuple)):
                item[k] = list(val)
            else:
                # Safe NA check for scalars
                try:
                    item[k] = (None if pd.isna(val) else val)
                except Exception:
                    item[k] = val
        # Ensure lists
        for lk in ["highlights", "criticisms", "interest"]:
            v = item.get(lk)
            if isinstance(v, list):
                pass
            elif v is None:
                item[lk] = []
            elif isinstance(v, str):
                parts = [p.strip() for p in re.split(r"[;•|-]", v) if p.strip()]
                item[lk] = parts if parts else [v]
            else:
                item[lk] = []
        out.append(item)
    return out


# Optional CLI usage
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to jharkhand_places_clean.json")
    ap.add_argument("--interests", nargs="*", default=[], help="Interests like Waterfall Photography Bird watching")
    ap.add_argument("--budget", default="", help="Low/Moderate/High")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--group", default="", help="e.g., Families, friends")
    ap.add_argument("--month", default="", help="e.g., Nov, winter")
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    data = json.load(open(args.data, "r", encoding="utf-8"))
    recs = recommend_places(
        all_places=data,
        interests=args.interests,
        budget=args.budget,
        days=args.days,
        group_size_desc=args.group,
        month_hint=args.month,
        top_n=args.topn
    )
    print(json.dumps({"recommendations": recs}, ensure_ascii=False, indent=2))
