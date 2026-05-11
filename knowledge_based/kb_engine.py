"""
knowledge_based/kb_engine.py
-----------------------------
Knowledge-Based Recommender using constraint filtering + case-based scoring.

Usage:
    from knowledge_based.kb_engine import KBEngine
    kb = KBEngine("data/products.csv")
    recs = kb.recommend(category="Electronics", max_price=500, min_rating=4.0, top_n=5)
"""

import pandas as pd
import numpy as np


class KBEngine:
    def __init__(self, products_path: str):
        self.products_df = pd.read_csv(products_path)
        print(f"[KB] Loaded {len(self.products_df)} products")

    def recommend(
        self,
        category:   str   = None,
        brand:      str   = None,
        max_price:  float = None,
        min_price:  float = None,
        min_rating: float = None,
        keywords:   list  = None,
        top_n:      int   = 5,
    ) -> list[dict]:
        """
        Filter products by hard constraints, then rank by soft score.

        Parameters
        ----------
        category   : exact category name  (e.g. "Electronics")
        brand      : exact brand name     (e.g. "Apple")
        max_price  : upper price bound
        min_price  : lower price bound
        min_rating : minimum avg_rating
        keywords   : list of words that must appear in description
        top_n      : number of results
        """
        df = self.products_df.copy()

        # ── Hard Constraints (Filter) ────────────────────────────────────────
        active_constraints = []

        if category:
            df = df[df["category"].str.lower() == category.lower()]
            active_constraints.append(f"category = {category}")

        if brand:
            df = df[df["brand"].str.lower() == brand.lower()]
            active_constraints.append(f"brand = {brand}")

        if max_price is not None:
            df = df[df["price"] <= max_price]
            active_constraints.append(f"price ≤ ${max_price}")

        if min_price is not None:
            df = df[df["price"] >= min_price]
            active_constraints.append(f"price ≥ ${min_price}")

        if min_rating is not None:
            df = df[df["avg_rating"] >= min_rating]
            active_constraints.append(f"rating ≥ {min_rating}★")

        if keywords:
            for kw in keywords:
                df = df[df["description"].str.contains(kw, case=False, na=False)]
            active_constraints.append(f"keywords: {', '.join(keywords)}")

        if df.empty:
            return []

        # ── Soft Scoring (Rank within filtered set) ───────────────────────────
        df = df.copy()

        # Normalize price (lower = better within budget)
        if len(df) > 1:
            p_min, p_max = df["price"].min(), df["price"].max()
            span = p_max - p_min if p_max != p_min else 1
            df["price_score"] = 1 - (df["price"] - p_min) / span
        else:
            df["price_score"] = 1.0

        # Normalize rating
        df["rating_score"] = (df["avg_rating"] - 1) / 4   # scale 1-5 → 0-1

        # Popularity proxy (num_reviews)
        if "num_reviews" in df.columns and len(df) > 1:
            r_min, r_max = df["num_reviews"].min(), df["num_reviews"].max()
            span = r_max - r_min if r_max != r_min else 1
            df["pop_score"] = (df["num_reviews"] - r_min) / span
        else:
            df["pop_score"] = 0.5

        df["total_score"] = (
            df["rating_score"] * 0.50 +
            df["price_score"]  * 0.30 +
            df["pop_score"]    * 0.20
        )

        df = df.sort_values("total_score", ascending=False).head(top_n)

        constraint_str = ", ".join(active_constraints) if active_constraints else "no filters"

        results = []
        for _, row in df.iterrows():
            results.append({
                "product_id":  int(row["product_id"]),
                "name":        row["name"],
                "category":    row["category"],
                "brand":       row["brand"],
                "price":       row["price"],
                "avg_rating":  row["avg_rating"],
                "score":       round(row["total_score"], 4),
                "method":      "Knowledge-Based",
                "explanation": (f"Recommended based on your selected requirements: "
                                f"{constraint_str}. "
                                f"Ranked by rating ({row['avg_rating']}★), "
                                f"value (${row['price']}), and popularity.")
            })
        return results
