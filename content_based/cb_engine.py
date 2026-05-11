"""
content_based/cb_engine.py
---------------------------
Implements 2 Content-Based methods:
  1. TF-IDF Cosine Similarity  (on product descriptions)
  2. Category + Brand Matching (weighted feature overlap)

Usage:
    from content_based.cb_engine import CBEngine
    cb = CBEngine("data/ratings.csv", "data/products.csv")
    recs = cb.recommend(user_id=0, method="tfidf", top_n=5)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CBEngine:
    def __init__(self, ratings_path: str, products_path: str):
        self.ratings_df  = pd.read_csv(ratings_path)
        self.products_df = pd.read_csv(products_path).set_index("product_id")

        # ── TF-IDF matrix over product descriptions ──────────────────────────
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.products_df["description"].fillna("")
        )
        self.pid_list = list(self.products_df.index)

        print(f"[CB] TF-IDF matrix: {self.tfidf_matrix.shape} "
              f"| Vocab size: {len(self.tfidf.vocabulary_)}")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _user_liked_products(self, user_id: int, min_rating: int = 4) -> list[int]:
        """Return product IDs rated >= min_rating by this user."""
        df = self.ratings_df
        liked = df[(df.user_id == user_id) & (df.rating >= min_rating)]["product_id"].tolist()
        return liked

    def _not_rated(self, user_id: int) -> list[int]:
        rated = self.ratings_df[self.ratings_df.user_id == user_id]["product_id"].tolist()
        return [p for p in self.pid_list if p not in rated]

    # ── Method 1: TF-IDF Cosine Similarity ──────────────────────────────────

    def _build_user_profile_tfidf(self, liked_pids: list[int]) -> np.ndarray:
        """
        User profile = weighted average of TF-IDF vectors of liked products.
        Weight = user's rating for that product.
        """
        if not liked_pids:
            return None
        indices = [self.pid_list.index(p) for p in liked_pids if p in self.pid_list]
        ratings = []
        for p in liked_pids:
            r = self.ratings_df[
                (self.ratings_df.product_id == p)
            ]["rating"].values
            ratings.append(float(r[0]) if len(r) > 0 else 4.0)

        vectors = self.tfidf_matrix[indices].toarray()
        weights = np.array(ratings).reshape(-1, 1)
        profile = np.average(vectors, axis=0, weights=weights.flatten())
        return profile.reshape(1, -1)

    def recommend_tfidf(self, user_id: int, top_n: int = 5) -> list[dict]:
        liked = self._user_liked_products(user_id)
        if not liked:
            liked = self.ratings_df[self.ratings_df.user_id == user_id]["product_id"].tolist()
        if not liked:
            return []

        profile = self._build_user_profile_tfidf(liked)
        if profile is None:
            return []

        not_rated_pids = self._not_rated(user_id)
        cand_indices   = [self.pid_list.index(p) for p in not_rated_pids if p in self.pid_list]
        cand_pids      = [self.pid_list[i] for i in cand_indices]

        cand_vectors = self.tfidf_matrix[cand_indices]
        sims = cosine_similarity(profile, cand_vectors)[0]

        top_indices = np.argsort(sims)[::-1][:top_n]

        # Top keywords in user profile for explanation
        feature_names = self.tfidf.get_feature_names_out()
        top_kw_idx    = np.argsort(profile[0])[::-1][:4]
        top_keywords  = [feature_names[i] for i in top_kw_idx]

        results = []
        liked_names = [self.products_df.loc[p, "name"] for p in liked[:2] if p in self.products_df.index]
        for i in top_indices:
            pid  = cand_pids[i]
            info = self.products_df.loc[pid]
            results.append({
                "product_id":  pid,
                "name":        info["name"],
                "category":    info["category"],
                "brand":       info["brand"],
                "price":       info["price"],
                "avg_rating":  info["avg_rating"],
                "score":       round(float(sims[i]), 4),
                "method":      "Content-Based (TF-IDF Cosine)",
                "explanation": (f"Recommended because it matches your interest profile. "
                                f"Top keywords: {', '.join(top_keywords)}. "
                                f"Similar to: {', '.join(liked_names)}.")
            })
        return results

    # ── Method 2: Category + Brand Feature Matching ──────────────────────────

    def recommend_feature_match(self, user_id: int, top_n: int = 5) -> list[dict]:
        user_ratings = self.ratings_df[self.ratings_df.user_id == user_id]
        if user_ratings.empty:
            return []

        # Build preference profile from ratings
        cat_scores   = {}
        brand_scores = {}
        price_points = []

        for _, row in user_ratings.iterrows():
            pid = row["product_id"]
            r   = row["rating"]
            if pid not in self.products_df.index:
                continue
            info = self.products_df.loc[pid]
            cat_scores[info["category"]]   = cat_scores.get(info["category"], 0)   + r
            brand_scores[info["brand"]]    = brand_scores.get(info["brand"], 0)     + r
            price_points.append(info["price"])

        fav_cat   = max(cat_scores,   key=cat_scores.get)   if cat_scores   else None
        fav_brand = max(brand_scores, key=brand_scores.get) if brand_scores else None
        avg_price = np.mean(price_points) if price_points else 500

        not_rated = self._not_rated(user_id)
        results   = []

        for pid in not_rated:
            if pid not in self.products_df.index:
                continue
            info = self.products_df.loc[pid]

            cat_match   = 1.0 if info["category"] == fav_cat   else 0.2
            brand_match = 0.8 if info["brand"]    == fav_brand else 0.1
            # Price proximity (closer to avg = higher score)
            price_diff  = abs(info["price"] - avg_price) / (avg_price + 1)
            price_score = max(0, 1 - price_diff)
            rating_norm = info["avg_rating"] / 5.0

            score = (cat_match   * 0.40 +
                     brand_match * 0.30 +
                     price_score * 0.15 +
                     rating_norm * 0.15)

            explanation_parts = []
            if cat_match == 1.0:
                explanation_parts.append(f'matches your preferred category "{fav_cat}"')
            if brand_match == 0.8:
                explanation_parts.append(f'matches your preferred brand "{fav_brand}"')
            explanation_parts.append(f"avg price near your usual spend (${avg_price:.0f})")

            results.append({
                "product_id":  pid,
                "name":        info["name"],
                "category":    info["category"],
                "brand":       info["brand"],
                "price":       info["price"],
                "avg_rating":  info["avg_rating"],
                "score":       round(score, 4),
                "method":      "Content-Based (Feature Match)",
                "explanation": "Recommended because it " + ", ".join(explanation_parts) + "."
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

    # ── Unified entry point ──────────────────────────────────────────────────

    def recommend(self, user_id: int, method: str = "tfidf", top_n: int = 5) -> list[dict]:
        """
        method: "tfidf" | "feature_match"
        """
        if method == "tfidf":
            return self.recommend_tfidf(user_id, top_n)
        elif method == "feature_match":
            return self.recommend_feature_match(user_id, top_n)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose: 'tfidf' or 'feature_match'")
