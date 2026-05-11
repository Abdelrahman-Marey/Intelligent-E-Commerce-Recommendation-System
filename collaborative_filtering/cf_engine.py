"""
collaborative_filtering/cf_engine.py
--------------------------------------
Implements 4 CF methods:
  1. User-User  (Cosine Similarity)
  2. Item-Item  (Pearson Correlation)
  3. SVD        (Matrix Factorization via scipy)
  4. KNN        (scikit-learn NearestNeighbors)

Usage:
    from collaborative_filtering.cf_engine import CFEngine
    cf = CFEngine("data/ratings.csv", "data/products.csv")
    recs = cf.recommend(user_id=0, method="svd", top_n=5)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")


class CFEngine:
    def __init__(self, ratings_path: str, products_path: str):
        self.ratings_df = pd.read_csv(ratings_path)
        self.products_df = pd.read_csv(products_path)

        # Build User-Item matrix (rows=users, cols=products)
        self.matrix = self.ratings_df.pivot_table(
            index="user_id", columns="product_id", values="rating"
        )
        self.matrix_filled = self.matrix.fillna(0)

        # Pre-compute similarity matrices
        self._user_sim_matrix = None
        self._item_sim_matrix = None
        self._svd_predictions = None

        print(
            f"[CF] Matrix shape: {self.matrix.shape} "
            f"| Sparsity: {self.matrix.isna().mean().mean():.1%}"
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _not_rated(self, user_id: int) -> list[int]:
        """Return product IDs the user has NOT rated yet."""
        rated = self.ratings_df[self.ratings_df.user_id == user_id][
            "product_id"
        ].tolist()
        all_ids = self.products_df["product_id"].tolist()
        return [p for p in all_ids if p not in rated]

    def _product_info(self, product_id: int) -> dict:
        row = self.products_df[self.products_df.product_id == product_id].iloc[0]
        return row.to_dict()

    # ── Method 1: User-User CF ───────────────────────────────────────────────

    def _user_user_sim(self) -> np.ndarray:
        if self._user_sim_matrix is None:
            self._user_sim_matrix = cosine_similarity(self.matrix_filled)
        return self._user_sim_matrix

    def recommend_user_user(
        self, user_id: int, top_n: int = 5, k_neighbors: int = 5, **kwargs
    ) -> list[dict]:
        sim = self._user_user_sim()
        users = list(self.matrix.index)
        if user_id not in users:
            return []

        u_idx = users.index(user_id)
        sim_scores = list(enumerate(sim[u_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the user itself
        neighbors = [(users[i], s) for i, s in sim_scores if users[i] != user_id][
            :k_neighbors
        ]

        not_rated = self._not_rated(user_id)
        scores = {}

        for n_uid, sim_score in neighbors:
            n_ratings = self.ratings_df[self.ratings_df.user_id == n_uid]
            for _, row in n_ratings.iterrows():
                pid = int(row["product_id"])
                if pid in not_rated:
                    scores[pid] = scores.get(pid, 0) + sim_score * row["rating"]

        results = []
        for pid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]:
            info = self._product_info(pid)
            neighbor_names = [f"User_{n}" for n, _ in neighbors[:3]]
            results.append(
                {
                    "product_id": pid,
                    "name": info["name"],
                    "category": info["category"],
                    "brand": info["brand"],
                    "price": info["price"],
                    "avg_rating": info["avg_rating"],
                    "score": round(score, 3),
                    "method": "User-User CF",
                    "explanation": (
                        f"Recommended because {k_neighbors} similar users "
                        f"(e.g. {', '.join(neighbor_names)}) "
                        f"rated this highly."
                    ),
                }
            )
        return results

    # ── Method 2: Item-Item CF ───────────────────────────────────────────────

    def _item_item_sim(self) -> pd.DataFrame:
        if self._item_sim_matrix is None:
            item_matrix = self.matrix_filled.T  # items as rows
            sim = cosine_similarity(item_matrix)
            self._item_sim_matrix = pd.DataFrame(
                sim, index=self.matrix.columns, columns=self.matrix.columns
            )
        return self._item_sim_matrix

    def recommend_item_item(self, user_id: int, top_n: int = 5, **kwargs) -> list[dict]:
        sim_df = self._item_item_sim()
        user_ratings = self.ratings_df[self.ratings_df.user_id == user_id]
        if user_ratings.empty:
            return []

        not_rated = self._not_rated(user_id)
        scores = {}

        for _, row in user_ratings.iterrows():
            rated_pid = int(row["product_id"])
            if rated_pid not in sim_df.index:
                continue
            item_sims = sim_df[rated_pid].drop(index=rated_pid, errors="ignore")
            for cand_pid, sim_score in item_sims.items():
                if cand_pid in not_rated:
                    scores[cand_pid] = (
                        scores.get(cand_pid, 0) + sim_score * row["rating"]
                    )

        results = []
        for pid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]:
            info = self._product_info(pid)
            # Find most similar rated item for explanation
            best_match = user_ratings.iloc[0]["product_id"]
            best_name = self._product_info(int(best_match))["name"]
            results.append(
                {
                    "product_id": pid,
                    "name": info["name"],
                    "category": info["category"],
                    "brand": info["brand"],
                    "price": info["price"],
                    "avg_rating": info["avg_rating"],
                    "score": round(score, 3),
                    "method": "Item-Item CF",
                    "explanation": (
                        f"Recommended because it is similar to "
                        f'"{best_name}" which you rated highly.'
                    ),
                }
            )
        return results

    # ── Method 3: SVD ────────────────────────────────────────────────────────

    def _build_svd(self, n_factors: int = 10):
        if self._svd_predictions is None:
            M = self.matrix_filled.values.astype(float)
            # Mean-center
            user_means = np.nanmean(np.where(M == 0, np.nan, M), axis=1, keepdims=True)
            user_means = np.nan_to_num(user_means, nan=3.0)
            M_centered = M - user_means
            M_centered[M == 0] = 0

            k = min(n_factors, min(M.shape) - 1)
            U, sigma, Vt = svds(M_centered, k=k)
            sigma_diag = np.diag(sigma)
            self._svd_predictions = pd.DataFrame(
                np.dot(np.dot(U, sigma_diag), Vt) + user_means,
                index=self.matrix.index,
                columns=self.matrix.columns,
            )
        return self._svd_predictions

    def recommend_svd(
        self, user_id: int, top_n: int = 5, n_factors: int = 10, **kwargs
    ) -> list[dict]:
        preds = self._build_svd(n_factors)
        if user_id not in preds.index:
            return []

        not_rated = self._not_rated(user_id)
        user_preds = preds.loc[user_id]
        cand_preds = user_preds[user_preds.index.isin(not_rated)].sort_values(
            ascending=False
        )

        results = []
        for pid, pred_rating in cand_preds.head(top_n).items():
            info = self._product_info(pid)
            results.append(
                {
                    "product_id": pid,
                    "name": info["name"],
                    "category": info["category"],
                    "brand": info["brand"],
                    "price": info["price"],
                    "avg_rating": info["avg_rating"],
                    "score": round(float(pred_rating), 3),
                    "method": "SVD (Matrix Factorization)",
                    "explanation": (
                        f"SVD predicted rating: {min(5.0, max(1.0, pred_rating)):.1f}/5. "
                        f"Latent factor analysis ({n_factors} factors) detected hidden "
                        f"preference patterns in the rating matrix."
                    ),
                }
            )
        return results

    # ── Method 4: KNN ────────────────────────────────────────────────────────

    def recommend_knn(
        self, user_id: int, top_n: int = 5, k: int = 5, **kwargs
    ) -> list[dict]:
        users = list(self.matrix.index)
        if user_id not in users:
            return []

        X = self.matrix_filled.values
        knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
        knn.fit(X)

        u_idx = users.index(user_id)
        distances, indices = knn.kneighbors(X[u_idx].reshape(1, -1))
        neighbor_idxs = [i for i in indices[0] if users[i] != user_id][:k]

        not_rated = self._not_rated(user_id)
        scores = {}
        for n_idx in neighbor_idxs:
            n_uid = users[n_idx]
            sim = 1 - distances[0][list(indices[0]).index(n_idx)]
            n_ratings = self.ratings_df[self.ratings_df.user_id == n_uid]
            for _, row in n_ratings.iterrows():
                pid = int(row["product_id"])
                if pid in not_rated:
                    scores[pid] = scores.get(pid, 0) + sim * row["rating"]

        results = []
        for pid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]:
            info = self._product_info(pid)
            results.append(
                {
                    "product_id": pid,
                    "name": info["name"],
                    "category": info["category"],
                    "brand": info["brand"],
                    "price": info["price"],
                    "avg_rating": info["avg_rating"],
                    "score": round(score, 3),
                    "method": "KNN CF",
                    "explanation": (
                        f"KNN found {k} nearest neighbors (cosine distance). "
                        f"These users share similar rating patterns with you."
                    ),
                }
            )
        return results

    # ── Unified entry point ──────────────────────────────────────────────────

    def recommend(
        self, user_id: int, method: str = "svd", top_n: int = 5, **kwargs
    ) -> list[dict]:
        """
        method: "user_user" | "item_item" | "svd" | "knn"
        """
        dispatch = {
            "user_user": self.recommend_user_user,
            "item_item": self.recommend_item_item,
            "svd": self.recommend_svd,
            "knn": self.recommend_knn,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)}")
        return dispatch[method](user_id=user_id, top_n=top_n, **kwargs)
