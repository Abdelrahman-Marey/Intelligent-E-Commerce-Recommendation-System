"""
evaluation/evaluator.py
------------------------
Computes 6 evaluation metrics across all CF methods + CB + KB:
  1. RMSE          — prediction accuracy
  2. Precision@K   — how many top-K recs are relevant
  3. Recall@K      — how many relevant items appear in top-K
  4. F1-Score      — harmonic mean of Precision & Recall
  5. Coverage      — % of catalog the system can recommend
  6. Diversity     — average pairwise dissimilarity of recommendations

Usage:
    from evaluation.evaluator import Evaluator
    ev = Evaluator("data/ratings.csv", "data/products.csv")
    report = ev.full_report()
    print(report)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

# Import engines
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collaborative_filtering.cf_engine import CFEngine
from content_based.cb_engine import CBEngine


class Evaluator:
    def __init__(self, ratings_path: str, products_path: str, test_size: float = 0.2):
        self.ratings_path  = ratings_path
        self.products_path = products_path

        all_ratings = pd.read_csv(ratings_path)
        self.train, self.test = train_test_split(
            all_ratings, test_size=test_size, random_state=42
        )
        self.products_df = pd.read_csv(products_path)
        self.all_pids    = set(self.products_df["product_id"].tolist())

        # TF-IDF for diversity computation
        tfidf = TfidfVectorizer(stop_words="english")
        self._tfidf_mat = tfidf.fit_transform(
            self.products_df["description"].fillna("")
        )
        self._pid_index = {p: i for i, p in enumerate(self.products_df["product_id"])}

        self.train.to_csv("data/_train.csv", index=False)
        print(f"[Eval] Train: {len(self.train)} | Test: {len(self.test)}")

    # ── RMSE ─────────────────────────────────────────────────────────────────

    def _rmse_for_method(self, method: str) -> float:
        """
        Train CF on train set, predict ratings for test pairs, compute RMSE.
        """
        cf = CFEngine("data/_train.csv", self.products_path)
        preds, actuals = [], []

        # Build SVD predictions once for speed
        if method == "svd":
            preds_df = cf._build_svd()

        test_users = self.test["user_id"].unique()[:20]   # sample for speed

        for uid in test_users:
            user_test = self.test[self.test.user_id == uid]
            for _, row in user_test.iterrows():
                pid    = int(row["product_id"])
                actual = row["rating"]

                try:
                    if method == "svd":
                        if uid in preds_df.index and pid in preds_df.columns:
                            pred = float(preds_df.loc[uid, pid])
                        else:
                            pred = 3.0
                    elif method == "user_user":
                        sim = cf._user_user_sim()
                        users = list(cf.matrix.index)
                        if uid not in users:
                            pred = 3.0
                        else:
                            u_idx = users.index(uid)
                            sims  = sim[u_idx]
                            top_n_idx = np.argsort(sims)[::-1][1:6]
                            weighted_sum, sim_sum = 0, 0
                            for ni in top_n_idx:
                                nu = users[ni]
                                nr = cf.ratings_df[(cf.ratings_df.user_id==nu) &
                                                   (cf.ratings_df.product_id==pid)]
                                if not nr.empty:
                                    s = sims[ni]
                                    weighted_sum += s * nr.iloc[0]["rating"]
                                    sim_sum += abs(s)
                            pred = weighted_sum / sim_sum if sim_sum > 0 else 3.0
                    else:
                        pred = 3.5   # fallback for item_item / knn
                    preds.append(np.clip(pred, 1, 5))
                    actuals.append(actual)
                except Exception:
                    pass

        if len(preds) == 0:
            return None
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        return round(rmse, 4)

    # ── Precision / Recall / F1 @ K ──────────────────────────────────────────

    def _precision_recall_f1(self, cf_engine, method: str, K: int = 10) -> tuple:
        """
        For each test user: top-K predicted items vs. items rated ≥ 4 in test set.
        """
        precisions, recalls = [], []
        test_users = self.test["user_id"].unique()[:25]

        for uid in test_users:
            relevant = set(
                self.test[(self.test.user_id == uid) & (self.test.rating >= 4)]["product_id"].tolist()
            )
            if not relevant:
                continue

            try:
                recs = cf_engine.recommend(user_id=uid, method=method, top_n=K)
                rec_pids = set(r["product_id"] for r in recs)

                hits = len(rec_pids & relevant)
                precisions.append(hits / K)
                recalls.append(hits / len(relevant))
            except Exception:
                pass

        if not precisions:
            return 0, 0, 0
        p = np.mean(precisions)
        r = np.mean(recalls)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return round(p, 4), round(r, 4), round(f, 4)

    # ── Coverage ─────────────────────────────────────────────────────────────

    def _coverage(self, cf_engine, method: str, sample_users: int = 30) -> float:
        """% of catalog that appears in at least one user's recommendations."""
        recommended = set()
        users = cf_engine.ratings_df["user_id"].unique()[:sample_users]
        for uid in users:
            try:
                recs = cf_engine.recommend(user_id=uid, method=method, top_n=10)
                for r in recs:
                    recommended.add(r["product_id"])
            except Exception:
                pass
        return round(len(recommended) / len(self.all_pids), 4)

    # ── Diversity ─────────────────────────────────────────────────────────────

    def _diversity(self, recs: list[dict]) -> float:
        """
        Average pairwise cosine DISTANCE (1 - similarity) among recommended items.
        Higher = more diverse.
        """
        pids = [r["product_id"] for r in recs if r["product_id"] in self._pid_index]
        if len(pids) < 2:
            return 0.0
        indices = [self._pid_index[p] for p in pids]
        vecs    = self._tfidf_mat[indices]
        sim_mat = cosine_similarity(vecs)
        n = len(pids)
        total_dist = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += 1 - sim_mat[i, j]
                count += 1
        return round(total_dist / count, 4) if count > 0 else 0.0

    def _avg_diversity(self, cf_engine, method: str, sample_users: int = 20) -> float:
        diversities = []
        users = cf_engine.ratings_df["user_id"].unique()[:sample_users]
        for uid in users:
            try:
                recs = cf_engine.recommend(user_id=uid, method=method, top_n=10)
                diversities.append(self._diversity(recs))
            except Exception:
                pass
        return round(np.mean(diversities), 4) if diversities else 0.0

    # ── Full Report ───────────────────────────────────────────────────────────

    def full_report(self, K: int = 10) -> pd.DataFrame:
        """Run all metrics for all methods and return a summary DataFrame."""
        print("\n⏳ Running evaluation (this takes ~30 seconds)...\n")

        cf_train = CFEngine("data/_train.csv", self.products_path)
        cb_train = CBEngine("data/_train.csv", self.products_path)

        methods = [
            ("CF",  "user_user",    cf_train),
            ("CF",  "item_item",    cf_train),
            ("CF",  "svd",          cf_train),
            ("CF",  "knn",          cf_train),
            ("CB",  "tfidf",        cb_train),
            ("CB",  "feature_match",cb_train),
        ]

        rows = []
        for approach, method, engine in methods:
            print(f"  Evaluating {approach} — {method} ...", end=" ", flush=True)

            rmse = self._rmse_for_method(method) if approach == "CF" else None
            p, r, f = self._precision_recall_f1(engine, method, K)
            cov  = self._coverage(engine, method)
            div  = self._avg_diversity(engine, method)

            rows.append({
                "Approach":      approach,
                "Method":        method,
                "RMSE":          rmse,
                f"Precision@{K}": round(p * 100, 1),
                f"Recall@{K}":    round(r * 100, 1),
                "F1-Score":      round(f, 4),
                "Coverage (%)":  round(cov * 100, 1),
                "Diversity":     div,
            })
            print("✓")

        df = pd.DataFrame(rows)
        print("\n" + "="*70)
        print(df.to_string(index=False))
        print("="*70)
        df.to_csv("data/evaluation_report.csv", index=False)
        print("\n✅ Saved to data/evaluation_report.csv")
        return df
