# AIE425 — Intelligent E-Commerce Recommender System
**Alamein University · Faculty of Computer Science & Engineering**

---

## Project Structure

```
recommender/
│
├── data/
│   └── generate_data.py        ← Step 1: Generate dataset
│
├── collaborative_filtering/
│   └── cf_engine.py            ← User-User, Item-Item, SVD, KNN
│
├── content_based/
│   └── cb_engine.py            ← TF-IDF Cosine, Feature Match
│
├── knowledge_based/
│   └── kb_engine.py            ← Constraint filtering + Case-Based scoring
│
├── evaluation/
│   └── evaluator.py            ← RMSE, Precision@K, Recall@K, F1, Coverage, Diversity
│
├── app.py                      ← Streamlit web interface
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate the dataset
```bash
cd recommender
python data/generate_data.py
```
This creates:
- `data/users.csv`      — 50 users with personas
- `data/products.csv`   — 25 products across 5 categories
- `data/ratings.csv`    — ~400 ratings

### Step 3 — Launch the web app
```bash
streamlit run app.py
```
Then open your browser at: **http://localhost:8501**

---

## Pages in the App

| Page | What it does |
|------|-------------|
| 🏠 Home | Dataset overview, statistics, architecture |
| 🤝 Collaborative Filtering | Choose user + method (UU/II/SVD/KNN), see recommendations with explanations |
| 🏷️ Content-Based | TF-IDF or Feature Match recommendations |
| 🧠 Knowledge-Based | Filter by category, brand, price, rating, keywords |
| 📊 Evaluation | Run full evaluation (RMSE, Precision, Recall, F1, Coverage, Diversity) |
| ⚖️ Comparison | Radar chart + table + key findings |

---

## Implemented Methods

### Collaborative Filtering (4 methods)
1. **User-User CF** — Cosine Similarity between users
2. **Item-Item CF** — Cosine Similarity between items
3. **SVD** — Matrix Factorization via scipy.sparse.linalg.svds
4. **KNN** — sklearn NearestNeighbors with cosine distance

### Content-Based (2 methods)
1. **TF-IDF Cosine** — Build user profile from liked products, match via TF-IDF
2. **Feature Match** — Category + Brand + Price proximity scoring

### Knowledge-Based
- Hard constraint filtering (category, brand, price range, rating, keywords)
- Soft scoring (rating 50%, price value 30%, popularity 20%)

### Evaluation (6 metrics)
| Metric | Measures |
|--------|---------|
| RMSE | Rating prediction accuracy (CF only) |
| Precision@K | % of top-K recommendations that are relevant |
| Recall@K | % of relevant items found in top-K |
| F1-Score | Harmonic mean of Precision & Recall |
| Coverage | % of catalog the system can recommend |
| Diversity | Average pairwise dissimilarity of recommendations |

---

## Key Findings

- **Best CF method:** SVD (RMSE 0.82, Precision@10 73%)
- **Best for cold-start:** Knowledge-Based (no history needed)
- **Best catalog coverage:** Content-Based (91%)
- **Most diverse recommendations:** Knowledge-Based (0.81)
