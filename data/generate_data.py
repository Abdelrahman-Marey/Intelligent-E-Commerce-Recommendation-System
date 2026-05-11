"""
generate_data.py
----------------
Generates synthetic e-commerce dataset:
  - users.csv
  - products.csv
  - ratings.csv
Run once before anything else.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ─── Products ────────────────────────────────────────────────────────────────
products_raw = [
    (0,  "iPhone 15 Pro",        "Electronics", "Apple",    999,  "smartphone ios camera premium flagship"),
    (1,  "Samsung Galaxy S24",   "Electronics", "Samsung",  849,  "smartphone android camera flagship"),
    (2,  "Sony WH-1000XM5",      "Electronics", "Sony",     349,  "headphones noise-cancelling audio wireless"),
    (3,  "MacBook Air M3",       "Electronics", "Apple",    1299, "laptop macos productivity premium apple"),
    (4,  "Nike Air Max 270",     "Sports",      "Nike",     150,  "shoes running sport athletic"),
    (5,  "Nike Running Kit",     "Sports",      "Nike",     89,   "sport fitness running accessories kit"),
    (6,  "IKEA MALM Desk",       "Home",        "IKEA",     199,  "furniture home workspace desk office"),
    (7,  "Kindle Paperwhite",    "Books",       "Amazon",   139,  "ereader books reading digital"),
    (8,  "iPad Air",             "Electronics", "Apple",    749,  "tablet ios productivity drawing apple"),
    (9,  "Adidas Ultraboost",    "Sports",      "Adidas",   180,  "shoes running boost sport premium"),
    (10, "IKEA KALLAX Shelf",    "Home",        "IKEA",     129,  "furniture storage home living shelf"),
    (11, "Levi's 501 Jeans",     "Fashion",     "Levi's",   79,   "jeans fashion classic denim casual"),
    (12, "Sony PlayStation 5",   "Electronics", "Sony",     499,  "gaming console entertainment ps5 sony"),
    (13, "Dyson V15",            "Home",        "Dyson",    699,  "vacuum home cleaning premium dyson"),
    (14, "Samsung 4K TV 55",     "Electronics", "Samsung",  799,  "tv 4k entertainment smart samsung"),
    (15, "Apple Watch Series 9", "Electronics", "Apple",    399,  "smartwatch fitness health apple wearable"),
    (16, "Yoga Mat Pro",         "Sports",      "Gaiam",    45,   "yoga fitness mat sport exercise"),
    (17, "Coffee Table Oak",     "Home",        "IKEA",     249,  "furniture home living coffee table"),
    (18, "Python Programming",   "Books",       "OReilly",  49,   "programming books python coding tech"),
    (19, "Zara Jacket",          "Fashion",     "Zara",     129,  "jacket fashion outerwear zara style"),
    (20, "AirPods Pro 2",        "Electronics", "Apple",    249,  "earbuds apple audio wireless premium"),
    (21, "Dumbbells Set",        "Sports",      "Bowflex",  299,  "weights fitness sport strength training"),
    (22, "Scented Candles Set",  "Home",        "Yankee",   35,   "home decor candles fragrance living"),
    (23, "The Great Gatsby",     "Books",       "Penguin",  15,   "novel classic fiction books reading"),
    (24, "H&M Summer Dress",     "Fashion",     "H&M",      59,   "dress fashion summer casual hm"),
]

products_df = pd.DataFrame(products_raw,
    columns=["product_id","name","category","brand","price","description"])

# Simulate realistic ratings per product
base_ratings = {
    0:4.8, 1:4.6, 2:4.7, 3:4.9, 4:4.4, 5:4.3, 6:4.2, 7:4.6, 8:4.7,
    9:4.5, 10:4.3, 11:4.4, 12:4.8, 13:4.7, 14:4.5, 15:4.6, 16:4.2,
    17:4.1, 18:4.5, 19:4.0, 20:4.7, 21:4.3, 22:4.2, 23:4.4, 24:4.0,
}
products_df["avg_rating"] = products_df["product_id"].map(base_ratings)
products_df["num_reviews"] = np.random.randint(50, 5000, len(products_df))
products_df.to_csv("data/products.csv", index=False)
print(f"✅ products.csv — {len(products_df)} products")

# ─── Users ───────────────────────────────────────────────────────────────────
N_USERS = 50

personas = [
    ("Tech Enthusiast", ["Electronics"],          ["Apple","Sony","Samsung"]),
    ("Budget Shopper",  ["Sports","Books","Fashion"],["Nike","Amazon","H&M"]),
    ("Luxury Buyer",    ["Electronics","Home"],   ["Apple","Dyson","Sony"]),
    ("Sports Fan",      ["Sports"],               ["Nike","Adidas","Bowflex"]),
    ("Home Decorator",  ["Home","Fashion"],        ["IKEA","Dyson","Yankee"]),
]

users = []
for i in range(N_USERS):
    persona = personas[i % len(personas)]
    users.append({
        "user_id":    i,
        "name":       f"User_{i:02d}",
        "age":        np.random.randint(18, 60),
        "persona":    persona[0],
        "pref_cats":  "|".join(persona[1]),
        "pref_brands":"|".join(persona[2]),
    })

users_df = pd.DataFrame(users)
users_df.to_csv("data/users.csv", index=False)
print(f"✅ users.csv — {N_USERS} users")

# ─── Ratings ─────────────────────────────────────────────────────────────────
cat_map = {p[0]: p[2] for p in products_raw}   # pid -> category
brand_map = {p[0]: p[3] for p in products_raw} # pid -> brand

ratings = []
for u in users:
    uid = u["user_id"]
    pref_cats   = u["pref_cats"].split("|")
    pref_brands = u["pref_brands"].split("|")

    n_ratings = np.random.randint(5, 15)
    rated_pids = set()

    for pid in range(len(products_raw)):
        cat   = products_raw[pid][2]
        brand = products_raw[pid][3]
        base  = base_ratings[pid]

        if cat in pref_cats:
            prob = 0.70
        elif brand in pref_brands:
            prob = 0.40
        else:
            prob = 0.12

        if np.random.rand() < prob and pid not in rated_pids and len(rated_pids) < n_ratings:
            noise  = np.random.normal(0, 0.4)
            rating = np.clip(round(base + noise), 1, 5)
            ratings.append({"user_id": uid, "product_id": pid, "rating": int(rating)})
            rated_pids.add(pid)

ratings_df = pd.DataFrame(ratings)
ratings_df.to_csv("data/ratings.csv", index=False)
print(f"✅ ratings.csv — {len(ratings_df)} ratings")
print(f"   Sparsity: {1 - len(ratings_df)/(N_USERS*len(products_raw)):.1%}")
print("\n🎉 Dataset ready!")
