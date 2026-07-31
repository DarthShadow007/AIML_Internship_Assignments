"""
Movie Recommendation System
Collaborative filtering via matrix factorization (SVD) on the MovieLens 100K dataset.
Downloads the dataset automatically on first run.
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA_DIR = "ml-100k"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = "ml-100k.zip"

# -------------------- 1. Download dataset --------------------
if not os.path.exists(DATA_DIR):
    print("Downloading MovieLens 100K dataset...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(".")
    print("Download complete.")

ratings_path = os.path.join(DATA_DIR, "u.data")
ratings = pd.read_csv(
    ratings_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
)
print(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users "
      f"and {ratings['item_id'].nunique()} movies.")

# -------------------- 2. Build user-item matrix --------------------
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

n_users = ratings["user_id"].max()
n_items = ratings["item_id"].max()

train_matrix = np.zeros((n_users, n_items))
for row in train.itertuples():
    train_matrix[row.user_id - 1, row.item_id - 1] = row.rating

# -------------------- 3. Matrix factorization via SVD --------------------
# Mean-center ratings per user before factorizing
user_means = np.true_divide(
    train_matrix.sum(axis=1), (train_matrix != 0).sum(axis=1), where=(train_matrix != 0).sum(axis=1) != 0
)
user_means = np.nan_to_num(user_means)
train_centered = train_matrix - user_means[:, None]
train_centered[train_matrix == 0] = 0  # keep unobserved entries at 0 after centering

k = 20  # number of latent factors
U, sigma, Vt = np.linalg.svd(train_centered, full_matrices=False)
U_k, sigma_k, Vt_k = U[:, :k], np.diag(sigma[:k]), Vt[:k, :]

predicted = np.dot(np.dot(U_k, sigma_k), Vt_k) + user_means[:, None]

# -------------------- 4. Evaluate on test set --------------------
y_true, y_pred = [], []
for row in test.itertuples():
    y_true.append(row.rating)
    pred = predicted[row.user_id - 1, row.item_id - 1]
    y_pred.append(np.clip(pred, 1, 5))

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\nTest RMSE: {rmse:.4f}")

# -------------------- 5. Recommend top-N movies for a sample user --------------------
items_path = os.path.join(DATA_DIR, "u.item")
movies = pd.read_csv(
    items_path, sep="|", encoding="latin-1", header=None,
    names=["item_id", "title"] + [f"col{i}" for i in range(22)]
)[["item_id", "title"]]

sample_user = 1
already_rated = set(ratings[ratings["user_id"] == sample_user]["item_id"])
user_predicted_ratings = predicted[sample_user - 1]

top_n = 10
candidate_ids = [i for i in range(1, n_items + 1) if i not in already_rated]
top_ids = sorted(candidate_ids, key=lambda i: user_predicted_ratings[i - 1], reverse=True)[:top_n]

print(f"\nTop {top_n} recommendations for user {sample_user}:")
for movie_id in top_ids:
    title = movies[movies["item_id"] == movie_id]["title"].values[0]
    print(f"  {title} (predicted rating: {user_predicted_ratings[movie_id - 1]:.2f})")