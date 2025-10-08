# train_models.py
# Builds:
# 1) Item-item similarity matrix using cosine on columns of user-item matrix (item-based CF)
# 2) Content-based TF-IDF vectors on 'genres'+'overview' and cosine similarity
# Saves models as joblib pickles.

import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

MODELDIR = "models"
os.makedirs(MODELDIR, exist_ok=True)

def load_artifacts():
    movies = joblib.load(os.path.join(MODELDIR, "movies.pkl"))
    ratings = joblib.load(os.path.join(MODELDIR, "ratings.pkl"))
    ui = joblib.load(os.path.join(MODELDIR, "ui_matrix.pkl"))
    return movies, ratings, ui

def build_item_similarity(ui_matrix):
    # ui_matrix: rows=userId, cols=movieId
    # item vectors are columns
    item_matrix = ui_matrix.T.values  # shape (n_items, n_users)
    # compute cosine similarity between items
    sim = cosine_similarity(item_matrix)
    # index mapping
    movie_ids = list(ui_matrix.columns)
    return sim, movie_ids

def build_content_similarity(movies_df):
    # build a text field combining genres and overview
    txt = (movies_df['genres'].fillna('') + " " + movies_df['overview'].fillna('')).astype(str)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(txt)
    content_sim = cosine_similarity(X)
    movie_ids = movies_df['movieId'].tolist()
    return content_sim, movie_ids, tfidf

def main():
    movies, ratings, ui = load_artifacts()
    print("Building item similarity (CF)...")
    item_sim, item_movie_ids = build_item_similarity(ui)
    print("Item similarity matrix:", item_sim.shape)

    print("Building content similarity (TF-IDF)...")
    content_sim, content_movie_ids, tfidf = build_content_similarity(movies)
    print("Content similarity matrix:", content_sim.shape)

    # Save all
    joblib.dump({
        "item_sim": item_sim,
        "item_movie_ids": item_movie_ids
    }, os.path.join(MODELDIR, "item_sim.pkl"))

    joblib.dump({
        "content_sim": content_sim,
        "content_movie_ids": content_movie_ids,
        "tfidf": tfidf
    }, os.path.join(MODELDIR, "content_sim.pkl"))

    print("Saved item_sim.pkl and content_sim.pkl to", MODELDIR)

if __name__ == "__main__":
    main()
