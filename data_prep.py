# data_prep.py
# Loads CSVs, basic cleaning, builds user-item matrix and saves cleaned files.

import os
import pandas as pd
import numpy as np
import joblib

DATA_DIR = "data"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def load_movies(path):
    # tries to load movies with 'movieId','title','genres','overview' (overview optional)
    df = pd.read_csv(path)
    expected = ['movieId','title']
    if not all(c in df.columns for c in expected):
        raise ValueError(f"{path} must contain columns {expected}")
    # ensure columns exist
    if 'genres' not in df.columns:
        df['genres'] = ''
    if 'overview' not in df.columns:
        df['overview'] = ''
    # keep necessary columns
    df = df[['movieId','title','genres','overview']].drop_duplicates(subset=['movieId'])
    df['movieId'] = df['movieId'].astype(int)
    return df

def load_ratings(path):
    df = pd.read_csv(path)
    # basic check
    if not all(c in df.columns for c in ['userId','movieId','rating']):
        raise ValueError(f"{path} must contain columns ['userId','movieId','rating']")
    df = df[['userId','movieId','rating']].copy()
    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    return df

def build_user_item_matrix(ratings_df):
    # pivot to user x item matrix. fill missing with 0
    ui = ratings_df.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='mean').fillna(0)
    return ui

def save_pickle(obj, path):
    joblib.dump(obj, path)

def main():
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError("Put movies.csv and ratings.csv into ./data directory")
    movies = load_movies(movies_path)
    ratings = load_ratings(ratings_path)

    ui_matrix = build_user_item_matrix(ratings)
    print("User-Item matrix shape:", ui_matrix.shape)

    # save cleaned artifacts
    save_pickle(movies, os.path.join(OUT_DIR, "movies.pkl"))
    save_pickle(ratings, os.path.join(OUT_DIR, "ratings.pkl"))
    save_pickle(ui_matrix, os.path.join(OUT_DIR, "ui_matrix.pkl"))
    print("Saved movies.pkl, ratings.pkl, ui_matrix.pkl in", OUT_DIR)

if __name__ == "__main__":
    main()
