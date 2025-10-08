# recommend.py
# Functions to get recommendations using:
# - Item-based Collaborative Filtering (neighbourhood)
# - Content-based similarity
# - Hybrid (weighted)

import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

MODELDIR = "models"

# load artifacts
movies = joblib.load(f"{MODELDIR}/movies.pkl")
ratings = joblib.load(f"{MODELDIR}/ratings.pkl")
ui_matrix = joblib.load(f"{MODELDIR}/ui_matrix.pkl")
item_sim_data = joblib.load(f"{MODELDIR}/item_sim.pkl")
content_sim_data = joblib.load(f"{MODELDIR}/content_sim.pkl")

item_sim = item_sim_data["item_sim"]
item_movie_ids = item_sim_data["item_movie_ids"]  # columns order of ui_matrix
content_sim = content_sim_data["content_sim"]
content_movie_ids = content_sim_data["content_movie_ids"]

movieid_to_index_item = {mid: idx for idx, mid in enumerate(item_movie_ids)}
movieid_to_index_content = {mid: idx for idx, mid in enumerate(content_movie_ids)}

def get_top_n_similar_items_itemcf(movie_id, top_n=10):
    # use item_sim matrix
    if movie_id not in movieid_to_index_item:
        return []
    idx = movieid_to_index_item[movie_id]
    sims = item_sim[idx]
    top_idx = np.argsort(sims)[::-1][1:top_n+1]  # skip self
    top_movie_ids = [item_movie_ids[i] for i in top_idx]
    scores = sims[top_idx]
    return list(zip(top_movie_ids, scores))

def get_top_n_similar_items_content(movie_id, top_n=10):
    if movie_id not in movieid_to_index_content:
        return []
    idx = movieid_to_index_content[movie_id]
    sims = content_sim[idx]
    top_idx = np.argsort(sims)[::-1][1:top_n+1]
    top_movie_ids = [content_movie_ids[i] for i in top_idx]
    scores = sims[top_idx]
    return list(zip(top_movie_ids, scores))

def recommend_for_user_itemcf(user_id, top_n=10):
    # For each item the user rated, aggregate neighbor items weighted by user's rating
    if user_id not in ui_matrix.index:
        return []
    user_vector = ui_matrix.loc[user_id]
    rated = user_vector[user_vector > 0]
    score_agg = defaultdict(float)
    weight_agg = defaultdict(float)
    for movie_id, rating in rated.items():
        if movie_id not in movieid_to_index_item:
            continue
        idx = movieid_to_index_item[movie_id]
        sims = item_sim[idx]
        for j, sim_score in enumerate(sims):
            candidate_mid = item_movie_ids[j]
            if candidate_mid in rated.index:  # skip already rated
                continue
            score_agg[candidate_mid] += sim_score * rating
            weight_agg[candidate_mid] += abs(sim_score)
    # final scores = score_agg / weight_agg
    final_scores = []
    for mid in score_agg:
        if weight_agg[mid] > 0:
            final_scores.append((mid, score_agg[mid] / weight_agg[mid]))
    final_scores.sort(key=lambda x: x[1], reverse=True)
    top = final_scores[:top_n]
    return top

def recommend_for_user_hybrid(user_id, top_n=10, alpha=0.6):
    # hybrid: alpha * itemCF + (1-alpha) * content-based seeding
    # Step 1: itemCF scores
    itemcf = dict(recommend_for_user_itemcf(user_id, top_n=top_n*3))
    # Step 2: content-based -> for each movie user rated highly, add content neighbors
    if user_id not in ui_matrix.index:
        return []
    user_vector = ui_matrix.loc[user_id]
    high_rated = user_vector[user_vector >= 4]  # seed from high-rated
    content_scores = defaultdict(float)
    content_weights = defaultdict(float)
    for movie_id, rating in high_rated.items():
        if movie_id not in movieid_to_index_content:
            continue
        neighbors = get_top_n_similar_items_content(movie_id, top_n=25)
        for mid, sim in neighbors:
            if mid in user_vector[user_vector > 0].index:
                continue
            content_scores[mid] += sim * rating
            content_weights[mid] += sim
    normalized_content = {}
    for mid in content_scores:
        if content_weights[mid] > 0:
            normalized_content[mid] = content_scores[mid] / content_weights[mid]

    # Merge
    merged = {}
    # include itemcf
    for mid, s in itemcf.items():
        merged[mid] = merged.get(mid, 0) + alpha * s
    for mid, s in normalized_content.items():
        merged[mid] = merged.get(mid, 0) + (1 - alpha) * s

    # sort and return top_n
    sorted_m = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_m

def movie_info(movie_id):
    row = movies[movies['movieId'] == movie_id]
    if row.empty:
        return {}
    r = row.iloc[0].to_dict()
    return {'movieId': int(r['movieId']), 'title': r['title'], 'genres': r.get('genres',''), 'overview': r.get('overview','')}

if __name__ == "__main__":
    # quick smoke test
    sample_movie = movies['movieId'].iloc[0]
    print("Sample movie:", movie_info(sample_movie))
    print("Top item-CF neighbors:", get_top_n_similar_items_itemcf(sample_movie, 5))
    # if user exists
    if len(ui_matrix.index) > 0:
        sample_user = ui_matrix.index[0]
        print("Recommendations for user", sample_user, recommend_for_user_hybrid(sample_user, 10))
