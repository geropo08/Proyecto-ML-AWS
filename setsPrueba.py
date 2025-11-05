# ==========================================
# SISTEMA DE RECOMENDACIÓN - LEAVE ONE OUT
# ==========================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

# =====================
# 1. CARGA DE DATOS
# =====================
linksTable = pd.read_csv("ml-latest-small/links.csv")
moviesTable = pd.read_csv("ml-latest-small/movies.csv")
ratingsTable = pd.read_csv("ml-latest-small/ratings.csv")
tagsTable = pd.read_csv("ml-latest-small/tags.csv")

# =====================
# 2. DIVISIÓN LEAVE-ONE-OUT
# =====================
train_rows = []
test_rows = []

for user_id, group in ratingsTable.groupby('userId'):
    if len(group) > 1:
        test_sample = group.sample(1, random_state=42)   # deja 1 fuera
        train_sample = group.drop(test_sample.index)
        test_rows.append(test_sample)
        train_rows.append(train_sample)
    else:
        train_rows.append(group)

train_df = pd.concat(train_rows)
test_df = pd.concat(test_rows)

# =====================
# 3. MATRIZ USUARIO–PELÍCULA
# =====================
user_movie_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# =====================
# 4. SIMILITUD ENTRE USUARIOS
# =====================
similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(similarity, 
                             index=user_movie_matrix.index, 
                             columns=user_movie_matrix.index)

# =====================
# 5. FUNCIÓN DE RECOMENDACIÓN
# =====================
def recomendar_peliculas(user_id, user_movie_matrix, similarity_df, top_n=5):
    sim_scores = similarity_df[user_id]
    user_ratings = user_movie_matrix.loc[user_id]
    peliculas_vistas = user_ratings[user_ratings > 0].index

    weighted_ratings = similarity_df[user_id].values @ user_movie_matrix.values
    sim_sums = similarity_df[user_id].values.sum()
    predicted_ratings = weighted_ratings / (sim_sums + 1e-9)

    pred_series = pd.Series(predicted_ratings, index=user_movie_matrix.columns)
    pred_series = pred_series.drop(peliculas_vistas, errors='ignore')

    top_movies = pred_series.sort_values(ascending=False).head(top_n)
    return top_movies.index

# =====================
# 6. EVALUACIÓN LEAVE-ONE-OUT
# =====================
hits = 0
total = 0
N = 5  # número de recomendaciones

for user_id in test_df['userId'].unique():
    test_movies = test_df[test_df['userId'] == user_id]['movieId'].values
    if user_id not in similarity_df.index:
        continue
    recs = recomendar_peliculas(user_id, user_movie_matrix, similarity_df, top_n=N)
    if len(test_movies) > 0:
        total += 1
        if np.intersect1d(recs, test_movies).size > 0:
            hits += 1

precision_leave_one_out = hits / total
print(f"Precisión Leave-One-Out (Top-{N}): {precision_leave_one_out:.3f}")

