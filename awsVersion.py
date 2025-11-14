#First Imports
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
import boto3
import pickle
import io

from sklearn.feature_extraction.text import TfidfVectorizer

s3 = boto3.client('s3')
bucket_name = "movie-training-data-412914223210-us-east-1"

# Download file content into memory
objLink = s3.get_object(Bucket=bucket_name, Key="links.csv")
csv_dataLink = objLink['Body'].read().decode('utf-8')
objMovie = s3.get_object(Bucket=bucket_name, Key="movies.csv")
csv_dataMovie = objMovie['Body'].read().decode('utf-8')
objRating = s3.get_object(Bucket=bucket_name, Key="ratings.csv")
csv_dataRating = objRating['Body'].read().decode('utf-8')
objTag = s3.get_object(Bucket=bucket_name, Key="tags.csv")
csv_dataTag = objTag['Body'].read().decode('utf-8')
objMetadata = s3.get_object(Bucket=bucket_name, Key="movies_metadata.csv")
csv_dataMetadata = objMetadata['Body'].read().decode('utf-8')

# Convert to DataFrame
linksTable = pd.read_csv(io.StringIO(csv_dataLink))
moviesTable = pd.read_csv(io.StringIO(csv_dataMovie))
ratingsTable = pd.read_csv(io.StringIO(csv_dataRating))
tagsTable = pd.read_csv(io.StringIO(csv_dataTag))
moviesMetadata = pd.read_csv(io.StringIO(csv_dataMetadata))


# Split each user's ratings into train (to generate recommendations) and test (to evaluate)
def train_test_split_ratings(ratings_df, test_size=0.2):
    train_list = []
    test_list = []
    for user_id, group in ratings_df.groupby('userId'):
        if len(group) < 5:
            continue  # skip users with too few ratings
        train, test = train_test_split(group, test_size=test_size, random_state=42)
        train_list.append(train)
        test_list.append(test)
    return pd.concat(train_list), pd.concat(test_list)

def cf_based_scores(ratings_df, similarity_df, user_id, n_vecinos=5):
    if user_id not in similarity_df.index:
        return pd.DataFrame(columns=['movieId', 'cf_score'])
    
    vecinos = similarity_df[user_id].drop(index=user_id).sort_values(ascending=False).head(n_vecinos)
    ratings_vecinos = ratings_df[ratings_df['userId'].isin(vecinos.index)].copy()
    ratings_vecinos['similarity'] = ratings_vecinos['userId'].map(vecinos)
    ratings_vecinos['weighted_rating'] = ratings_vecinos['rating'] * ratings_vecinos['similarity']
    
    recomendacion_scores = (
        ratings_vecinos
        .groupby('movieId')
        .agg(cf_score=('weighted_rating', lambda x: np.sum(x) / np.sum(ratings_vecinos.loc[x.index, 'similarity'])))
        .reset_index()
    )
    return recomendacion_scores

def content_based_scores(user_id, ratings_df, movies_df, cosine_sim):
    user_rated = ratings_df[ratings_df['userId'] == user_id]
    liked = user_rated[user_rated['rating'] >= 4]

    if liked.empty:
        return pd.DataFrame(columns=['movieId', 'cb_score'])

    sim_scores = np.zeros(cosine_sim.shape[0])
    for _, row in liked.iterrows():
        match = movies_df[movies_df['movieId'] == row['movieId']]
        if match.empty:
            continue
        movie_idx = match.index[0]
        sim_scores += cosine_sim[movie_idx] * row['rating']

    rated_positions = movies_df[movies_df['movieId'].isin(user_rated['movieId'])].index
    sim_scores[rated_positions] = 0

    return pd.DataFrame({
        'movieId': movies_df['movieId'],
        'cb_score': sim_scores
    })

def hybrid_recommendations(user_id, ratings_df, movies_df, cosine_sim, similarity_df, alpha=0.5, top_n=10):
    # get both score tables
    cb = content_based_scores(user_id, ratings_df, movies_df, cosine_sim)
    cf = cf_based_scores(ratings_df, similarity_df, user_id, n_vecinos=5)
    
    # merge on movieId
    hybrid = pd.merge(cb, cf, on='movieId', how='outer').fillna(0)
    
    # final weighted score
    hybrid['final_score'] = alpha * hybrid['cf_score'] + (1 - alpha) * hybrid['cb_score']
    
    # remove seen movies
    seen = ratings_df[ratings_df['userId'] == user_id]['movieId']
    hybrid = hybrid[~hybrid['movieId'].isin(seen)]
    
    # attach movie titles
    hybrid = hybrid.merge(movies_df[['movieId', 'title']], on='movieId', how='left')
    
    return hybrid.sort_values('final_score', ascending=False).head(top_n)

def hybrid_predicted_ratings(user_id, ratings_df, movies_df, cosine_sim, similarity_df, alpha=0.5):
    """
    Returns predicted ratings for all unseen movies for a given user.
    """
    cb = content_based_scores(user_id, ratings_df, movies_df, cosine_sim)
    cf = cf_based_scores(ratings_df, similarity_df, user_id, n_vecinos=5)

    if cb['cb_score'].max() > 0:
        cb['cb_score'] = (cb['cb_score'] - cb['cb_score'].min()) / (cb['cb_score'].max() - cb['cb_score'].min())
    if cf['cf_score'].max() > 0:
        cf['cf_score'] = (cf['cf_score'] - cf['cf_score'].min()) / (cf['cf_score'].max() - cf['cf_score'].min())
    
    hybrid = pd.merge(cb, cf, on='movieId', how='outer').fillna(0)
    hybrid['final_score'] = alpha * hybrid['cf_score'] + (1 - alpha) * hybrid['cb_score']

    # Clip or normalize predictions to match rating scale
    if not hybrid['final_score'].empty:
        min_score, max_score = hybrid['final_score'].min(), hybrid['final_score'].max()
        if max_score > min_score:  # avoid divide by zero
            hybrid['pred_rating'] = 1 + 4 * (hybrid['final_score'] - min_score) / (max_score - min_score)
        else:
            hybrid['pred_rating'] = 3  # neutral value if constant scores
    else:
        hybrid['pred_rating'] = []

    return hybrid[['movieId', 'pred_rating']]

def rmse_mse_hybrid(train_ratings, test_ratings, movies_df, cosine_sim, similarity_df, alpha=0.5):
    """
    Compute RMSE and MSE across all users in the test set.
    """
    preds = []
    trues = []

    for user_id in test_ratings['userId'].unique():
        hybrid_preds = hybrid_predicted_ratings(user_id, train_ratings, movies_df, cosine_sim, similarity_df, alpha)
        
        # True ratings for the user
        true_user = test_ratings[test_ratings['userId'] == user_id][['movieId', 'rating']]
        
        # Merge predictions with actual ratings
        merged = pd.merge(hybrid_preds, true_user, on='movieId', how='inner')
        if not merged.empty:
            preds.extend(merged['pred_rating'])
            trues.extend(merged['rating'])

    if not preds:
        print("⚠️ No overlapping movies between test set and predictions.")
        return None, None

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)

    print(f"📉 MSE:  {mse:.4f}")
    print(f"📈 RMSE: {rmse:.4f}")
    return mse, rmse

train_ratings, test_ratings = train_test_split_ratings(ratingsTable)

# Create matrix user-movie
user_movie_matrix  = ratingsTable.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Similarity between users
similarity = cosine_similarity(user_movie_matrix)
user_based_similarity = pd.DataFrame(similarity, index=user_movie_matrix .index, columns=user_movie_matrix.index)


moviesMetadata['imdb_id_numeric'] = moviesMetadata['imdb_id'].str.replace('tt', '')
linksTable['imdbId_str'] = linksTable['imdbId'].astype(int).astype(str).str.zfill(7)
# LeftJoin
movies_merged = linksTable.merge(
    moviesMetadata,
    left_on='imdbId_str',
    right_on='imdb_id_numeric',
    how='left'
)
movies_merged_coinc=movies_merged[movies_merged['title'].notna()]
movies_merged_coinc=movies_merged_coinc[['movieId', 'overview']]
# inner Join
movies_merged_description = moviesTable.merge(
    movies_merged_coinc,
    left_on='movieId',
    right_on='movieId',
    how='inner'
)
movies_merged_description=movies_merged_description[movies_merged_description['overview'].notna()]
movies_merged_description['content'] = movies_merged_description['genres'].str.replace('|',' ') + " " + movies_merged_description['overview']
movies_merged_description = movies_merged_description.drop_duplicates(subset='movieId').reset_index(drop=True)


#model = SentenceTransformer('all-MiniLM-L6-v2')
#embeddings = model.encode(movies_merged_description['content'], show_progress_bar=True)
#embeddings_similarity = cosine_similarity(embeddings)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_merged_description['content'])
# cosine_sim[i,j] = similaridad entre pelicula i y pelicula j
cosine_sim_TFIDF = cosine_similarity(tfidf_matrix)
#See matrix 20x20 of similarity

for alpha in [0.2, 0.5, 0.8]:
    print(f"\n--- Hybrid model (alpha={alpha}) ---")
    rmse_mse_hybrid(train_ratings, test_ratings, movies_merged_description, cosine_sim_TFIDF, user_based_similarity, alpha=alpha)



# Save similarity_df (Pandas DataFrame)
buffer = io.BytesIO()
pickle.dump(user_based_similarity, buffer)
buffer.seek(0)
s3.upload_fileobj(buffer, bucket_name, "artifacts/user_based_similarity.pkl")

# Save cosine_sim_embeddings (NumPy array)
buffer = io.BytesIO()
np.save(buffer, cosine_sim_TFIDF)
buffer.seek(0)
s3.upload_fileobj(buffer, bucket_name, "artifacts/cosine_sim_TFIDF.npy")

print("Export ended!")