import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

np.random.seed(42)

def recomendar_peliculas(ratingsTable,similarity_df, usuario_id, n_vecinos=5, n_recomendaciones=10):
    # Verificar que el usuario existe
    if usuario_id not in similarity_df.index:
        raise ValueError(f"El usuario {usuario_id} no existe en la matriz de similitud.")
    
    # Obtener usuarios más similares (vecinos)
    vecinos = (
        similarity_df[usuario_id]
        .drop(index=usuario_id)
        .sort_values(ascending=False)
        .head(n_vecinos)
    )
    
    
    # Ratings de los vecinos
    ratings_vecinos = ratingsTable[ratingsTable['userId'].isin(vecinos.index)].copy()
    
    # Agregar columna de similitud
    ratings_vecinos['similarity'] = ratings_vecinos['userId'].map(vecinos)
    
    # Calcular rating ponderado por similitud
    ratings_vecinos['weighted_rating'] = ratings_vecinos['rating'] * ratings_vecinos['similarity']
    
    # Promedio ponderado de rating por película
    recomendacion_scores = (
        ratings_vecinos
        .groupby('movieId', as_index=True)
        .agg(weighted_mean=('weighted_rating', lambda x: np.sum(x) / np.sum(ratings_vecinos.loc[x.index, 'similarity'])))
        ['weighted_mean']
        .sort_values(ascending=False)
        .rename('score')
    )
    
    # Eliminar películas ya vistas por el usuario objetivo
    peliculas_vistas = ratingsTable[ratingsTable['userId'] == usuario_id]['movieId']
    recomendacion_scores = recomendacion_scores[~recomendacion_scores.index.isin(peliculas_vistas)]
    
    # Combinar con títulos de películas
    recomendaciones = (
        pd.DataFrame(recomendacion_scores.head(n_recomendaciones))
        .reset_index()
        .merge(moviesTable, on='movieId')[['movieId', 'title', 'score']]
    )
    recomendaciones.columns = ['movieId', 'title', 'score']
    
    return recomendaciones, vecinos

def evaluar_modelo_leave_one_out(ratingsTable, n_vecinos=5, k_recs=15):
    usuarios = ratingsTable['userId'].unique()
    #usuarios = np.random.choice(ratingsTable['userId'].unique(), size=200, replace=False)
    resultados = []
    
    for usuario in usuarios:
        # Seleccionar películas vistas por el usuario
        peliculas_usuario = ratingsTable[
            (ratingsTable['userId'] == usuario) & (ratingsTable['rating'] >= 4.5)]['movieId'].values
        
        if len(peliculas_usuario) <= 1:
            continue  # este usuario no tiene películas "buenas" para probar
        
        # Dejar una película fuera para test
        pelicula_test = np.random.choice(peliculas_usuario)
        
        # Train = resto de películas
        train_ratings = ratingsTable[
            ~((ratingsTable['userId'] == usuario) & (ratingsTable['movieId'] == pelicula_test))
        ]
        
        # Recalcular matriz usuario-película (solo train)
        user_movie_matrix = train_ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        similarity = cosine_similarity(user_movie_matrix)
        sim_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
        
        # Recomendaciones
        try:
            recs = recomendar_peliculas(train_ratings, sim_df, usuario_id=usuario, n_vecinos=n_vecinos, n_recomendaciones=k_recs)[0]
        except:
            continue
        
        # Evaluar: ¿la película test está en las recomendadas?
        acierto = int(pelicula_test in recs['movieId'].values)
        resultados.append(acierto)
    
    precision = np.mean(resultados)
    print(f"Precision@{k_recs}: {precision:.3f}")
    return precision


def evaluar_modelo_leave_many_out(ratingsTable, n_vecinos=5, k_recs=15, n_test=3):

    usuarios = ratingsTable['userId'].unique()
    resultados = []

    for usuario in usuarios:
        # Seleccionar películas "buenas" del usuario
        peliculas_usuario = ratingsTable[
            (ratingsTable['userId'] == usuario) & (ratingsTable['rating'] >= 4.5)
        ]['movieId'].values
        
        if len(peliculas_usuario) <= n_test:
            continue  # usuario no tiene suficientes películas "buenas" para test
        
        # Dejar n_test películas fuera para test
        peliculas_test = np.random.choice(peliculas_usuario, size=n_test, replace=False)
        
        # Train = resto de películas
        train_ratings = ratingsTable[
            ~((ratingsTable['userId'] == usuario) & ratingsTable['movieId'].isin(peliculas_test))
        ]
        
        # Recalcular matriz usuario-película (solo train)
        user_movie_matrix = train_ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        similarity = cosine_similarity(user_movie_matrix)
        sim_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
        
        # Recomendaciones
        try:
            recs = recomendar_peliculas(train_ratings, sim_df, usuario_id=usuario, n_vecinos=n_vecinos, n_recomendaciones=k_recs)[0]
        except:
            continue
        
        # Evaluar: ¿alguna de las películas test está en las recomendadas?
        acierto = int(any(pelicula in recs['movieId'].values for pelicula in peliculas_test))
        resultados.append(acierto)
    
    precision = np.mean(resultados)
    print(f"Precision@{k_recs} (Leave-{n_test}-Out): {precision:.3f}")
    return precision

def evaluar_modelo_leave_one_out_msqe(ratingsTable, n_vecinos=5):
    usuarios = ratingsTable['userId'].unique()
    errores = []

    for usuario in usuarios:
        # Películas que el usuario calificó
        peliculas_usuario = ratingsTable[ratingsTable['userId'] == usuario]['movieId'].values
        
        if len(peliculas_usuario) <= 1:
            continue  # no se puede hacer leave-one-out con solo 1
        
        # Dejar una película fuera
        pelicula_test = np.random.choice(peliculas_usuario)
        rating_real = ratingsTable[
            (ratingsTable['userId'] == usuario) & (ratingsTable['movieId'] == pelicula_test)
        ]['rating'].values[0]

        # Conjunto de entrenamiento
        train_ratings = ratingsTable[
            ~((ratingsTable['userId'] == usuario) & (ratingsTable['movieId'] == pelicula_test))
        ]

        # Matriz usuario-película y similitudes
        user_movie_matrix = train_ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        similarity = cosine_similarity(user_movie_matrix)
        sim_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
        
        # Obtener los vecinos más similares
        if usuario not in sim_df.index:
            continue
        vecinos = (
            sim_df[usuario]
            .drop(index=usuario)
            .sort_values(ascending=False)
            .head(n_vecinos)
        )
        
        # Buscar ratings de los vecinos sobre la película de test
        ratings_vecinos = ratingsTable[
            (ratingsTable['userId'].isin(vecinos.index)) &
            (ratingsTable['movieId'] == pelicula_test)
        ].copy()
        
        if len(ratings_vecinos) == 0:
            continue  # ningún vecino calificó esa película
        
        # Calcular predicción ponderada
        ratings_vecinos['similarity'] = ratings_vecinos['userId'].map(vecinos)
        prediccion = np.sum(ratings_vecinos['rating'] * ratings_vecinos['similarity']) / np.sum(ratings_vecinos['similarity'])
        
        # Error cuadrático
        error = (rating_real - prediccion) ** 2
        errores.append(error)
    
    if len(errores) == 0:
        print("No se pudieron calcular errores (quizás faltan datos de vecinos).")
        return None
    
    msqe = np.mean(errores)
    print(f"MSQE (Mean Squared Error): {msqe:.4f}")
    return msqe

def evaluar_modelo_leave_one_out_msqe_pearson(ratingsTable, n_vecinos=5, sample_size=None):
    # Submuestreo opcional para acelerar
    usuarios = ratingsTable['userId'].unique()
    if sample_size is not None and sample_size < len(usuarios):
        usuarios = np.random.choice(usuarios, size=sample_size, replace=False)

    # Crear matriz usuario-película (sin rellenar con ceros)
    user_movie_matrix = ratingsTable.pivot_table(
        index='userId', columns='movieId', values='rating'
    )

    # Calcular la similitud Pearson una sola vez
    sim_df = user_movie_matrix.T.corr(method='pearson')

    errores = []

    for usuario in usuarios:
        peliculas_usuario = ratingsTable[ratingsTable['userId'] == usuario]['movieId'].values
        if len(peliculas_usuario) <= 1:
            continue

        # Dejar una película fuera
        pelicula_test = np.random.choice(peliculas_usuario)
        rating_real = ratingsTable[
            (ratingsTable['userId'] == usuario) & (ratingsTable['movieId'] == pelicula_test)
        ]['rating'].values[0]

        # Vecinos del usuario (ya precomputados)
        if usuario not in sim_df.columns:
            continue

        vecinos = (
            sim_df[usuario]
            .drop(index=usuario)
            .sort_values(ascending=False)
            .head(n_vecinos)
        )

        # Buscar ratings de los vecinos sobre la película de test
        ratings_vecinos = ratingsTable[
            (ratingsTable['userId'].isin(vecinos.index)) &
            (ratingsTable['movieId'] == pelicula_test)
        ].copy()

        if len(ratings_vecinos) == 0:
            continue

        # Calcular predicción ponderada
        ratings_vecinos['similarity'] = ratings_vecinos['userId'].map(vecinos)
        numerador = np.sum(ratings_vecinos['rating'] * ratings_vecinos['similarity'])
        denominador = np.sum(np.abs(ratings_vecinos['similarity']))  # abs por seguridad
        prediccion = numerador / denominador if denominador != 0 else np.nan

        if np.isnan(prediccion):
            continue

        # Error cuadrático
        error = (rating_real - prediccion) ** 2
        errores.append(error)

    if len(errores) == 0:
        print("No se pudieron calcular errores (quizás faltan datos de vecinos).")
        return None

    msqe = np.mean(errores)
    rmse = np.sqrt(msqe)
    print(f"MSQE (Mean Squared Error): {msqe:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

    return msqe, rmse

def evaluar_modelo_leave_one_out_msqe_cosine(ratingsTable, n_vecinos=5):
    usuarios = ratingsTable['userId'].unique()
    errores = []

    for usuario in usuarios:
        # Películas que calificó el usuario
        peliculas_usuario = ratingsTable[ratingsTable['userId'] == usuario]['movieId'].values
        
        # Si solo calificó una, no se puede hacer leave-one-out
        if len(peliculas_usuario) <= 1:
            continue

        # Elegir una película de test
        pelicula_test = np.random.choice(peliculas_usuario)
        rating_real = ratingsTable[
            (ratingsTable['userId'] == usuario) & (ratingsTable['movieId'] == pelicula_test)
        ]['rating'].values[0]

        # Conjunto de entrenamiento (sin la película de test)
        train_ratings = ratingsTable[
            ~((ratingsTable['userId'] == usuario) & (ratingsTable['movieId'] == pelicula_test))
        ]

        # Crear matriz usuario-película (sin fuga)
        user_movie_matrix = train_ratings.pivot_table(index='userId', columns='movieId', values='rating')

        # Rellenar NaN con el promedio del usuario (mejor que 0)
        user_means = user_movie_matrix.mean(axis=1)
        user_movie_filled = user_movie_matrix.apply(lambda row: row.fillna(user_means[row.name]), axis=1)

        # Calcular similitud coseno entre usuarios
        similarity = cosine_similarity(user_movie_filled)
        sim_df = pd.DataFrame(similarity, index=user_movie_filled.index, columns=user_movie_filled.index)

        # Verificar que el usuario esté en la matriz
        if usuario not in sim_df.index:
            continue

        # Vecinos más similares
        vecinos = (
            sim_df.loc[usuario]
            .drop(index=usuario)
            .sort_values(ascending=False)
            .head(n_vecinos)
        )

        # Buscar calificaciones de vecinos sobre la película de test
        ratings_vecinos = ratingsTable[
            (ratingsTable['userId'].isin(vecinos.index)) &
            (ratingsTable['movieId'] == pelicula_test)
        ].copy()

        if len(ratings_vecinos) == 0:
            continue  # ningún vecino calificó esa película

        # Agregar las similitudes de los vecinos
        ratings_vecinos['similarity'] = ratings_vecinos['userId'].map(vecinos)

        # Predicción ponderada
        prediccion = np.sum(
            ratings_vecinos['rating'] * ratings_vecinos['similarity']
        ) / np.sum(ratings_vecinos['similarity'])

        # Error cuadrático
        error = (rating_real - prediccion) ** 2
        errores.append(error)

    if len(errores) == 0:
        print("No se pudieron calcular errores (quizás faltan vecinos con esa película).")
        return None

    msqe = np.mean(errores)
    print(f"MSQE (Mean Squared Error): {msqe:.4f}")
    return msqe
# Read CSV file
linksTable = pd.read_csv("ml-latest-small/links.csv")
moviesTable = pd.read_csv("ml-latest-small/movies.csv")
ratingsTable = pd.read_csv("ml-latest-small/ratings.csv")
tagsTable = pd.read_csv("ml-latest-small/tags.csv")

# Crear matriz usuario–película
user_movie_matrix  = ratingsTable.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Calcular similitud entre usuarios
similarity = cosine_similarity(user_movie_matrix)


similarity_df = pd.DataFrame(similarity, index=user_movie_matrix .index, columns=user_movie_matrix.index)


#Ver matriz de similaridad de 20x20
plt.figure(figsize=(10,8))
sns.heatmap(similarity_df.iloc[:20, :20], cmap='coolwarm', annot=True)
plt.title("Similitud entre los primeros 20 usuarios")
plt.show()



recomend = recomendar_peliculas(ratingsTable, similarity_df, usuario_id=17, n_vecinos=5, n_recomendaciones=10)
top10=recomend[0]
vecinosUsr=recomend[1]
print(top10)
print(vecinosUsr)

# Ejecutar evaluación
#precisionLOO = evaluar_modelo_leave_one_out(ratingsTable, n_vecinos=5, k_recs=10)

#precisionLMO = evaluar_modelo_leave_many_out(ratingsTable, n_vecinos=5, k_recs=15, n_test=4)
#print(precision)

#msqeLMO = evaluar_modelo_leave_one_out_msqe(ratingsTable=ratingsTable, n_vecinos=3)


#Utilizar similitud de pearson:
#similarity_df_pearson = user_movie_matrix.T.corr(method='pearson')

#msqeLMO_pearson = evaluar_modelo_leave_one_out_msqe_pearson(ratingsTable=ratingsTable, n_vecinos=3)

msqeLMO_pearson = evaluar_modelo_leave_one_out_msqe_cosine(ratingsTable=ratingsTable, n_vecinos=3)

sim_df = similarity_df.copy()

# Reemplazar la diagonal (similitud consigo mismo) por NaN
np.fill_diagonal(sim_df.values, np.nan)

# Media de similitud por usuario (por fila)
media_por_usuario = sim_df.mean(axis=1).round(2)

# Mediana de similitud por usuario (por fila)
mediana_por_usuario = sim_df.median(axis=1).round(2)

# Maxima de similitud por usuario (por fila)
max_por_usuario = sim_df.max(axis=1).round(2)

# Estadísticas de las máximas
media_max = max_por_usuario.mean().round(2)
mediana_max = max_por_usuario.median().round(2)
max_max = max_por_usuario.max().round(2)
cuantiles = max_por_usuario.quantile([0.25, 0.5, 0.75, 1.0]).round(2)

# Combinar en un solo DataFrame si quieres
resumen_similitud = pd.DataFrame({
    'media': media_por_usuario,
    'mediana': mediana_por_usuario,
    'maximo': max_por_usuario
})



print(resumen_similitud)
print(media_max)
print(mediana_max)
print(max_max)
print(cuantiles)


# Crear boxplot
plt.figure(figsize=(8,5))
plt.boxplot(max_por_usuario, vert=True, patch_artist=True, 
            boxprops=dict(facecolor='skyblue', color='blue'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'),
            flierprops=dict(marker='o', color='orange', alpha=0.5))

plt.title('Distribución de la máxima similitud por usuario')
plt.ylabel('Similitud')
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


