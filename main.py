from load import load_data

X, Y, numeric_indices, overview_index, genres_index = load_data("data/tmdb_movies_data.csv")

print(len(X), "films chargés")
print("Exemple X[0] :", X[0])
print("Y[0] (vote_average) :", Y[0])