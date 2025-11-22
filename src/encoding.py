from typing import List
#**********************Les encodages des colonnes texte pour eviter l'overfitting ******************#

def split_pipe(value):
  if not isinstance(value, str):
    return []
  parts = [p.strip() for p in value.split("|")]
  return [p for p in parts if p]

# k = 5 pour eviter l'overfitting
# il y a 5 genres les plus fréquents a peu près
def encode_genres(df, top_k=5):
  
  df = df.copy()
  # Construire une liste de listes de genres
  genre_lists = df["genres"].apply(split_pipe).tolist()

  # Compter les occurrences de chaque genre
  genres = [g for ss_list in genre_lists for g in ss_list]
  # Dictionnaire de fréquence des genres
  genre_freq = {}
  for g in genres:
    genre_freq[g] = genre_freq.get(g, 0) + 1

  # Trier par fréquence décroissante et prendre les top_k
  top_genres = [g for g, _ in sorted(genre_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]

  df["genre_count"] = df["genres"].apply(lambda x: len(split_pipe(x)))

  # Colonnes binaires pour les genres les plus fréquents
  for g in top_genres:
    col_name = f"genre_{g.replace(' ', '_')}"
    df[col_name] = df["genres"].apply(lambda x: 1 if g in split_pipe(x) else 0)

  return df


def encode_cast(df):
  df = df.copy()

  # Nombre d'acteurs listés
  df["cast_size"] = df["cast"].apply(lambda x: len(split_pipe(x)))

  # Acteur principal = premier de la liste
  df["main_actor"] = df["cast"].apply(lambda x: split_pipe(x)[0] if split_pipe(x) else "Unknown")

  # Nombre de films où l'acteur principal apparaît
  actor_counts = df["main_actor"].value_counts()
  # Nombre de films où l'acteur principal apparaît
  df["main_actor_movie_count"] = df["main_actor"].map(actor_counts)

  return df


def encode_director(df):
  df = df.copy()
  director_counts = df["director"].value_counts()
  # Nombre de films du réalisateur
  df["director_movie_count"] = df["director"].map(director_counts)
  return df


def encode_production_companies(df):
  df = df.copy()
  # Nombre de companies listées
  df["num_companies"] = df["production_companies"].apply(lambda x: len(split_pipe(x)))
  # Première compagnie listée
  df["main_company"] = df["production_companies"].apply(lambda x: split_pipe(x)[0] if split_pipe(x) else "Unknown")
  company_counts = df["main_company"].value_counts()
  # Nombre de films cette compagnie a produit
  df["main_company_movie_count"] = df["main_company"].map(company_counts)
  return df

# Retourne un dataframe numérique prêt pour l'entraînement sans vote_average
def encode_all(df):
  df = df.copy()
  # Encodages
  df = encode_genres(df, top_k=5)
  df = encode_cast(df)
  df = encode_director(df)
  df = encode_production_companies(df)

  # On retire les colonnes texte originales
  cols_to_drop = ["genres", "cast", "director", "production_companies", "main_actor", "main_company"]
  for c in cols_to_drop:
    if c in df.columns:
      df = df.drop(columns=[c])

  return df