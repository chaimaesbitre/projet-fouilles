import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#**********************Les encodages des colonnes texte pour eviter l'overfitting ******************#

def split_pipe(value):
  if not isinstance(value, str):
    return []
  parts = [p.strip() for p in value.split("|")]
  return [p for p in parts if p]

# k = 5 genres les plus fréquents
def encode_genres(df, top_k=5):
  df = df.copy()
  all_genres = []
  # Extraire tous les genres
  for txt in df["genres"]:
    all_genres.append(split_pipe(txt))

  # On compte les occurrences
  occ = {}
  for g in all_genres:
    occ[g] = occ.get(g, 0) + 1

  # garder les top_k
  sorted_genres = sorted(occ.items(), key=lambda x: x[1], reverse=True)
  top_genres = [g for g, _ in sorted_genres[:top_k]]

  # créer les colonnes
  for g in top_genres:
    df["genre_" + g] = df["genres"].apply(lambda lst: 1 if g in split_pipe(lst) else 0)
    return df


def encode_cast(df):
  df = df.copy()
  df["main_actor"] = df["cast"].apply(lambda x: split_pipe(x)[0] if isinstance(x, str) else "Unknown")
  top_actors = df["main_actor"].value_counts().head(10).index.tolist()

  for actor in top_actors:
    df[f"actor_{actor}"] = (df["main_actor"] == actor).astype(int)

  return df

# encodage du réalisateur
# On crée une colonne pour le réalisateur principal
def encode_cast(df):
  df = df.copy()
  # Liste des acteurs principaux
  main_actors = []
  for text in df["cast"]:
    lst = split_pipe(text)
    # Ajouter l'acteur principal ou "Unknown" si la liste est vide
    main_actors.append(lst[0] if lst else "Unknown")
  df["main_actor"] = main_actors
  # compter les acteurs
  freq = {}
  for a in main_actors:
    freq[a] = freq.get(a, 0) + 1

  # Les 10 acteurs les plus fréquents
  top_actors = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
  top_actors = [a for a, _ in top_actors]
  # Colonnes binaires pour les acteurs principaux
  for a in top_actors:
    df["actor_" + a] = (df["main_actor"] == a).astype(int)
  return df

# encodage du réalisateur
def encode_production_companies(df):
  df = df.copy()
  df["main_company"] = df["production_companies"].apply(lambda x: split_pipe(x)[0] if isinstance(x, str) and split_pipe(x) else "Unknown")
  # Les 10 compagnies de production les plus fréquentes (sans convertir en list)
  top_companies = df["main_company"].value_counts().head(10).index
  # Colonnes binaires pour les compagnies principales
  for c in top_companies:
    df[f"company_{c}"] = (df["main_company"] == c).astype(int)
  return df

def encode_director(df):
  df = df.copy()
  # extraire le réalisateur principal
  df["main_director"] = df["director"].apply(lambda x: split_pipe(x)[0] if isinstance(x, str) and split_pipe(x) else "Unknown")
  # prendre les 10 réalisateurs les plus fréquents
  top_directors = df["main_director"].value_counts().head(10).index
  # one-hot encodage des réalisateurs top 10
  for d in top_directors:
    df[f"director_{d}"] = (df["main_director"] == d).astype(int)
  return df


# TF-IDF sur keywords
# max_features = 1000 : nombre maximum de mots à garder
# min-df = 10 : un mot doit apparaitre dans au moins 10 documents pour être gardé
def encode_keywords_tfidf(df, max_features=1000, min_df=10):
  df = df.copy()
  if "keywords" not in df.columns:
    return df
  corpus = df["keywords"].fillna("").str.replace("|", " ", regex=False)
  vectorizer = TfidfVectorizer(
    max_features=max_features,
    min_df=min_df
  )
  X = vectorizer.fit_transform(corpus)
  # Créer un DataFrame à partir de la matrice TF-IDF
  feature_names = vectorizer.get_feature_names_out()
  tfidf_df = pd.DataFrame(
    X.toarray(),
    columns=[f"kw_{w}" for w in feature_names],
    index=df.index,
  )
  # Concaténer avec le DataFrame original
  df = pd.concat([df.drop(columns=["keywords"]), tfidf_df], axis=1)
  return df

def encode_all(df):
    df = df.copy()
    df = encode_genres(df)
    df = encode_cast(df)
    df = encode_director(df)
    df = encode_production_companies(df)
    # Ajout TF-IDF sur keywords
    df = encode_keywords_tfidf(df)
    
    # Suppression des colonnes texte
    cols_to_drop = ["genres", "cast", "director", "production_companies", "main_actor", "main_company"]
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df
