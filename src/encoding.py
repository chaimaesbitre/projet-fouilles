import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#**********************Les encodages des colonnes texte pour eviter l'overfitting ******************#

def split_pipe(value):
  if not isinstance(value, str) or value.strip() == "":
    return []
  parts = [p.strip() for p in value.split("|")]
  return [p for p in parts if p]

# ENCODAGE GENRES
# k = 5 genres les plus fréquents
def get_top_genres(df, top_k=5):
  all_genres = []
  for txt in df["genres"]:
    all_genres.extend(split_pipe(txt))
  occ = {}
  for g in all_genres:
    occ[g] = occ.get(g, 0) + 1
  sorted_genres = sorted(occ.items(), key=lambda x: x[1], reverse=True)
  top_genres = [g for g, _ in sorted_genres[:top_k]]
  return top_genres

# ajout des colonnes binaires pour les genres
def add_genre_columns(df, top_genres):
  df = df.copy()
  for g in top_genres:
    col_name = f"genre_{g.replace(' ', '_')}"
    df[col_name] = df["genres"].apply(
        lambda lst: 1 if g in split_pipe(lst) else 0
    )
  return df

# ENCODAGE ACTEURS
# l'acteur principal de chaque film
def get_main_actor_column(df):
  acteurs = []
  for text in df["cast"]:
    lst = split_pipe(text)
    acteurs.append(lst[0] if lst else "Unknown")
  return acteurs

# les 10 acteurs principaux les plus fréquents
def get_top_main_actors(df, top_k=10):
  main_actors = get_main_actor_column(df)
  freq = {}
  for a in main_actors:
    freq[a] = freq.get(a, 0) + 1
  top_acteurs = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
  return [a for a, _ in top_acteurs]

# ajout des colonnes binaires pour les acteurs principaux
def add_actor_columns(df, top_acteurs):
  df = df.copy()
  df["main_actor"] = get_main_actor_column(df)
  for a in top_acteurs:
    df["actor_" + a] = (df["main_actor"] == a).astype(int)
  return df

# ENCODAGE COMPAGNIES DE PRODUCTION
# on garde la compagnie principale
def get_main_company_column(df):
  companies = []
  for text in df["production_companies"]:
    lst = split_pipe(text)
    companies.append(lst[0] if lst else "Unknown")
  return companies

# les 10 compagnies de production les plus fréquentes
def get_top_main_companies(df, top_k=10):
  main_companies = get_main_company_column(df)
  freq = {}
  for c in main_companies:
    freq[c] = freq.get(c, 0) + 1
  top_companies = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
  return [c for c, _ in top_companies]

# ajout des colonnes binaires pour les compagnies principales
def add_company_columns(df, top_companies):
  df = df.copy()
  df["main_company"] = get_main_company_column(df)
  for c in top_companies:
    col_name = f"company_{c}"
    df[col_name] = (df["main_company"] == c).astype(int)
  return df


# ENCODAGE RÉALISATEURS
# on garde le réalisateur principal
def get_main_director_column(df):
  realisateur = []
  for text in df["director"]:
    lst = split_pipe(text)
    realisateur.append(lst[0] if lst else "Unknown")
  return realisateur

# les 10 réalisateurs les plus fréquents
def get_top_main_realisateur(df, top_k=10):
  main_realisateur = get_main_director_column(df)
  freq = {}
  for d in main_realisateur:
    freq[d] = freq.get(d, 0) + 1
  top_realisateur = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
  return [d for d, _ in top_realisateur]

# ajout des colonnes binaires pour les réalisateurs principaux
def add_director_columns(df, top_realisateur):
  df = df.copy()
  df["main_director"] = get_main_director_column(df)
  for d in top_realisateur:
    col_name = f"director_{d}"
    df[col_name] = (df["main_director"] == d).astype(int)
  return df



# TF-IDF sur keywords
# max_features = 1000 : nombre maximum de mots à garder
# min-df = 10 : un mot doit apparaitre dans au moins 10 documents pour être gardé
def encode_keywords_tfidf_train_test(X_train, X_test, max_features=1000, min_df=10):
  if "keywords" not in X_train.columns:
    return X_train, X_test

  train_corpus = X_train["keywords"].fillna("").str.replace("|", " ", regex=False)
  test_corpus  = X_test["keywords"].fillna("").str.replace("|", " ", regex=False)
  vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
  X_train_tfidf = vectorizer.fit_transform(train_corpus)
  X_test_tfidf  = vectorizer.transform(test_corpus)

  feature_names = vectorizer.get_feature_names_out()
  train_tfidf_df = pd.DataFrame(
    X_train_tfidf.toarray(),
    columns=[f"kw_{w}" for w in feature_names],
    index=X_train.index,
  )
  test_tfidf_df = pd.DataFrame(
    X_test_tfidf.toarray(),
    columns=[f"kw_{w}" for w in feature_names],
    index=X_test.index,
  )

  X_train = pd.concat([X_train.drop(columns=["keywords"]), train_tfidf_df], axis=1)
  X_test  = pd.concat([X_test.drop(columns=["keywords"]),  test_tfidf_df], axis=1)
  return X_train, X_test
  
# ENCODAGE COMPLET  
def encode_all(X_train, X_test):
  X_train = X_train.copy()
  X_test = X_test.copy()

  # Genres
  top_genres = get_top_genres(X_train, top_k=5)
  X_train = add_genre_columns(X_train, top_genres)
  X_test  = add_genre_columns(X_test, top_genres)

  # Acteurs principaux
  top_actors = get_top_main_actors(X_train, top_k=10)
  X_train = add_actor_columns(X_train, top_actors)
  X_test  = add_actor_columns(X_test, top_actors)

  # Réalisateurs
  top_realisateur = get_top_main_realisateur(X_train, top_k=10)
  X_train = add_director_columns(X_train, top_realisateur)
  X_test  = add_director_columns(X_test, top_realisateur)

  # Compagnies de production
  top_companies = get_top_main_companies(X_train, top_k=10)
  X_train = add_company_columns(X_train, top_companies)
  X_test  = add_company_columns(X_test, top_companies)

  # TF-IDF sur les keywords
  X_train, X_test = encode_keywords_tfidf_train_test(X_train, X_test)

  # Suppression des colonnes texte
  cols_to_drop = ["genres", "cast", "director", "production_companies", "keywords","main_actor", "main_company", "main_director"]
  for c in cols_to_drop:
    if c in X_train.columns:
      X_train = X_train.drop(columns=[c])
    if c in X_test.columns:
      X_test = X_test.drop(columns=[c])

  return X_train, X_test
