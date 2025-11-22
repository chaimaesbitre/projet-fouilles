import pandas as pd


def load_data(path):
  return pd.read_csv(path)

def prepare_base_dataframe(df):
  # Colonnes à garder
  cols_to_keep = [
    "budget",
    "runtime",
    "genres",
    "cast",
    "director",
    "production_companies",
    "release_year",
    "vote_average",
  ]

  df_clean = df[cols_to_keep].copy()

  # Supprimer les lignes avec vote_average manquant
  df_clean = df_clean.dropna(subset=["vote_average"])

  # Remplir les NaN numériques par la médiane
  numerique_col = ["budget", "runtime", "release_year"]
  for col in numerique_col:
      if col in df_clean.columns:
          df_clean[col] = df_clean[col].fillna(df_clean[col].median())

  # Remplir les NaN texte par "Unknown"
  text_cols = ["genres", "cast", "director", "production_companies"]
  for col in text_cols:
    if col in df_clean.columns:
      df_clean[col] = df_clean[col].fillna("Unknown")
  df_clean = df_clean.reset_index(drop=True)

  return df_clean
