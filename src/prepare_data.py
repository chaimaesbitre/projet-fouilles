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
    "keywords",
    "vote_average",
  ]

  df = df[cols_to_keep].copy()

  # Supprimer les lignes avec vote_average manquant
  df = df.dropna(subset=["vote_average"])

  # Remplir les NaN numériques par la médiane
  numerique_col = ["budget", "runtime", "release_year"]
  for col in numerique_col:
      if col in df.columns:
          df[col] = df[col].fillna(df[col].median())
  # Remplir les NaN texte par "Unknown"
  text_cols = ["genres", "cast", "director", "production_companies", "keywords"]
  for col in text_cols:
      df[col] = df[col].fillna("Unknown")

  return df