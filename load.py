from math import nan
import csv

def col_index(header, col_name):
  return header.index(col_name)

def to_float_or_nan(s):
    if s is None or s == "":
        return nan
    try:
        return float(s)
    except ValueError:
        return nan

def load_data(file_path):
  # informations des films
  X = []
  # note moyenne
  Y = []
  with open(file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)

    # indices des colonnes
    indice_popularite = col_index(header, "popularity")
    indice_budget = col_index(header, "budget")
    indice_revenue = col_index(header, "revenue")
    indice_runtime = col_index(header, "runtime")
    indice_vote_count = col_index(header, "vote_count")
    indice_release_year = col_index(header, "release_year")
    indice_overview = col_index(header, "overview")
    indice_genres = col_index(header, "genres")
    indice_vote_avg = col_index(header, "vote_average")

    for data_col in reader:
      # On ignorer les lignes vides
      if not data_col or len(data_col) <= indice_vote_avg:
        continue

      # note moyenne
      vote_avg = to_float_or_nan(data_col[indice_vote_avg])
      if vote_avg is nan:
        # si pas de note, on peut choisir de skip la ligne
        continue
      
      # features numériques
      popularite = to_float_or_nan(data_col[indice_popularite])
      budget = to_float_or_nan(data_col[indice_budget])
      revenue = to_float_or_nan(data_col[indice_revenue])
      runtime = to_float_or_nan(data_col[indice_runtime])
      vote_count = to_float_or_nan(data_col[indice_vote_count])
      release_year = to_float_or_nan(data_col[indice_release_year])

      # features textuelles
      overview = data_col[indice_overview] if indice_overview < len(data_col) else ""
      genres = data_col[indice_genres] if indice_genres < len(data_col) else ""

      x_row = [
        popularite,   
        budget,       
        revenue,      
        runtime,      
        vote_count,   
        release_year, 
        overview,    
        genres,       
      ]
      X.append(x_row)
      Y.append(vote_avg)

    # indices des colonnes numériques dans x_row
    numeric_indices = [0, 1, 2, 3, 4, 5]
    overview_index = 6
    genres_index = 7
    return X, Y, numeric_indices, overview_index, genres_index