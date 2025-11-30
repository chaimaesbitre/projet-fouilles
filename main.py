from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from src.prepare_data import load_data, prepare_base_dataframe
from src.encoding import encode_all
from src.modeling import (
    baseline_metrics_from_y,
    train_linear_regression,
    train_random_forest,
    evaluate,
)
from src.visualization import (
    plot_model_comparison,
    plot_predictions,
    plot_error_distribution,
    plot_accuracy_comparison,
)


def main():
  # charger et préparer les données
  df = load_data("data/tmdb_movies_data.csv")
  df_base = prepare_base_dataframe(df)

  # séparation features/target 
  X = df_base.drop(columns=["vote_average"])
  y = df_base["vote_average"]

  # séparation train/test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  X_train_enc, X_test_enc = encode_all(X_train, X_test)

  # baseline + modèles
  baseline = baseline_metrics_from_y(y_train, y_test)

  lineaire_model = train_linear_regression(X_train_enc, y_train)
  lineaire = evaluate(lineaire_model, X_test_enc, y_test)

  rf_model = train_random_forest(X_train_enc, y_train)
  rf = evaluate(rf_model, X_test_enc, y_test)

  # résultats pour visualisation
  results = {
    "baseline": baseline,
    "linear_regression": lineaire,
    "random_forest": rf,
    "y_test": y_test,
    "y_pred_linear": lineaire["pred"],
    "y_pred_rf": rf["pred"],
  }

  # validation croisée sur le train encodé 
  print("\n===== VALIDATION CROISÉE SUR RANDOM FOREST =====")

  rf_cv = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
  )

  cv_scores = cross_val_score(
    rf_cv,
    X_train_enc,
    y_train,
    cv=5, # 5 entrainements
    scoring="neg_mean_squared_error",
    n_jobs=-1,
  )
  rmse_scores = np.sqrt(-cv_scores)
  print(f"RMSE moyen (CV 5-fold) : {rmse_scores.mean():.4f}") 
  print(f"RMSE écart-type : {rmse_scores.std():.4f}")

  # affichage des métriques principales
  print("\n===== Baseline (prédiction constante = moyenne du train) =====")
  print(f"Valeur constante prédite : {baseline['value']:.3f}")
  print(f"MSE : {baseline['mse']:.4f}")
  print(f"R²  : {baseline['r2']:.4f}")
  print(f"Accuracy ±0.5 : {baseline['acc_±0.5']:.2f}%")
  print(f"Accuracy ±1   : {baseline['acc_±1']:.2f}%")
  print(f"Accuracy ±2   : {baseline['acc_±2']:.2f}%")

  print("\n===== Régression linéaire =====")
  print(f"MSE : {lineaire['mse']:.4f}")
  print(f"R²  : {lineaire['r2']:.4f}")
  print(f"Accuracy ±0.5 : {lineaire['acc_±0.5']:.2f}%")
  print(f"Accuracy ±1   : {lineaire['acc_±1']:.2f}%")
  print(f"Accuracy ±2   : {lineaire['acc_±2']:.2f}%")

  print("\n===== Random Forest =====")
  print(f"MSE : {rf['mse']:.4f}")
  print(f"R²  : {rf['r2']:.4f}")
  print(f"Accuracy ±0.5 : {rf['acc_±0.5']:.2f}%")
  print(f"Accuracy ±1   : {rf['acc_±1']:.2f}%")
  print(f"Accuracy ±2   : {rf['acc_±2']:.2f}%")

  # graphiques
  plot_model_comparison(results) # MSE/R^2
  plot_accuracy_comparison(results) # accuracy
  plot_predictions(y_test, results["y_pred_linear"],"Régression linéaire : Prédictions vs Réalité")
  plot_predictions(y_test, results["y_pred_rf"],"Random Forest : Prédictions vs Réalité")
  plot_error_distribution(y_test, results["y_pred_linear"],"Erreurs (Régression Linéaire)")
  plot_error_distribution(y_test, results["y_pred_rf"],"Erreurs (Random Forest)")

if __name__ == "__main__":
  main()

