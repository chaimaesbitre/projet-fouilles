from src.prepare_data import load_data, prepare_base_dataframe
from src.encoding import encode_all
from src.modeling import compare_models
from src.visualization import (
    plot_model_comparison, plot_predictions, plot_error_distribution
)

def main():
    df = load_data("data/tmdb_movies_data.csv")
    df_base = prepare_base_dataframe(df)
    df_encoded = encode_all(df_base)

    results, (y_test, y_pred_lin, y_pred_rf) = compare_models(df_encoded)
    print("===== Résultats (MSE / R²) =====")
    print(f"Baseline constante (moyenne du train) : {results['baseline_value']:.3f}")
    print(f"Baseline MSE : {results['baseline_mse']:.4f}")
    print(f"Régression linéaire - MSE: {results['linear_mse']:.4f}, R²: {results['linear_r2']:.4f}")
    print(f"Random Forest      - MSE: {results['rf_mse']:.4f}, R²: {results['rf_r2']:.4f}")

    print("\n===== % de bonnes prédictions (tolérance ±0.5) =====")
    print(f"Baseline constante : {results['baseline_acc'] * 100:.2f}%")
    print(f"Régression linéaire: {results['linear_acc'] * 100:.2f}%")
    print(f"Random Forest      : {results['rf_acc'] * 100:.2f}%")

    print("===== Résultats =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Graphiques
    plot_model_comparison(results)
    plot_predictions(y_test, y_pred_lin, "Régression linéaire : Prédictions")
    plot_predictions(y_test, y_pred_rf, "Random Forest : Prédictions")
    plot_error_distribution(y_test, y_pred_lin, "Erreurs (Régression Linéaire)")
    plot_error_distribution(y_test, y_pred_rf, "Erreurs (Random Forest)")

if __name__ == "__main__":
    main()
