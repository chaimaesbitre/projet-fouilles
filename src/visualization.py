import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results):
  models = ["Baseline", "Régression Linéaire", "Random Forest"]    
  mse = [
    results["baseline"]["mse"],
    results["linear_regression"]["mse"],
    results["random_forest"]["mse"],
  ]
  r2 = [
    results["baseline"]["r2"],
    results["linear_regression"]["r2"],
    results["random_forest"]["r2"],
  ]
  plt.figure(figsize=(10, 5))

  # MSE
  plt.subplot(1, 2, 1)
  sns.barplot(x=models, y=mse)
  plt.title("Comparaison des MSE")
  plt.ylabel("MSE")

  # R²
  plt.subplot(1, 2, 2)
  sns.barplot(x=models, y=r2)
  plt.title("Comparaison des R²")
  plt.ylabel("R²")

  plt.tight_layout()
  plt.show()

def plot_accuracy_comparison(results):
  models = ["Baseline", "Régression Linéaire", "Random Forest"]
  acc1 = [
    results["baseline"]["acc_±1"],
    results["linear_regression"]["acc_±1"],
    results["random_forest"]["acc_±1"],
  ]
  plt.figure(figsize=(6, 4))
  sns.barplot(x=models, y=acc1)
  plt.title("Accuracy à ±1 point")
  plt.ylabel("Accuracy (%)")
  plt.xlabel("Modèle")
  plt.ylim(0, 100)
  plt.tight_layout()
  plt.show()

def plot_predictions(y_test, y_pred, title="Prédictions vs Réalité"):
  plt.figure(figsize=(6, 6))
  plt.scatter(y_test, y_pred, alpha=0.5)
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
  plt.xlabel("Vraies valeurs")
  plt.ylabel("Prédictions")
  plt.title(title)
  plt.show()

def plot_error_distribution(y_test, y_pred, title="Distribution des erreurs"):
  errors = y_test - y_pred
  plt.figure(figsize=(8, 4))
  sns.histplot(errors, bins=30, kde=True)
  plt.title(title)
  plt.xlabel("Erreur (y_true - y_pred)")
  plt.show()
