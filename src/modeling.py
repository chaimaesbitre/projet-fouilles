import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def accuracy_with_tolerance(y_true, y_pred, tolerance): # tolerance de 1 cad plus ou moins 1
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  return np.mean(np.abs(y_true - y_pred) <= tolerance)

# train de la régression linéaire
def train_linear_regression(X_train, y_train):
  model = LinearRegression()
  model.fit(X_train, y_train)
  return model

# train de la random forest
def train_random_forest(X_train, y_train, random_state=42):
  model = RandomForestRegressor(
    n_estimators=100, # nombre d'arbres + y en + cst mieux mais + d'arbre + lent
    random_state=random_state,
    n_jobs=-1, # utilise tous les coeurs dispo
  )
  model.fit(X_train, y_train)
  return model


def evaluate(model, X_test, y_test):
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  # pourcentage de la prediction du model
  acc_05 = accuracy_with_tolerance(y_test, y_pred, 0.5) * 100
  acc_1 = accuracy_with_tolerance(y_test, y_pred, 1.0) * 100
  acc_2 = accuracy_with_tolerance(y_test, y_pred, 2.0) * 100

  return {
    "mse": mse,
    "r2": r2,
    "r2_percent": r2 * 100,
    "acc_±0.5": acc_05,
    "acc_±1": acc_1,
    "acc_±2": acc_2,
    "pred": y_pred,
}


def baseline_metrics_from_y(y_train, y_test):
  baseline_value = y_train.mean()
  y_pred_baseline = np.full(len(y_test), baseline_value)

  mse = mean_squared_error(y_test, y_pred_baseline)
  r2 = r2_score(y_test, y_pred_baseline)
  # pourcentage pour la baseline
  acc_05 = accuracy_with_tolerance(y_test, y_pred_baseline, 0.5) * 100 
  acc_1 = accuracy_with_tolerance(y_test, y_pred_baseline, 1.0) * 100 
  acc_2 = accuracy_with_tolerance(y_test, y_pred_baseline, 2.0) * 100 

  return {
    "value": baseline_value,
    "mse": mse,
    "r2": r2,
    "r2_percent": r2 * 100,
    "acc_±0.5": acc_05,
    "acc_±1": acc_1,
    "acc_±2": acc_2,
    "pred": y_pred_baseline,
  }
