"""Experiment 13: Car Price"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)

def experiment_13_car_price():
    print("\nEXPERIMENT 13: CAR PRICE PREDICTION")
    rng = np.random.default_rng(42)
    n = 300
    age = rng.integers(0, 15, n)
    km = rng.integers(5000, 180000, n)
    engine = rng.integers(800, 3000, n)
    y = 1800000 - age * 85000 - km * 2.2 + engine * 220 + rng.normal(0, 50000, n)
    X = pd.DataFrame({"Age": age, "Kilometres": km, "EngineCC": engine})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("R2 score:", round(r2_score(y_test, pred), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
    sample = pd.DataFrame({"Age": [5], "Kilometres": [40000], "EngineCC": [1500]})
    print("Estimated car price:", round(model.predict(sample)[0], 2))

if __name__ == "__main__":
    experiment_13_car_price()
