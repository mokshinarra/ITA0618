"""Experiment 14: House Price"""

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

def experiment_14_house_price():
    print("\nEXPERIMENT 14: HOUSE PRICE PREDICTION")
    rng = np.random.default_rng(10)
    n = 300
    area = rng.integers(500, 3500, n)
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 40, n)
    price = area * 3500 + bedrooms * 180000 - age * 12000 + rng.normal(0, 100000, n)
    X = pd.DataFrame({"Area_sqft": area, "Bedrooms": bedrooms, "Age_years": age})
    X_train, X_test, y_train, y_test = train_test_split(
        X, price, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    print("R2 score:", round(r2_score(y_test, pred), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
    sample = pd.DataFrame({"Area_sqft": [1200], "Bedrooms": [3], "Age_years": [5]})
    print("Estimated house price:", round(model.predict(sample)[0], 2))

if __name__ == "__main__":
    experiment_14_house_price()
