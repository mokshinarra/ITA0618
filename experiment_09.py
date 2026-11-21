"""Experiment 9: Linear Vs Polynomial"""

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

def experiment_09_linear_vs_polynomial():
    print("\nEXPERIMENT 9: LINEAR VS POLYNOMIAL REGRESSION")
    X = np.arange(1, 11).reshape(-1, 1).astype(float)
    y = np.array([2, 5, 10, 17, 26, 37, 50, 65, 82, 101], dtype=float)
    linear = LinearRegression().fit(X, y)
    polynomial = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    polynomial.fit(X, y)
    pred_linear = linear.predict(X)
    pred_poly = polynomial.predict(X)
    print("Linear Regression R2:", r2_score(y, pred_linear))
    print("Polynomial Regression R2:", r2_score(y, pred_poly))
    plt.scatter(X, y, label="Actual")
    plt.plot(X, pred_linear, label="Linear")
    plt.plot(X, pred_poly, label="Polynomial (degree 2)")
    plt.xlabel("X"); plt.ylabel("Y"); plt.title("Linear vs Polynomial Regression")
    plt.legend(); plt.show()

if __name__ == "__main__":
    experiment_09_linear_vs_polynomial()
