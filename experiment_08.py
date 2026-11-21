"""Experiment 8: Linear Regression"""

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

def experiment_08_linear_regression():
    print("\nEXPERIMENT 8: LINEAR REGRESSION")
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
    y = np.array([3, 5, 7, 9, 11, 13, 15, 17], dtype=float)
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    print("Slope:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("Predicted values:", predictions)
    print("R2 score:", r2_score(y, predictions))
    print("Prediction for X=9:", model.predict([[9]])[0])
    plt.scatter(X, y, label="Actual")
    plt.plot(X, predictions, label="Regression line")
    plt.xlabel("X"); plt.ylabel("Y"); plt.title("Linear Regression")
    plt.legend(); plt.show()

if __name__ == "__main__":
    experiment_08_linear_regression()
