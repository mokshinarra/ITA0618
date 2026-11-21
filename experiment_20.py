"""Experiment 20: Future Sales"""

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

def experiment_20_future_sales():
    print("\nEXPERIMENT 20: FUTURE SALES PREDICTION")
    rng = np.random.default_rng(20)
    months = np.arange(1, 37).reshape(-1, 1)
    sales = 1000 + 75 * months.ravel() + rng.normal(0, 100, len(months))
    model = LinearRegression().fit(months, sales)
    future_months = np.arange(37, 43).reshape(-1, 1)
    future_sales = model.predict(future_months)
    print("Slope:", round(model.coef_[0], 2))
    print("Intercept:", round(model.intercept_, 2))
    for month, value in zip(future_months.ravel(), future_sales):
        print(f"Month {month}: Predicted sales = {value:.2f}")
    plt.scatter(months, sales, label="Past sales")
    plt.plot(months, model.predict(months), label="Fitted trend")
    plt.plot(future_months, future_sales, "o--", label="Future prediction")
    plt.xlabel("Month"); plt.ylabel("Sales")
    plt.title("Future Sales Prediction")
    plt.legend(); plt.show()

if __name__ == "__main__":
    experiment_20_future_sales()
