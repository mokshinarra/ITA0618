"""Experiment 17: Mobile Price"""

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

def experiment_17_mobile_price():
    print("\nEXPERIMENT 17: MOBILE PRICE PREDICTION")
    X, y = make_classification(
        n_samples=1000, n_features=8, n_informative=6, n_redundant=0,
        n_classes=4, n_clusters_per_class=1, random_state=17
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    show_classification_results("Mobile price range (synthetic demonstration data)",
                                y_test, model.predict(X_test))

if __name__ == "__main__":
    experiment_17_mobile_price()
