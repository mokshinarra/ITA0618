"""Experiment 18: Perceptron Iris"""

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

def experiment_18_perceptron_iris():
    print("\nEXPERIMENT 18: PERCEPTRON-BASED IRIS CLASSIFICATION")
    iris = load_iris()
    # Perceptron is used here for a two-class demonstration.
    mask = iris.target < 2
    X, y = iris.data[mask], iris.target[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), Perceptron(max_iter=2000, random_state=42))
    model.fit(X_train, y_train)
    show_classification_results("Perceptron (Setosa vs Versicolor)",
                                y_test, model.predict(X_test))

if __name__ == "__main__":
    experiment_18_perceptron_iris()
