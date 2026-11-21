"""Experiment 16: Compare Classifiers"""

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

def experiment_16_compare_classifiers():
    print("\nEXPERIMENT 16: COMPARISON OF CLASSIFICATION ALGORITHMS")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Logistic Regression": make_pipeline(StandardScaler(),
                                              LogisticRegression(max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, pred)
        print(f"{name}: Accuracy = {results[name]:.4f}")
    plt.bar(results.keys(), results.values())
    plt.ylabel("Accuracy"); plt.ylim(0, 1.05)
    plt.title("Classification Algorithm Comparison")
    plt.xticks(rotation=20); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    experiment_16_compare_classifiers()
