"""Experiment 10: Em"""

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

def experiment_10_em():
    print("\nEXPERIMENT 10: EXPECTATION-MAXIMIZATION")
    rng = np.random.default_rng(42)
    group1 = rng.normal(loc=[2, 2], scale=0.6, size=(100, 2))
    group2 = rng.normal(loc=[6, 6], scale=0.7, size=(100, 2))
    X = np.vstack((group1, group2))
    model = GaussianMixture(n_components=2, random_state=42)
    model.fit(X)
    labels = model.predict(X)
    print("Cluster means:\n", model.means_)
    print("Number of iterations:", model.n_iter_)
    print("Converged:", model.converged_)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(model.means_[:, 0], model.means_[:, 1], marker="X", s=200)
    plt.title("Gaussian Mixture Model using EM")
    plt.xlabel("Feature 1"); plt.ylabel("Feature 2"); plt.show()

if __name__ == "__main__":
    experiment_10_em()
