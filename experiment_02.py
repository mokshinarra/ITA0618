"""Experiment 2: Candidate Elimination"""

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

def experiment_02_candidate_elimination():
    print("\nEXPERIMENT 2: CANDIDATE-ELIMINATION ALGORITHM")
    data = [
        ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
        ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
        ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
        ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"],
    ]
    n = 6
    S = ["Ø"] * n
    G = [["?"] * n]

    def covers(h, x):
        return all(a == "?" or a == b for a, b in zip(h, x))

    for row in data:
        x, label = row[:-1], row[-1]
        if label == "Yes":
            G = [g for g in G if covers(g, x)]
            for i in range(n):
                if S[i] == "Ø":
                    S[i] = x[i]
                elif S[i] != x[i]:
                    S[i] = "?"
        else:
            new_G = []
            for g in G:
                if covers(g, x):
                    for i in range(n):
                        if g[i] == "?" and S[i] != "?" and S[i] != "Ø" and S[i] != x[i]:
                            candidate = g.copy()
                            candidate[i] = S[i]
                            if covers(candidate, x):
                                continue
                            new_G.append(candidate)
                else:
                    new_G.append(g)
            # Remove duplicate hypotheses
            G = [list(v) for v in {tuple(h) for h in new_G}]
    print("Specific boundary S:", S)
    print("General boundary G:", G)

if __name__ == "__main__":
    experiment_02_candidate_elimination()
