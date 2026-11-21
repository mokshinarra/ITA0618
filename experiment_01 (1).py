"""Experiment 1: Find S"""

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

def experiment_01_find_s():
    print("\nEXPERIMENT 1: FIND-S ALGORITHM")
    data = [
        ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
        ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
        ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
        ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"],
    ]
    attributes = ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"]
    hypothesis = ["Ø"] * len(attributes)

    for row in data:
        if row[-1] == "Yes":
            for i, value in enumerate(row[:-1]):
                if hypothesis[i] == "Ø":
                    hypothesis[i] = value
                elif hypothesis[i] != value:
                    hypothesis[i] = "?"
    print("Most specific hypothesis:", hypothesis)

if __name__ == "__main__":
    experiment_01_find_s()
