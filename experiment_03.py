"""Experiment 3: Id3"""

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

def experiment_03_id3():
    print("\nEXPERIMENT 3: ID3 DECISION TREE")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
    )
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    show_classification_results("ID3-style decision tree", y_test, pred)
    sample = X_test[0].reshape(1, -1)
    print("Sample:", sample, "Predicted class:", iris.target_names[model.predict(sample)[0]])

if __name__ == "__main__":
    experiment_03_id3()
