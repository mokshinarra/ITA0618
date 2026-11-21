"""Experiment 19: Bank Loan"""

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

def experiment_19_bank_loan():
    print("\nEXPERIMENT 19: BANK LOAN PREDICTION")
    rng = np.random.default_rng(19)
    n = 600
    income = rng.integers(15000, 150000, n)
    credit_score = rng.integers(300, 850, n)
    loan_amount = rng.integers(5000, 200000, n)
    employment_years = rng.integers(0, 35, n)
    # Demonstration target generated from simple illustrative rules.
    approved = ((credit_score >= 620) & (income * 2.0 >= loan_amount) &
                (employment_years >= 1)).astype(int)
    X = np.column_stack([income, credit_score, loan_amount, employment_years])
    X_train, X_test, y_train, y_test = train_test_split(
        X, approved, test_size=0.25, random_state=42, stratify=approved
    )
    model = make_pipeline(StandardScaler(), GaussianNB())
    model.fit(X_train, y_train)
    show_classification_results("Bank loan approval (synthetic demonstration data)",
                                y_test, model.predict(X_test))

if __name__ == "__main__":
    experiment_19_bank_loan()
