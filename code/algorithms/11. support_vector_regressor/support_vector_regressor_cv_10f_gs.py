# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 6 2025

@author: Sami (Updated for 10-Fold Cross-Validation with Support Vector Regression & Hyperparameter Tuning)
"""

import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV

# Load the dataset
file_path = "../../../data/processed_data/processed_county_data_mental_health_and_socioeconomic_attr_vif_dropped.csv"
data = pd.read_csv(file_path)

# Exclude 'FIPS', 'Year', 'State', and 'County' columns and handle missing values
data = data.drop(columns=['FIPS', 'Year', 'State', 'County'], errors='ignore')
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Define dependent variables
# dVar1 = 'Poor mental health days'
dVar2 = 'Frequent Mental Distress'

# Define dependent variables
# y1 = data[dVar1]
y2 = data[dVar2]

# Define independent variables
X = data.drop(columns=[dVar2], errors='ignore')

# Standardizing the features (important for SVR)
scaler_X = StandardScaler()
#scaler_y1 = StandardScaler()
scaler_y2 = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
#y1_scaled = scaler_y1.fit_transform(y1.values.reshape(-1, 1)).flatten()
y2_scaled = scaler_y2.fit_transform(y2.values.reshape(-1, 1)).flatten()

# Define hyperparameter grid for SVR tuning
param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Different kernels
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'epsilon': [0.01, 0.1, 0.5, 1.0]  # Tolerance for error margin
}

# Function to save results to a CSV file
def save_results_to_csv(algorithm_name, target_name, mse, r2, results_file='../model_performance.csv'):
    result_data = pd.DataFrame({
        'Algorithm': [algorithm_name],
        'Dependent Variable': [target_name],
        'MSE': [mse],
        'R2': [r2]
    })
    if os.path.exists(results_file):
        result_data.to_csv(results_file, mode='a', header=False, index=False)
    else:
        result_data.to_csv(results_file, mode='w', header=True, index=False)

# Function to perform SVR with hyperparameter tuning
def cross_validate_svr_with_tuning(X, y, target_name, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform Grid Search to find the best hyperparameters
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best hyperparameters
    best_kernel = grid_search.best_params_['kernel']
    best_C = grid_search.best_params_['C']
    best_epsilon = grid_search.best_params_['epsilon']
    best_model = grid_search.best_estimator_

    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Compute average results across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"\n10-Fold Cross-Validation Results for SVR ({target_name}, Best Kernel={best_kernel}, Best C={best_C}, Best Epsilon={best_epsilon}):")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv(f'Support Vector Regression (10-Fold CV, Best Kernel={best_kernel}, Best C={best_C}, Best Epsilon={best_epsilon})', target_name, avg_mse, avg_r2)

    return best_kernel, best_C, best_epsilon, avg_mse, avg_r2

# Run Hyperparameter Tuning + 10-Fold CV for both dependent variables
#best_kernel_1, best_C_1, best_epsilon_1, mse_1, r2_1 = cross_validate_svr_with_tuning(X_scaled, y1_scaled, dVar1)
best_kernel_2, best_C_2, best_epsilon_2, mse_2, r2_2 = cross_validate_svr_with_tuning(X_scaled, y2_scaled, dVar2)
