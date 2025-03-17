# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 6 2025

@author: Sami (Updated for 10-Fold Cross-Validation with Gaussian Process Regression & Hyperparameter Tuning)
"""

import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV

# Load the dataset
file_path = "../../data_consolidated_dropped_race_matched_renamed_vifDropped.csv"
data = pd.read_csv(file_path)

# Exclude 'FIPS', 'Year', 'State', and 'County' columns and handle missing values
data = data.drop(columns=['FIPS', 'Year', 'State', 'County'], errors='ignore')
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Define dependent variables
dVar1 = 'Poor mental health days'
dVar2 = '% Frequent mental distress'

# Define dependent variables
y1 = data[dVar1]
y2 = data[dVar2]

# Define independent variables
X = data.drop(columns=[dVar1, dVar2, 'Poverty rate'], errors='ignore')

# Standardizing the features (important for GPR)
scaler_X = StandardScaler()
scaler_y1 = StandardScaler()
scaler_y2 = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y1_scaled = scaler_y1.fit_transform(y1.values.reshape(-1, 1)).flatten()
y2_scaled = scaler_y2.fit_transform(y2.values.reshape(-1, 1)).flatten()

# Define hyperparameter grid for GPR tuning
param_grid = {
    'kernel': [C(1.0) * RBF(length_scale) for length_scale in [0.1, 1.0, 10.0, 100.0]],
    'alpha': [1e-10, 1e-5, 1e-2, 1.0]  # Noise level
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

# Function to perform Gaussian Process Regression with hyperparameter tuning
def cross_validate_gpr_with_tuning(X, y, target_name, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform Grid Search to find the best hyperparameters
    gpr = GaussianProcessRegressor(n_restarts_optimizer=10, random_state=42)
    grid_search = GridSearchCV(gpr, param_grid, cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best hyperparameters
    best_kernel = grid_search.best_params_['kernel']
    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_model.fit(X_train, y_train)
        y_pred, std_dev = best_model.predict(X_test, return_std=True)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Compute average results across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"\n10-Fold Cross-Validation Results for Gaussian Process Regression ({target_name}, Best Kernel={best_kernel}, Best Alpha={best_alpha}):")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv(f'Gaussian Process Regression (10-Fold CV, Best Kernel={best_kernel}, Best Alpha={best_alpha})', target_name, avg_mse, avg_r2)

    return best_kernel, best_alpha, avg_mse, avg_r2

# Run Hyperparameter Tuning + 10-Fold CV for both dependent variables
best_kernel_1, best_alpha_1, mse_1, r2_1 = cross_validate_gpr_with_tuning(X_scaled, y1_scaled, dVar1)
best_kernel_2, best_alpha_2, mse_2, r2_2 = cross_validate_gpr_with_tuning(X_scaled, y2_scaled, dVar2)
