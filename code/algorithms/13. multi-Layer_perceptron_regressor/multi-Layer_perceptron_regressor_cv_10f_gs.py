# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 6 2025

@author: Sami (Updated for 10-Fold Cross-Validation with MLP Regression & Hyperparameter Tuning)
"""

import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
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

# Standardizing the features (important for MLP)
scaler_X = StandardScaler()
scaler_y1 = StandardScaler()
scaler_y2 = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y1_scaled = scaler_y1.fit_transform(y1.values.reshape(-1, 1)).flatten()
y2_scaled = scaler_y2.fit_transform(y2.values.reshape(-1, 1)).flatten()

# Define hyperparameter grid for MLP tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Different network architectures
    'activation': ['relu', 'tanh'],  # Activation functions
    'solver': ['adam', 'lbfgs'],  # Optimizers
    'learning_rate': ['constant', 'adaptive']  # Learning rate schedules
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

# Function to perform MLP Regression with hyperparameter tuning
def cross_validate_mlp_with_tuning(X, y, target_name, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform Grid Search to find the best hyperparameters
    mlp = MLPRegressor(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best hyperparameters
    best_hidden_layer_sizes = grid_search.best_params_['hidden_layer_sizes']
    best_activation = grid_search.best_params_['activation']
    best_solver = grid_search.best_params_['solver']
    best_learning_rate = grid_search.best_params_['learning_rate']
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

    print(f"\n10-Fold Cross-Validation Results for MLP Regression ({target_name}, Best hidden_layers={best_hidden_layer_sizes}, Best activation={best_activation}, Best solver={best_solver}, Best learning_rate={best_learning_rate}):")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv(f'MLP Regression (10-Fold CV, Best hidden_layers={best_hidden_layer_sizes}, Best activation={best_activation}, Best solver={best_solver}, Best learning_rate={best_learning_rate})', target_name, avg_mse, avg_r2)

    return best_hidden_layer_sizes, best_activation, best_solver, best_learning_rate, avg_mse, avg_r2

# Run Hyperparameter Tuning + 10-Fold CV for both dependent variables
best_hidden_layer_sizes_1, best_activation_1, best_solver_1, best_learning_rate_1, mse_1, r2_1 = cross_validate_mlp_with_tuning(X_scaled, y1_scaled, dVar1)
best_hidden_layer_sizes_2, best_activation_2, best_solver_2, best_learning_rate_2, mse_2, r2_2 = cross_validate_mlp_with_tuning(X_scaled, y2_scaled, dVar2)
