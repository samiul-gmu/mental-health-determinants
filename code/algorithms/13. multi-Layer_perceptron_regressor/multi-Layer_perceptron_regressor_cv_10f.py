# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 6 2025

@author: Sami (Updated for 10-Fold Cross-Validation with Multi-Layer Perceptron Regression)
"""

import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

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

# Function to perform 10-fold cross-validation using MLP Regression
def cross_validate_mlp(X, y, target_name, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mse_scores = []
    r2_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                             solver=solver, max_iter=max_iter, random_state=42, learning_rate='adaptive')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Average MSE and R² across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"\n10-Fold Cross-Validation Results for MLP Regression ({target_name}, hidden_layers={hidden_layer_sizes}, activation={activation}, solver={solver}, max_iter={max_iter}):")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv('Multi-Layer Perceptron Regression (10-Fold CV)', target_name, avg_mse, avg_r2)

# Run 10-fold cross-validation for both dependent variables using MLP Regression
cross_validate_mlp(X_scaled, y1_scaled, dVar1, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
cross_validate_mlp(X_scaled, y2_scaled, dVar2, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
