# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 6 2025

@author: Sami (Updated for 10-Fold Cross-Validation with Grid Search)
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
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

# Define a range of alpha values for tuning
alpha_values = [0.01, 0.1, 1, 10, 100]

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

# Function to perform Grid Search and 10-fold cross-validation
def cross_validate_model_with_tuning(X, y, target_name, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform Grid Search for the best alpha
    ridge = Ridge()
    param_grid = {'alpha': alpha_values}
    grid_search = GridSearchCV(ridge, param_grid, cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best alpha
    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    mse_scores = []
    r2_scores = []
    coefficients_list = []
    standard_errors_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        # Compute coefficients
        coefficients = best_model.coef_

        # Compute standard errors using pseudo-inverse for numerical stability
        residuals = y_train - best_model.predict(X_train)
        residual_variance = np.var(residuals, ddof=X_train.shape[1])
        X_matrix = np.hstack([np.ones((X_train.shape[0], 1)), X_train])  # Add constant term
        cov_matrix = np.linalg.pinv(X_matrix.T @ X_matrix + best_alpha * np.eye(X_matrix.shape[1])) * residual_variance
        standard_errors = np.sqrt(np.diag(cov_matrix)[1:])  # Exclude constant SE

        coefficients_list.append(coefficients)
        standard_errors_list.append(standard_errors)

    # Compute average results across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_coefficients = np.mean(coefficients_list, axis=0)
    avg_standard_errors = np.mean(standard_errors_list, axis=0)

    print(f"\n10-Fold Cross-Validation Results for {target_name} (Best α={best_alpha}):")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv(f'Ridge Regression (10-Fold CV, Alpha={best_alpha})', target_name, avg_mse, avg_r2)

    # Save coefficient results
    results = pd.DataFrame({
        'Variable': X.columns,
        'B (Average Coefficient)': avg_coefficients,
        'SE (Average Standard Error)': avg_standard_errors
    })

    results.to_csv(f"{target_name.replace(' ', '_').lower()}_ridge_coefficients_tuned.csv", index=False)

    return best_alpha, avg_mse, avg_r2

# Run Hyperparameter Tuning + 10-fold CV for both dependent variables
# best_alpha_1, mse_1, r2_1 = cross_validate_model_with_tuning(X, y1, dVar1)
best_alpha_2, mse_2, r2_2 = cross_validate_model_with_tuning(X, y2, dVar2)
