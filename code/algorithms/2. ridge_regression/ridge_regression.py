# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 07:37:34 2025

@author: Sami
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
# X1 = data.drop(columns=[dVar1, dVar2], errors='ignore')
X2 = data.drop(columns=[dVar2], errors='ignore')

# Address multicollinearity: Drop highly correlated variables (e.g., Poverty Rate)
# X1 = X1.drop(columns=['Poverty rate'], errors='ignore')
# X2 = X2.drop(columns=['Poverty rate'], errors='ignore')

# Split the data into training and testing sets
# X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Function to save results to a CSV file
def save_results_to_csv(algorithm_name, target_name, mse, r2, results_file='../model_performance.csv'):
    """
    Saves the model performance results to a CSV file.

    Parameters:
        algorithm_name: Name of the algorithm
        target_name: Name of the dependent variable
        mse: Mean Squared Error
        r2: R-squared value
        results_file: Path to the CSV file to save results (default: '../model_performance.csv')
    """
    # Prepare the results as a DataFrame
    result_data = pd.DataFrame({
        'Algorithm': [algorithm_name],
        'Dependent Variable': [target_name],
        'MSE': [mse],
        'R2': [r2]
    })

    # Check if the CSV file exists
    if os.path.exists(results_file):
        # If the file exists, append the new results
        result_data.to_csv(results_file, mode='a', header=False, index=False)
    else:
        # If the file does not exist, create it and add the results
        result_data.to_csv(results_file, mode='w', header=True, index=False)

# Function to fit Ridge Regression, calculate B and SE
def fit_and_evaluate_ridge(X_train, X_test, y_train, y_test, target_name, alpha=1.0):
    """
    Fits a Ridge regression model, calculates coefficients (B) and standard errors (SE),
    and evaluates model performance.
    
    Parameters:
        X_train: Training feature set
        X_test: Test feature set
        y_train: Training target variable
        y_test: Test target variable
        target_name: Name of the dependent variable
        alpha: Regularization strength (default: 1.0)
    """
    
    # Fit the Ridge regression model
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nRidge Regression Model for {target_name} (Alpha={alpha})")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Save results to CSV
    save_results_to_csv('Ridge Regression', target_name, mse, r2)

    # Compute coefficients (B values)
    coefficients = model.coef_

    # Compute standard errors (SE) using residual variance and covariance matrix
    residuals = y_train - model.predict(X_train)
    residual_variance = np.var(residuals, ddof=X_train.shape[1])
    X_matrix = np.hstack([np.ones((X_train.shape[0], 1)), X_train])  # Add constant
    cov_matrix = np.linalg.inv(X_matrix.T @ X_matrix + alpha * np.eye(X_matrix.shape[1])) * residual_variance
    standard_errors = np.sqrt(np.diag(cov_matrix)[1:])  # Exclude constant SE

    # Create results DataFrame
    results = pd.DataFrame({'Variable': X_train.columns, 'B (Coefficient)': coefficients, 'SE (Standard Error)': standard_errors})
    
    print("\nModel Coefficients with Standard Errors:")
    print(results)

    # Save results to CSV
    results.to_csv(f"{target_name.replace(' ', '_').lower()}_ridge_coefficients_se.csv", index=False)

# Fit and evaluate Ridge Regression models
# fit_and_evaluate_ridge(X1_train, X1_test, y1_train, y1_test, dVar1, alpha=1.0)
fit_and_evaluate_ridge(X2_train, X2_test, y2_train, y2_test, dVar2, alpha=1.0)
