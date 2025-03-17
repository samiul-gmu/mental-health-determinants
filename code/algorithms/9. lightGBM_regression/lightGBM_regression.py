# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:03:09 2025

@author: Sami
"""

import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor
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
#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
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

# Function to fit and evaluate a LightGBM model
def fit_and_evaluate_lightgbm(X_train, X_test, y_train, y_test, target_name, num_leaves=31, learning_rate=0.1, n_estimators=100, max_depth=-1, reg_lambda=1, reg_alpha=0):
    """
    Fits a LightGBM regression model and evaluates its performance.

    Parameters:
        X_train: Training feature set
        X_test: Test feature set
        y_train: Training target variable
        y_test: Test target variable
        target_name: Name of the dependent variable
        num_leaves: Maximum number of leaves in a tree (default: 31)
        learning_rate: Shrinks the contribution of each tree (default: 0.1)
        n_estimators: Number of boosting rounds (default: 100)
        max_depth: Maximum depth of each individual tree (-1 means no limit)
        reg_lambda: L2 regularization (default: 1)
        reg_alpha: L1 regularization (default: 0)
    """

    # Fit the LightGBM Regressor
    model = LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, 
                          n_estimators=n_estimators, max_depth=max_depth,
                          reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                          random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nLightGBM Model for {target_name} (num_leaves={num_leaves}, learning_rate={learning_rate}, n_estimators={n_estimators}, max_depth={max_depth}, reg_lambda={reg_lambda}, reg_alpha={reg_alpha})")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Save results to CSV
    save_results_to_csv('LightGBM Regression', target_name, mse, r2)

    # Get feature importance scores
    feature_importance = pd.DataFrame({'Variable': X_train.columns, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("\nFeature Importance (LightGBM):")
    print(feature_importance)

    # Save feature importance scores to CSV
    feature_importance.to_csv(f"{target_name.replace(' ', '_').lower()}_lightgbm_feature_importance.csv", index=False)

# Fit and evaluate LightGBM models
#fit_and_evaluate_lightgbm(X1_train, X1_test, y1_train, y1_test, dVar1, num_leaves=31, learning_rate=0.1, n_estimators=100, max_depth=5)
fit_and_evaluate_lightgbm(X2_train, X2_test, y2_train, y2_test, dVar2, num_leaves=31, learning_rate=0.1, n_estimators=100, max_depth=5)
