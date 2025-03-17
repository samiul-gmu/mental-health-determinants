# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:45:01 2025

@author: Sami
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
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

# Function to fit and evaluate Extra Trees Regressor
def fit_and_evaluate_extratrees(X_train, X_test, y_train, y_test, target_name, n_estimators=100, max_depth=None, min_samples_split=2):
    """
    Fits an Extra Trees regression model and evaluates its performance.

    Parameters:
        X_train: Training feature set
        X_test: Test feature set
        y_train: Training target variable
        y_test: Test target variable
        target_name: Name of the dependent variable
        n_estimators: Number of trees in the ensemble (default: 100)
        max_depth: Maximum depth of each tree (default: None)
        min_samples_split: Minimum samples required to split an internal node (default: 2)
    """

    # Fit the Extra Trees Regressor
    model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                min_samples_split=min_samples_split, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nExtra Trees Regression Model for {target_name} (n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split})")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Save results to CSV
    save_results_to_csv('Extra Trees Regression', target_name, mse, r2)

    # Get feature importance scores
    feature_importance = pd.DataFrame({'Variable': X_train.columns, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("\nFeature Importance (Extra Trees):")
    print(feature_importance)

    # Save feature importances to CSV
    feature_importance.to_csv(f"{target_name.replace(' ', '_').lower()}_extratrees_feature_importance.csv", index=False)

# Fit and evaluate Extra Trees models
#fit_and_evaluate_extratrees(X1_train, X1_test, y1_train, y1_test, dVar1, n_estimators=100, max_depth=10, min_samples_split=5)
fit_and_evaluate_extratrees(X2_train, X2_test, y2_train, y2_test, dVar2, n_estimators=100, max_depth=10, min_samples_split=5)
