# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 6 2025

@author: Sami (Updated for 10-Fold Cross-Validation with Bagging Regression)
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
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

# Define independent variables
y1 = data[dVar1]
y2 = data[dVar2]
X = data.drop(columns=[dVar1, dVar2, 'Poverty rate'], errors='ignore')

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

# Function to perform 10-fold cross-validation using Bagging Regression
def cross_validate_bagging(X, y, target_name, n_estimators=100, max_samples=0.8, bootstrap=True, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mse_scores = []
    r2_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Base estimator (weak learner)
        base_estimator = DecisionTreeRegressor(max_depth=5)

        model = BaggingRegressor(base_estimator=base_estimator, n_estimators=n_estimators, 
                                 max_samples=max_samples, bootstrap=bootstrap, 
                                 random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Average MSE and R² across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"\n10-Fold Cross-Validation Results for Bagging Regression ({target_name}, n_estimators={n_estimators}, max_samples={max_samples}, bootstrap={bootstrap}):")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv('Bagging Regression (10-Fold CV)', target_name, avg_mse, avg_r2)

    # Feature Importance (fit on full dataset for importance computation)
    model.fit(X, y)  
    feature_importance = pd.DataFrame({'Variable': X.columns, 'Importance': model.estimators_[0].feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("\nFeature Importance (Bagging):")
    print(feature_importance)

    # Save feature importance scores to CSV
    feature_importance.to_csv(f"{target_name.replace(' ', '_').lower()}_bagging_feature_importance.csv", index=False)

# Run 10-fold cross-validation for both dependent variables using Bagging Regression
cross_validate_bagging(X, y1, dVar1)
cross_validate_bagging(X, y2, dVar2)
