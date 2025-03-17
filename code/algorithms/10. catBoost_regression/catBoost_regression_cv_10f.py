# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 6 2025

@author: Sami (Updated for 10-Fold Cross-Validation with CatBoost Regression)
"""

import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

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

# Function to perform 10-fold cross-validation using CatBoost Regression
def cross_validate_catboost(X, y, target_name, iterations=1000, learning_rate=0.1, depth=6, l2_leaf_reg=3, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mse_scores = []
    r2_scores = []
    feature_importance_list = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, 
                                  depth=depth, l2_leaf_reg=l2_leaf_reg, loss_function='RMSE', 
                                  random_seed=42, verbose=0)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        # Store feature importance
        feature_importance_list.append(model.get_feature_importance())

    # Average MSE and R² across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"\n10-Fold Cross-Validation Results for CatBoost Regression ({target_name}, iterations={iterations}, learning_rate={learning_rate}, depth={depth}, l2_leaf_reg={l2_leaf_reg}):")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv('CatBoost Regression (10-Fold CV)', target_name, avg_mse, avg_r2)

    # Compute average feature importance
    avg_feature_importance = np.mean(feature_importance_list, axis=0)

    feature_importance_df = pd.DataFrame({
        'Variable': X.columns,
        'Importance': avg_feature_importance
    }).sort_values(by='Importance', ascending=False)

    print("\nAverage Feature Importance (CatBoost):")
    print(feature_importance_df)

    # Save feature importance results to CSV
    feature_importance_df.to_csv(f"{target_name.replace(' ', '_').lower()}_catboost_feature_importance.csv", index=False)

# Run 10-fold cross-validation for both dependent variables using CatBoost Regression
# cross_validate_catboost(X, y1, dVar1, iterations=1000, learning_rate=0.1, depth=6)
cross_validate_catboost(X, y2, dVar2, iterations=1000, learning_rate=0.1, depth=6)
