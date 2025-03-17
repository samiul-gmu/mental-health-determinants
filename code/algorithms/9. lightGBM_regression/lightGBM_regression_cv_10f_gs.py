# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 6 2025

@author: Sami (Updated for 10-Fold Cross-Validation with LightGBM Regression & Hyperparameter Tuning)
"""

import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor
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

# Define hyperparameter grid for LightGBM tuning
param_grid = {
    'num_leaves': [10, 31, 50],  # Number of leaves in one tree
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
    'n_estimators': [50, 100, 200],  # Number of boosting rounds
    'max_depth': [3, 5, 10],  # Maximum depth of trees
    'reg_lambda': [0, 1, 10],  # L2 regularization
    'reg_alpha': [0, 1, 10]  # L1 regularization
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

# Function to perform LightGBM Regression with hyperparameter tuning
def cross_validate_lightgbm_with_tuning(X, y, target_name, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform Grid Search to find the best hyperparameters
    lgbm = LGBMRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(lgbm, param_grid, cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best hyperparameters
    best_num_leaves = grid_search.best_params_['num_leaves']
    best_learning_rate = grid_search.best_params_['learning_rate']
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_depth = grid_search.best_params_['max_depth']
    best_reg_lambda = grid_search.best_params_['reg_lambda']
    best_reg_alpha = grid_search.best_params_['reg_alpha']
    best_model = grid_search.best_estimator_

    mse_scores = []
    r2_scores = []
    feature_importance_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        # Store feature importance
        feature_importance_list.append(best_model.feature_importances_)

    # Compute average results across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_feature_importance = np.mean(feature_importance_list, axis=0)

    print(f"\n10-Fold Cross-Validation Results for LightGBM Regression ({target_name}, Best num_leaves={best_num_leaves}, Best learning_rate={best_learning_rate}, Best n_estimators={best_n_estimators}, Best max_depth={best_max_depth}, Best reg_lambda={best_reg_lambda}, Best reg_alpha={best_reg_alpha}):")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv(f'LightGBM Regression (10-Fold CV, Best num_leaves={best_num_leaves}, Best learning_rate={best_learning_rate}, Best n_estimators={best_n_estimators}, Best max_depth={best_max_depth}, Best reg_lambda={best_reg_lambda}, Best reg_alpha={best_reg_alpha})', target_name, avg_mse, avg_r2)

    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'Variable': X.columns,
        'Importance': avg_feature_importance
    }).sort_values(by='Importance', ascending=False)

    print("\nAverage Feature Importance (LightGBM):")
    print(feature_importance_df)

    feature_importance_df.to_csv(f"{target_name.replace(' ', '_').lower()}_lightgbm_feature_importance.csv", index=False)

    return best_num_leaves, best_learning_rate, best_n_estimators, best_max_depth, best_reg_lambda, best_reg_alpha, avg_mse, avg_r2

# Run Hyperparameter Tuning + 10-Fold CV for both dependent variables
#best_num_leaves_1, best_learning_rate_1, best_n_estimators_1, best_max_depth_1, best_reg_lambda_1, best_reg_alpha_1, mse_1, r2_1 = cross_validate_lightgbm_with_tuning(X, y1, dVar1)
best_num_leaves_2, best_learning_rate_2, best_n_estimators_2, best_max_depth_2, best_reg_lambda_2, best_reg_alpha_2, mse_2, r2_2 = cross_validate_lightgbm_with_tuning(X, y2, dVar2)
