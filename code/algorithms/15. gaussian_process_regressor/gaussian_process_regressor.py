# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:07:35 2025

@author: Sami
"""

import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
X1 = data.drop(columns=[dVar1, dVar2], errors='ignore')
X2 = data.drop(columns=[dVar1, dVar2], errors='ignore')

# Address multicollinearity: Drop highly correlated variables (e.g., Poverty Rate)
X1 = X1.drop(columns=['Poverty rate'], errors='ignore')
X2 = X2.drop(columns=['Poverty rate'], errors='ignore')

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Standardizing the features (important for GPR)
scaler_X1 = StandardScaler()
scaler_y1 = StandardScaler()
X1_train_scaled = scaler_X1.fit_transform(X1_train)
X1_test_scaled = scaler_X1.transform(X1_test)
y1_train_scaled = scaler_y1.fit_transform(y1_train.values.reshape(-1, 1)).flatten()
y1_test_scaled = scaler_y1.transform(y1_test.values.reshape(-1, 1)).flatten()

scaler_X2 = StandardScaler()
scaler_y2 = StandardScaler()
X2_train_scaled = scaler_X2.fit_transform(X2_train)
X2_test_scaled = scaler_X2.transform(X2_test)
y2_train_scaled = scaler_y2.fit_transform(y2_train.values.reshape(-1, 1)).flatten()
y2_test_scaled = scaler_y2.transform(y2_test.values.reshape(-1, 1)).flatten()

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

# Function to fit and evaluate Gaussian Process Regressor
def fit_and_evaluate_gpr(X_train, X_test, y_train, y_test, target_name, kernel=None, alpha=1e-10):
    """
    Fits a Gaussian Process Regression model and evaluates its performance.

    Parameters:
        X_train: Training feature set
        X_test: Test feature set
        y_train: Training target variable
        y_test: Test target variable
        target_name: Name of the dependent variable
        kernel: The kernel to use in the Gaussian process (default: RBF kernel)
        alpha: Noise term to improve numerical stability (default: 1e-10)
    """

    # Define the default kernel if none is provided
    if kernel is None:
        kernel = C(1.0) * RBF(length_scale=1.0)

    # Fit the Gaussian Process Regressor
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=42, n_restarts_optimizer=10)
    model.fit(X_train, y_train)

    # Predictions and uncertainty estimates
    y_pred, std_dev = model.predict(X_test, return_std=True)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nGaussian Process Regression Model for {target_name} (Kernel={kernel})")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Save results to CSV
    save_results_to_csv('Gaussian Process Regression', target_name, mse, r2)

    # Save predictions and standard deviations
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Std_Deviation': std_dev})
    results.to_csv(f"{target_name.replace(' ', '_').lower()}_gpr_predictions.csv", index=False)

# Fit and evaluate GPR models
fit_and_evaluate_gpr(X1_train_scaled, X1_test_scaled, y1_train_scaled, y1_test_scaled, dVar1)
fit_and_evaluate_gpr(X2_train_scaled, X2_test_scaled, y2_train_scaled, y2_test_scaled, dVar2)
