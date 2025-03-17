import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
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

# Function to perform 10-fold cross-validation
def cross_validate_model(X, y, target_name, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mse_scores = []
    r2_scores = []
    coefficients_list = []
    standard_errors_list = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        # Compute coefficients
        coefficients = model.coef_

        # Compute standard errors
        residuals = y_train - model.predict(X_train)
        residual_variance = np.var(residuals, ddof=X_train.shape[1])
        X_matrix = np.hstack([np.ones((X_train.shape[0], 1)), X_train])  # Add constant
        cov_matrix = np.linalg.inv(X_matrix.T @ X_matrix) * residual_variance
        standard_errors = np.sqrt(np.diag(cov_matrix)[1:])  # Exclude constant SE

        coefficients_list.append(coefficients)
        standard_errors_list.append(standard_errors)

    # Average MSE and R² across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"\n10-Fold Cross-Validation Results for {target_name}:")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")

    # Save results to CSV
    save_results_to_csv('Linear Regression (10-Fold CV)', target_name, avg_mse, avg_r2)

    # Compute average coefficients and standard errors
    avg_coefficients = np.mean(coefficients_list, axis=0)
    avg_standard_errors = np.mean(standard_errors_list, axis=0)

    results = pd.DataFrame({
        'Variable': X.columns,
        'B (Average Coefficient)': avg_coefficients,
        'SE (Average Standard Error)': avg_standard_errors
    })

    print("\nAverage Model Coefficients with Standard Errors:")
    print(results)

    # Save coefficient results to CSV
    results.to_csv(f"{target_name.replace(' ', '_').lower()}_coefficients_se.csv", index=False)

# Run 10-fold cross-validation for both dependent variables
# cross_validate_model(X, y1, dVar1)
cross_validate_model(X, y2, dVar2)
