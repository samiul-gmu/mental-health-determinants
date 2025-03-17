# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:29:29 2025

@author: Sami
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
file_path = "../data/processed_data/processed_county_data_mental_health_and_socioeconomic_attr.csv"
data = pd.read_csv(file_path)

# Exclude 'FIPS', 'year', 'State', and 'County' columns
data = data.drop(columns=['FIPS', 'Year', 'State', 'County', 'Frequent Mental Distress'], errors='ignore')

# High VIF
data = data.drop(columns=['# Mental Health Provider', 'Children in Poverty'], errors='ignore')


# Drop rows with NaN or Inf values
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Initialize a DataFrame to store VIF values
vif_data = pd.DataFrame()
vif_data["Variable"] = data.columns
vif_values = []

# Calculate VIF for each variable
for i in range(len(data.columns)):
    # Separate the current feature from the others
    X = data.drop(data.columns[i], axis=1)
    y = data.iloc[:, i]
    
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Calculate VIF
    vif = 1 / (1 - r_squared)
    vif_values.append(vif)

vif_data["VIF"] = vif_values

# Display the VIF values
print(vif_data)

# Optionally save the VIF values to a CSV file
vif_data.to_csv("../data/processed_data/vif_values_without_dependents.csv", index=False)
