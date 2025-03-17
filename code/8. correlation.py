# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:13:42 2025

@author: Sami
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load the dataset
file_path = "../data/processed_data/processed_county_data_mental_health_and_socioeconomic_attr.csv"
data = pd.read_csv(file_path)

# Exclude 'FIPS' and 'year' columns
data = data.drop(columns=['FIPS', 'Year', 'State', 'County'], errors='ignore')

# Drop rows with NaN or Inf values
data = data.replace([np.inf, -np.inf], np.nan).dropna()


# Initialize an empty DataFrame for the correlation matrix
correlation_matrix = pd.DataFrame(index=data.columns, columns=data.columns)

# Initialize an empty DataFrame for the annotated correlation matrix
annotated_matrix = pd.DataFrame(index=data.columns, columns=data.columns)


# Compute the correlation matrix and annotate with significance levels
for col1 in data.columns:
    for col2 in data.columns:
        if col1 == col2:
            correlation_matrix.loc[col1, col2] = 1.0  # Diagonal is 1
            annotated_matrix.loc[col1, col2] = 1.0  # Diagonal is 1
        elif data.columns.get_loc(col1) > data.columns.get_loc(col2):  # Lower triangle only
            print(data[col1], data[col2])   
            corr, p_value = pearsonr(data[col1], data[col2])
            if p_value < 0.01:
                annotated_matrix.loc[col1, col2] = f"{corr:.2f}**"
            elif p_value < 0.05:
                annotated_matrix.loc[col1, col2] = f"{corr:.2f}*"
            else:
                annotated_matrix.loc[col1, col2] = f"{corr:.2f}"
            correlation_matrix.loc[col1, col2] = f"{corr:.2f}"
        else:
            annotated_matrix.loc[col1, col2] = ""  # Leave upper triangle empty

# Save the lower triangle matrix to a CSVfile
correlation_matrix.to_csv("../data/processed_data/correlation_matrix_without_significance.csv")

# Save the annotated lower triangle matrix to a CSVfile
annotated_matrix.to_csv("../data/processed_data/correlation_matrix.csv")

# Display the annotated matrix
print(annotated_matrix)
