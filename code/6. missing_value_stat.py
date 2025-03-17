# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:58:06 2024

@author: Sami
"""

import pandas as pd

# Define the file path
file_path = '../data/processed_data/processed_county_data_mental_health_and_socioeconomic_attr.csv'

# Load the data
df = pd.read_csv(file_path)

# Calculate missing values for the entire dataset
total_missing = df.isnull().sum().sum()
total_values = df.size
missing_percentage = (total_missing / total_values)# * 100

# Calculate missing values per column
missing_per_column = df.isnull().sum()
missing_percentage_per_column = (df.isnull().sum() / len(df))# * 100

# Create a dataframe to store the results
missing_stats = pd.DataFrame({
    'Attribute': df.columns,
    'Missing Values': missing_per_column.values,
    'Percentage Missing': missing_percentage_per_column.values
})

# Add a row for overall missing values summary
summary_row = pd.DataFrame({
    'Attribute': ['Overall Missing'],
    'Missing Values': [total_missing],
    'Percentage Missing': [missing_percentage]
})

# Append the summary row to the bottom of the dataframe
missing_stats = pd.concat([missing_stats, summary_row], ignore_index=True)

# Save the results to a CSV file with better formatting
output_path = '../data/processed_data/missing_values_stats.csv'
missing_stats.to_csv(output_path, index=False)

print(f"Missing value statistics saved to {output_path}")
