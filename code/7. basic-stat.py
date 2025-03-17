# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:05:47 2025

@author: Sami
"""

import pandas as pd

# Load the dataset
file_path = "../data/processed_data/processed_county_data_mental_health_and_socioeconomic_attr.csv"
data = pd.read_csv(file_path)

# Exclude 'FIPS', 'Year', 'State' and 'County' columns and handle missing values
data = data.drop(columns=['FIPS', 'Year', 'State', 'County'], errors='ignore')

# Calculate descriptive statistics and transpose the results
descriptive_stats = data.describe().loc[['mean', 'std', 'min', 'max']].transpose()

# Display the transposed descriptive statistics
print(descriptive_stats)

# Optionally, save the transposed descriptive statistics to a CSV file
descriptive_stats.to_csv("../data/processed_data/descriptive_statistics.csv")
