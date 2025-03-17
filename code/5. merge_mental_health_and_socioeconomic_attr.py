# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:01:49 2025

@author: Sami
"""

import pandas as pd

# File paths
mental_health_file = "../data/processed_data/processed_county_data_mental_health_attr.csv"
socioeconomic_file = "../data/processed_data/processed_county_data_socioeconomic_attr.csv"
output_file = "../data/processed_data/processed_county_data_mental_health_and_socioeconomic_attr.csv"

# Load both datasets
mental_health_df = pd.read_csv(mental_health_file)
socioeconomic_df = pd.read_csv(socioeconomic_file)

# Select only the columns from socioeconomic_df excluding the first four
socioeconomic_columns_to_add = socioeconomic_df.columns[4:]

# Merge the dataframes on the first four columns (assuming they are the same in both files)
merged_df = pd.concat([mental_health_df, socioeconomic_df[socioeconomic_columns_to_add]], axis=1)

# Save the merged dataframe to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved as {output_file}")
