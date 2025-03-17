# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:03:54 2025

@author: Sami
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# Load the first CSV file into a DataFrame
# matrix1_path = 'correlation_matrix_without_significance_paper.csv'
# matrix1_df = pd.read_csv(matrix1_path, index_col=0)

# Load the second CSV file into a DataFrame
matrix2_path = '../data/processed_data/correlation_matrix_without_significance.csv'
matrix2_df = pd.read_csv(matrix2_path, index_col=0)

# Convert the DataFrames to numeric matrices
# matrix1 = matrix1_df.to_numpy()
matrix2 = matrix2_df.to_numpy()

# Generate heatmaps
plt.figure(figsize=(12, 8), dpi=150)  # Adjust DPI
sns.heatmap(matrix2, annot=True, cmap="coolwarm", xticklabels=matrix2_df.columns, yticklabels=matrix2_df.index)
plt.title("Corr Matrix")
plt.tight_layout()
plt.show()

