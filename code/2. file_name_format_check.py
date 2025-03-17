# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:53:53 2024

@author: Sami
"""

import os
import re

# Define the root directory, where the script will start its search
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw_data')

# Define the expected fixed parts of the file name
fixed_part_1 = " County Health Rankings "
fixed_part_2 = " Data.xlsx"

# Regular expression pattern to validate the file name
file_name_pattern = re.compile(r"^\d{4} County Health Rankings .+ Data\.xlsx$")

# Function to print messages in red
def print_in_red(message):
    print(f"\033[91m{message}\033[0m")

# Function to print messages in green
def print_in_green(message):
    print(f"\033[92m{message}\033[0m")

# Get the list of year folders and sort them in reverse order (to start with the oldest year)
year_folders = sorted(os.listdir(root_dir), reverse=True)

# Dictionary to store summary counts per year
summary = {}

# Loop through each year folder, starting from the oldest
for year_folder in year_folders:
    year_folder_path = os.path.join(root_dir, year_folder)
    
    # Check if it's a directory and if it contains the 'version_no_removed' folder
    version_removed_folder = os.path.join(year_folder_path, 'version_no_removed')
    if os.path.isdir(version_removed_folder):
        print(f"\nChecking files in: {version_removed_folder}")

        # Counters for valid and invalid files
        valid_count = 0
        invalid_count = 0

        # Iterate over files in the 'version_no_removed' folder
        for file_name in os.listdir(version_removed_folder):
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):  # Only consider Excel files
                # Check if the file name matches the expected pattern
                if file_name_pattern.match(file_name):
                    print_in_green(f"{file_name}: follows the correct format. ✔")
                    valid_count += 1
                else:
                    print_in_red(f"{file_name}: does NOT follow the correct format. ✘")
                    invalid_count += 1

        # Store the summary for this year
        summary[year_folder] = {"Valid": valid_count, "Invalid": invalid_count}

# Print final summary
print("\n--- Summary Report ---")
for year, counts in summary.items():
    print(f"\nYear: {year}")
    print(f"✔ Valid files: {counts['Valid']}")
    print(f"✘ Invalid files: {counts['Invalid']}")

print("\nFile name check completed.")
