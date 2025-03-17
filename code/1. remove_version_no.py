# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:44:10 2024

@author: Sami
"""

import os
import re
import shutil
import pandas as pd

# Define the root directory, where the script will start its search
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw_data')
print(root_dir)

# Function to convert xls to xlsx
def convert_xls_to_xlsx(input_file, output_file):
    # Change the file extension from .xls to .xlsx
    # output_file = os.path.splitext(input_file)[0] + '.xlsx'
    
    # Read the xls file
    xls_file = pd.ExcelFile(input_file, engine='xlrd')
    
    # Create a writer object for the xlsx output file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Loop through each sheet and save it to the new xlsx file
        for sheet_name in xls_file.sheet_names:
            df = xls_file.parse(sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Converted {input_file} to {output_file}")

# Function to remove version info from the file name
def remove_version_info(file_name):
    # Regular expression to match ' - v1_0', ' - v1', or similar patterns and remove them
    new_name = re.sub(r' - v\d+(_\d+)?', '', file_name)
    return new_name

# Walk through all the directories within the 'raw_data' folder
for year_folder in os.listdir(root_dir):
    print(year_folder)
    year_folder_path = os.path.join(root_dir, year_folder)

    # Check if it's a directory (year folder)
    if os.path.isdir(year_folder_path):
        # Create a new folder called 'version_no_removed' inside each year folder
        new_folder_path = os.path.join(year_folder_path, 'version_no_removed')
        os.makedirs(new_folder_path, exist_ok=True)

        # Iterate over files in the year folder
        for file_name in os.listdir(year_folder_path):
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):  # Only consider Excel files
                # Remove the version info from the file name
                new_file_name = remove_version_info(file_name)

                # Define the source and destination paths
                source_file_path = os.path.join(year_folder_path, file_name)
                destination_file_path = os.path.join(new_folder_path, new_file_name)
                
                if destination_file_path.endswith(".xls"):
                    if "Ranking" in destination_file_path and "Rankings" not in destination_file_path:
                        destination_file_path = destination_file_path.replace("Ranking", "Rankings")
                    destination_file_path = destination_file_path[:-4] + ".xlsx"
                    convert_xls_to_xlsx(source_file_path, destination_file_path)
                else:
                    # Copy the file to the new location with the renamed file name
                    shutil.copy2(source_file_path, destination_file_path)
                print(f"Copied and renamed: {file_name} -> {new_file_name}")

print("All files have been processed.")