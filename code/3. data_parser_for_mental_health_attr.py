import os
import pandas as pd

# Define the root directory
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw_data')

# Define the output CSV path
output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed_data', 'processed_county_data_mental_health_attr.csv')

# Define the sheet names we are looking for
sheet_options = ["Ranked Measure Data", "Select Measure Data"]
additional_sheet = "Additional Measure Data"

# List of columns to copy from Ranked/Select Measure Data sheet
selected_columns = ['# Mental Health Provider', 'Mental Health Provider Rate',
                    '% Excessive Drinking', '% Uninsured']

# List of columns to copy from Additional Measure Data sheet
additional_columns = ['Frequent Mental Distress', 'Drug Overdose Deaths',
                      '% Insufficient Sleep']

# Initialize a flag to indicate whether we've added the column headers to the CSV
headers_added = False

# Function to extract year from the file name
def extract_year(file_name):
    return file_name.split(' ')[0]

# Get the list of year folders and sort them (no reverse, so older years come first)
year_folders = sorted(os.listdir(root_dir), reverse=True)

# Ensure processed_data directory exists
processed_data_dir = os.path.join(root_dir, '..', 'processed_data')
os.makedirs(processed_data_dir, exist_ok=True)

# Loop through each year folder, starting from the oldest
for year_folder in year_folders:
    year_folder_path = os.path.join(root_dir, year_folder)
    
    # Check if it's a directory and contains the 'version_no_removed' folder
    version_removed_folder = os.path.join(year_folder_path, 'version_no_removed')
    if os.path.isdir(version_removed_folder):
        # Iterate over files in the 'version_no_removed' folder
        for file_name in os.listdir(version_removed_folder):
            if file_name.endswith('.xlsx'):  # Only consider Excel files
                file_path = os.path.join(version_removed_folder, file_name)

                # Extract the year from the file name
                year = extract_year(file_name)

                # Open the Excel file and look for the necessary sheets
                xl = pd.ExcelFile(file_path)
                
                # Check for Ranked/Select Measure Data sheet
                sheet_name = None
                for option in sheet_options:
                    if option in xl.sheet_names:
                        sheet_name = option
                        break
                
                # Check for Additional Measure Data sheet
                additional_sheet_name = additional_sheet if additional_sheet in xl.sheet_names else None

                # If we found a Ranked/Select Measure Data sheet
                if sheet_name:
                    # Load the specific sheet
                    df = xl.parse(sheet_name)

                    # First three columns should be FIPS, State, and County. Data starts at the third row.
                    # Skip first two rows and get only relevant columns
                    df_filtered = df.iloc[2:, :3].copy()

                    # Rename columns (second row of Excel sheet) to FIPS, State, County
                    df_filtered.columns = ['FIPS', 'State', 'County']

                    # Add the year as a new column
                    df_filtered['Year'] = year

                    # Loop through selected_columns and copy the relevant columns
                    for col_name in selected_columns:
                        found_col = None
                        for col_idx, header in enumerate(df.iloc[0]):
                            # if isinstance(header, str) and col_name in header:
                            #     found_col = col_idx
                            #     break
                            if col_name == 'Mental Health Provider Rate' and ('Mental Health Provider Rate' in header or 'MHP Rate' in header):
                                found_col = col_idx
                                break
                            elif col_name in header:  # For other columns, just match the name directly
                                found_col = col_idx
                                break
                        
                        if found_col is not None:
                            # # Add the column to the filtered dataframe
                            # df_filtered[col_name] = df.iloc[2:, found_col].values
                            # Add the column to the filtered dataframe under the original 'Mental Health Provider Rate' name
                            if col_name == 'Mental Health Provider Rate':
                                df_filtered['Mental Health Provider Rate'] = df.iloc[2:, found_col].values
                            else:
                                df_filtered[col_name] = df.iloc[2:, found_col].values
                        else:
                            df_filtered[col_name] = ""
                            print(f"Warning: Could not find '{col_name}' in {file_name}")
                
                # If we found the Additional Measure Data sheet
                if additional_sheet_name:
                    # Load the Additional Measure Data sheet
                    df_additional = xl.parse(additional_sheet_name)

                    # Loop through additional_columns and copy the relevant columns
                    for col_name in additional_columns:
                        found_col = None
                        for col_idx, header in enumerate(df_additional.iloc[0]):
                            if isinstance(header, str) and col_name in header:
                                found_col = col_idx
                                break
                        
                        if found_col is not None:
                            # Add the column to the filtered dataframe
                            if col_name in selected_columns:
                                col_name = col_name + " 2"
                            df_filtered[col_name] = df_additional.iloc[2:, found_col].values
                        else:
                            df_filtered[col_name] = ""
                            print(f"Warning: Could not find '{col_name}' in {file_name}")

                # Drop rows where all of FIPS, State, and County are empty
                df_filtered = df_filtered.dropna(subset=['FIPS', 'State', 'County'], how='all')

                # If it's the first file, write both headers and data to CSV
                if not headers_added:
                    df_filtered.to_csv(output_csv, mode='w', index=False)
                    headers_added = True  # Headers added after first file
                else:
                    # Append the data without headers for subsequent files
                    df_filtered.to_csv(output_csv, mode='a', index=False, header=False)

                print(f"Processed: {file_name} (Sheet: {sheet_name}, Additional: {additional_sheet_name})")

print(f"Data collection complete. CSV saved at: {output_csv}")
