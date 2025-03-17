# mental-health-determinants

# Workflow

## Overview
This repository contains a sequence of scripts for processing and analyzing data related to mental health and socioeconomic attributes at the county level. Below is the step-by-step execution guide.

## Execution Steps

1. **Prepare Raw Data Files**
   - Execute `1. remove_version_no.py` to remove version numbers from raw data file names.
   - The modified files will be saved separately (without altering the original source files).
   - **Note:** 2024 FL has been manually handled.

2. **Check File Naming Format**
   - Run `2. file_name_format_check.py` to validate if the naming format is correct.
   - This step is crucial as subsequent processing relies on correct naming conventions.

3. **Process Mental Health Attributes**
   - Execute `3. data_parser_for_mental_health_attr.py` to extract mental health-related data.
   - The processed data will be saved as `processed_county_data_mental_health_attr.csv`.
   - **Note:** 2018 OHIO had an issue and has been manually addressed.

4. **Process Socioeconomic Attributes**
   - Execute `4. data_parser_for_socioeconomic_attr.py` to extract socioeconomic-related data.
   - The processed data will be saved as `processed_county_data_socioeconomic_attr.csv`.

5. **Merge Processed Data**
   - Execute `5. merge_mental_health_and_socioeconomic_attr.py` to combine the processed files into a single dataset.
   - The output file will be `processed_county_data_mental_health_and_socioeconomic_attr.csv`.

6. **Generate Missing Value Statistics**
   - Execute `6. missing_value_stat.py` to generate `missing_values_stats.csv` with relevant statistics.

7. **Generate Basic Descriptive Statistics**
   - Execute `7. basic-stat.py` to generate `descriptive_statistics.csv`.

8. **Generate Correlation Matrix**
   - Execute `8. correlation.py` to generate the correlation matrix as `correlation_matrix.csv`.

9. **Generate Correlation Heatmaps**
   - Execute `9. correlation_heatmaps.py` to generate the heatmap visualization (not saved by default).

10. **Variance Inflation Factor (VIF) Analysis**
    - Execute `10. VIFs_without_dependents.py` to compute `vif_values_without_dependents.csv`.
    - This excludes dependent variables and identifiers like FIPS code, Year, State, and County.
    - Initially, high VIFs were caused by `# Mental Health Provider` and `Children in Poverty`, so these are excluded.
    - A separate run including these variables is recommended for comparison.

11. **Algorithm Execution**
    - The next steps involve executing machine learning algorithms.
    - Each algorithm is located in its own folder and can be executed manually.
    - A central script is available to invoke all algorithms at once.

## Notes
- Ensure all scripts are executed in sequence to maintain data integrity.
- Adjustments may be required based on manual interventions and findings from earlier steps.
- Results will be saved as CSV files in the respective directories for further analysis.
