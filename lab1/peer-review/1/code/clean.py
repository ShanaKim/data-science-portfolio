import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from map_data import rename_tbi

def infer_missing_outcome(row):
    """
    unction to infer missing outcome values based on available data in the row.
    If all non-missing values in the row are the same, assigns that value.
    Otherwise, returns 'Unknown'
    """
    outcome = 'Unknown'
    not_missing = [data for data in row if data != np.nan] # Filter out NaN values
    if len(not_missing) == len(row) and not_missing.count(not_missing[0]) == len(not_missing):
        outcome = not_missing[0]  # Assign the common value if all are the same
    return outcome

def process(data):
    """
    function to clean and preprocess the dataset.
    - Fills missing values in 'PosIntFinal' using inferred outcomes.
    - Filters data to keep only rows where 'GCSTotal' >= 14.
    - Renames TBI-related columns.
    - Drops unnecessary columns.
    - Fills missing categorical values with the mode.
    - Replaces 'Unknown' and 'Unclear' with 0.
    """
    # Define outcome-related columns

    outcome_vars = ['HospHeadPosCT', 'Intub24Head', 'Neurosurgery', 'DeathTBI']

     # Fill missing values in 'PosIntFinal' using inferred outcomes

    data.loc[data['PosIntFinal'].isna(), 'PosIntFinal'] = data[data['PosIntFinal'].isna()][outcome_vars].apply(infer_missing_outcome, axis=1)
    data = data[data['PosIntFinal'] != 'Unknown']

    # Keep only rows where 'GCSTotal' >= 14
    data = data.loc[data['GCSTotal'] >= 14, :]

     # Rename TBI-related columns
    data = rename_tbi(data)

    # Define lists of columns to drop
    tbi_on_ct = [f'Finding{i}' for i in range(1, 15)] + [f'Finding{i}' for i in range(20, 24)] + ['PosCT']
    ctvars = ['CTForm1', 'IndAge', 'IndAmnesia', 'IndAMS', 'IndClinSFx',
                    'IndHA', 'IndHema', 'IndLOC', 'IndMech', 'IndNeuroD',
                    'IndRqstMD', 'IndRqstParent', 'IndRqstTrauma', 'IndSeiz', 'IndVomit',
                    'IndXraySFx', 'IndOth', 'CTSed', 'CTSedAgitate', 'CTSedAge', 
                    'CTSedRqst', 'CTSedOth']
    outcome_vars = ['HospHeadPosCT', 'DeathTBI', 'HospHead', 'Intub24Head', 'Neurosurgery']
    other_vars = ['CTDone', 'EDCT', 'EDDisposition', 'Observed']
    all_cols_to_drop = tbi_on_ct + outcome_vars + other_vars + ctvars

    data = data.drop(columns=all_cols_to_drop)
    # Drop additional personal and demographic columns
    data = data.drop(columns=['PatNum', 'EmplType', 'Certification', 'Ethnicity', 'Race', 'Gender', 'Dizzy',
                        'AgeInMonth', 'AgeinYears', 'GCSTotal', 'GCSGroup'])
    
    # Fill missing categorical values with the mode
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "O" else x)

    # Replace categorical values 'Unknown' and 'Unclear' with 0
    data = data.replace("Unknown", 0, inplace = True)
    data = data.replace("Unclear", 0, inplace = True)

    return data

if __name__ == "__main__":
    dir_path = "TBI PUD 10-08-2013.csv"  
    data = pd.read_csv(dir_path) 
    data = process(data)  
    print(data.head())  


