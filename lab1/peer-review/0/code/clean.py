import pandas as pd
import numpy as np

def clean_data(df_data):
    df_data = clean_seiz(df_data)
    df_data = clean_vomit(df_data)

    # Replace all 92 with NaN values
    df_data.replace(92, np.nan, inplace=True)

    # Clean labels
    ## Define relevant columns
    columns_to_keep = ["PatNum", "PosCT", "DeathTBI", "HospHeadPosCT", "PosIntFinal"]

    ## Filter dataset to keep only relevant columns
    df_filtered = df_data[columns_to_keep]

    ## Create a DataFrame with only NaN values in 'PosIntFinal'
    nan_df_class = df_filtered[df_filtered["PosIntFinal"].isna()]
    
    ## Create temporary df where PosCT is 1
    nan_df_class = df_filtered[df_filtered["PosIntFinal"].isna()]
    nan_df_class = nan_df_class[nan_df_class["PosCT"] == 1]

    ## Set PosIntFinal to 1 for these rows
    nan_df_class["PosIntFinal"] = 1

    ## Update df_data
    df_data.update(nan_df_class)

    ## Drop rows left with NaN in df_data
    df_data = df_data.dropna(subset=["PosIntFinal"])

    return df_data

def clean_seiz(df_data):
    """
    Cleans and enhances seizure-related data by:
    
    1. Creating a comprehensive seizure indicator (HadSeiz).
    2. Identifying contradictions in medical records.
    3. Standardizing missing values and inconsistencies.

    Parameters:
    df_data (pd.DataFrame): The input DataFrame containing seizure-related columns.

    Returns:
    pd.DataFrame: Updated DataFrame with improved seizure indicators integrated.
    """
    # Initialize HadSeiz with NaN
    df_data["HadSeiz"] = np.nan

    def update_had_seiz(row):
        """
        Updates HadSeiz based on IndSeiz, Seiz, SeizOccur, and SeizLen.
        """
        # --- CHECK 1: IndSeiz ---
        if pd.isna(row["HadSeiz"]):  
            if row["IndSeiz"] == 0:
                row["HadSeiz"] = 0
            elif row["IndSeiz"] == 1:
                row["HadSeiz"] = 1
        elif row["HadSeiz"] == 0 and row["IndSeiz"] == 1:
            row["HadSeiz"] = "contradiction"

        # --- CHECK 2: Seiz ---
        if pd.isna(row["HadSeiz"]):
            if row["Seiz"] == 0:
                row["HadSeiz"] = 0
            elif row["Seiz"] == 1:
                row["HadSeiz"] = 1
        elif row["HadSeiz"] == 0 and row["Seiz"] == 1:
            row["HadSeiz"] = "contradiction"

        # --- CHECK 3: SeizOccur ---
        if pd.isna(row["HadSeiz"]):
            if row["SeizOccur"] in [1, 2, 3]:
                row["HadSeiz"] = 1
        elif row["HadSeiz"] == 0 and row["SeizOccur"] in [1, 2, 3]:
            row["HadSeiz"] = "contradiction"

        # --- CHECK 4: SeizLen ---
        if pd.isna(row["HadSeiz"]):
            if row["SeizLen"] in [1, 2, 3, 4]:
                row["HadSeiz"] = 1
        elif row["HadSeiz"] == 0 and row["SeizLen"] in [1, 2, 3, 4]:
            row["HadSeiz"] = "contradiction"

        return row
    
    # Apply feature construction function directly on df_data
    df_data = df_data.apply(update_had_seiz, axis=1)

    # Update rows where HadSeiz == 0 to ensure IndSeiz, Seiz, SeizOccur, and SeizLen are also 0
    df_data.loc[df_data["HadSeiz"] == 0, ["Seiz", "SeizOccur", "SeizLen"]] = 0

    # Replace all 92 values with NaN
    df_data.replace(92, np.nan, inplace=True)

    return df_data



def clean_vomit(df_data):
    """
    Cleans and enhances vomiting-related data by:
    
    1. Selecting only relevant columns.
    2. Constructing indicators for evidence of vomiting.
    3. Standardizing missing values and inconsistencies.
    4. Updating existing vomiting-related columns based on logical deductions.
    
    Parameters:
    df_data (pd.DataFrame): The input DataFrame containing vomiting-related columns.
    
    Returns:
    pd.DataFrame: Updated DataFrame with improved vomiting indicators integrated.
    """
    # Select relevant columns
    selected_columns = ["PatNum", "VomitNbr", "VomitStart", "IndVomit", "VomitLast"]

    # Initialize new features
    df_data["VomitEvidenceCount"] = 0
    df_data["VomitLackOfEvidenceCount"] = 0

    # Increase vomitEvidenceCount based on conditions
    df_data["VomitEvidenceCount"] += df_data["VomitNbr"].isin([1, 2, 3]).astype(int)
    df_data["VomitEvidenceCount"] += df_data["VomitStart"].isin([1, 2, 3, 4]).astype(int)
    df_data["VomitEvidenceCount"] += df_data["VomitLast"].isin([1, 2, 3]).astype(int)
    df_data["VomitEvidenceCount"] += (df_data["IndVomit"] == 1).astype(int)

    # Increase vLackOfEvidence based on conditions
    df_data["VomitLackOfEvidenceCount"] += df_data["VomitNbr"].isin([92]) | pd.isna(df_data["VomitNbr"])
    df_data["VomitLackOfEvidenceCount"] += df_data["VomitStart"].isin([92]) | pd.isna(df_data["VomitStart"])
    df_data["VomitLackOfEvidenceCount"] += df_data["VomitLast"].isin([92]) | pd.isna(df_data["VomitLast"])
    df_data["VomitLackOfEvidenceCount"] += df_data["IndVomit"].isin([92]) | pd.isna(df_data["IndVomit"])

    # Define the columns to modify
    vomit_columns = ["VomitNbr", "VomitStart", "IndVomit", "VomitLast"]

    # Replace NaN or 92 with np.nan in specified columns
    df_data[vomit_columns] = df_data[vomit_columns].applymap(
        lambda x: np.nan if pd.isna(x) or x == 92 else x
    )

    return df_data