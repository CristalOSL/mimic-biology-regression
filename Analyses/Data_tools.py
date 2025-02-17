import pandas as pd
from sqlalchemy import create_engine
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime
from itertools import permutations
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import backtesting_forecaster_multiseries

def execute_query(sql_query, user, password, host, port, database):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')

    try:
        df = pd.read_sql_query(sql_query, engine)
        return df
    finally:
        engine.dispose()


# Connection parameters
f = open("./.db_login_info.json")
params = json.load(f)[0]
f.close()


def load_json(file):
    """Loads json file as panda dataframe"""
    f = open(f"../data/{file}.json")
    df = pd.DataFrame(json.load(f))
    f.close()
    return df



def sqlify(l):
    """Turns a python list to its representation as a string in sql. Strings are returned in quotes."""
    if isinstance(l[0], str):
        l = ["\'" + elt + "\'" for elt in l]
        return "(" + ", ".join(l) + ")"
    return "(" + ", ".join([str(elt) for elt in l]) + ")"

# Defines used variables
analyses = load_json("changed_analyses")
analysis_codes = analyses["itemid"].to_list()
analysis_codes

ckd_codes = load_json("ckd_codes")


def add_simple_diagnosis(df) :
    """Adds a simplified diagnosis column to a dataframe with specific diagnoses."""
    # Define a mapping for simplified categories
    simplified_mapping = {
    "5856": "Stage 5",
    "40391": "CKD with other diseases",
    "N186": "Stage 5",
    "I120": "CKD with other diseases",
    "5854": "Stage 4",
    "N184": "Stage 4",
    "I132": "CKD with other diseases",
    "5855": "Stage 5",
    "N185": "Stage 5",
    "40311": "CKD with other diseases",
    "40301": "CKD with other diseases",
    "40493": "CKD with other diseases",
    "I1311": "CKD with other diseases",
    "40492": "CKD with other diseases",
    "40403": "CKD with other diseases",
    "40413": "CKD with other diseases"
    }

    # Add a new column for simplified diagnosis
    df['simplified_diagnosis'] = df['icd_code'].map(simplified_mapping)


def separate_values(df, grandeurs):
    """Puts the measurements in distinct columns when they are in the same one."""
    # Separate the columns that need to be pivoted and the ones to remain unchanged
    df_measurements = df[['label', 'valuenum']]  # Columns related to the measurements
    df_other = df.drop(columns=['label', 'valuenum'])  # All other columns to remain unchanged

    # Pivoting the measurement-related DataFrame
    df_separated = df_measurements.pivot(columns='label', values='valuenum')

    # Reorder columns in the correct order (if needed)
    df_separated = df_separated[grandeurs]  # 'grandeurs0' contains the correct order of labels

    # If any measurement is missing, it will be represented as np.nan
    df_separated = df_separated.fillna(np.nan)

    # Merge the pivoted data back with the remaining columns (those not pivoted)
    df_separated = pd.concat([df_separated, df_other], axis=1)

    # Returns the result
    return df_separated



def impute_missing_values(df):
    """
    Imputes missing values in numerical columns of the dataframe using the median strategy.
    
    Parameters:
    df (pd.DataFrame): The input dataframe with missing values.

    Returns:
    pd.DataFrame: The dataframe with imputed values.
    """
    imputer = SimpleImputer(strategy="median")  # Use median imputation
    numeric_cols = df.select_dtypes(include=["number"]).columns  # Select only numeric columns
    
    df_imputed = df.copy()  # Create a copy to avoid modifying the original
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])  # Apply imputation

    return df_imputed