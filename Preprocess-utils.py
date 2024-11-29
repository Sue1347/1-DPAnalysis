import pandas as pd
import os
import numpy

file_path = "/home/kevin/Downloads/Datasets/DiagProgAnalysis"
file_name = "simple-dataset.csv"

def read_my_csv(file_path):
    """
    Reads CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    return df

def rename_columns_by_index(df, new_names):
    """
    Renames the columns of a DataFrame using their positional indices.
    """
    # Convert column indices to names and create a rename map
    rename_dict = {df.columns[idx]: new_name for idx, new_name in new_names.items()}
    
    # Rename columns
    df.rename(columns=rename_dict, inplace=True)
    return df

rename_dict = {
    "BMI": "PET BMI",
    "FL": "PET FL",
    "Number FLs": "PET Number FLs",
    "EMD": "PET EMD",
    "Number EMD": "PET Number EMD",
    "PMD": "PET PMD",
    "Number PMD": "PET Number PMD",
    "BMI.1": "MRI BMI",
    "FL.1": "MRI FL",
    "Number FLs.1": "MRI Number FLs",
    "EMD.1": "MRI EMD",
    "Number EMD.1": "MRI Number EMD",
    "PMD.1": "MRI PMD",
    "Number PMD.1": "MRI Number PMD",
}
diagnosis_elements = [
    "Stade",
    "PET BMI",
    "PET FL",
    "PET Number FLs",
    "PET EMD",
    "PET Number EMD",
    "PET PMD",
    "PET Number EMD",
    "MRI BMI",
    "MRI FL",
    "MRI Number FLs",
    "MRI EMD",
    "MRI Number EMD",
    "MRI PMD",
    "MRI Number EMD",
    ]
column_list = ["FL", "BMI", "EMD", "PMD"]


df = read_my_csv(os.path.join(file_path,file_name))
mri_idx = df.columns.get_loc('MRI global')

df.rename(columns=rename_dict, inplace=True)
df1 = df[diagnosis_elements]

df_pre = df1[df1["Stade"]=="Pre-CAR-T-CELLS"]
df_post = df1[df1["Stade"]=="Post-CAR-T-CELLS"]
print(df1["Stade"].count(), df_pre["Stade"].count(), df_post["Stade"].count())
assert(df1["Stade"].count() != df_pre["Stade"].count() + df_post["Stade"].count())

def make_tables(df, column_list):
    """
    """

    res = ""
    return res