import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi



file_path = "/home/kevin/Downloads/Datasets/DiagProgAnalysis"
file_name = "simple-dataset-SMM.csv"


def make_diagnosis_tables(df_func, column_list):
    """
    make the statistical table of diagnosis rate, return a table
    """
    type_list = ["PET", "MRI"] # because type is more fixed, so I do not use a variable for it

    res = np.zeros((3,len(column_list))) # for PET, MR, PET/MR; int(len(column_list))
    res_percent = np.zeros((3,len(column_list)))
    for i in range(len(type_list)):
        n = df_func["Stade"].count()
        for j in range(len(column_list)):
            res[i,j] = (df_func[type_list[i]+" "+column_list[j]]==1).sum()
            res_percent[i,j] = round(res[i,j]/n * 100, 2)
            
    for j in range(len(column_list)):
        res[2,j] = ((df_func[type_list[0]+" "+column_list[j]]==1) | (df_func[type_list[1]+" "+column_list[j]]==1)).sum()
        res_percent[2,j] = round(res[2,j]/n * 100, 2)
    return res, res_percent

rename_dict = {
    "TEP Global": "PET Global",
    "MRI global": "MRI Global",
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

df = pd.read_csv(os.path.join(file_path,file_name))
mri_idx = df.columns.get_loc('MRI global')

df.rename(columns=rename_dict, inplace=True)
print(df)

# clean the Nan data: 
df = df[df["PET Global"].notna()]


# print(df["Ig"].value_counts(normalize=True))
print("SUVmaxBM: ",df["SUVmaxBM"].mean(),df["SUVmaxBM"].std())
# print(df["SUVmaxBM"])

print(df["ADCMeanBMI"].mean(),df["ADCMeanBMI"].std())
# print(df_test.head())

# Keep the youngest age for each combination of "nom" and "prenom"
df_youngest = df.loc[df.groupby(["NOM", "Prenom"])["Age"].idxmin()]

# View the result
df_youngest.to_csv(os.path.join(file_path,"simple-dataset-SMM-clean.csv"), index=False)

# print(df.groupby(["NOM", "Prenom"]))