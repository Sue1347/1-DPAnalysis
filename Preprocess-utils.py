import pandas as pd
import os
import numpy as np

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

####################################################################################
"""choose the small area of the data to do the calculation of diagnosis performance test"""
# df1 = df[diagnosis_elements]

df_pre = df[df["Stade"]=="Pre-CAR-T-CELLS"]
df_post = df[df["Stade"]=="Post-CAR-T-CELLS"]

# if(df1["Stade"].count() != df_pre["Stade"].count() + df_post["Stade"].count()): 
#     print("The number is not correct")
#     # print(df1["Stade"].count(), df_pre["Stade"].count(), df_post["Stade"].count())

# res, res_percent1 = make_diagnosis_tables(df_pre, column_list)
# res, res_percent2 = make_diagnosis_tables(df_post, column_list)
# res, res_percent3 = make_diagnosis_tables(df1, column_list)

# """save it to csv files"""
# arr = np.concatenate((res_percent1,res_percent2, res_percent3))
# print(arr)
# save_tables_diagnosis = "tables_diagnosis.csv"
# np.savetxt(os.path.join(file_path,save_tables_diagnosis), arr, delimiter=',')
#####################################################################################

"""To draw a Kaplan-Meier Plot based on the current data"""
def Kaplan_Meier_plot(df_func):
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator

    
    df_func = df_func[["P ou R", "PFS"]]
    df_func["Event"] = df_func["P ou R"].notna()
    print(df_func.head())
    x, y, conf_int = kaplan_meier_estimator(df_func["Event"], df_func["PFS"], conf_type="log-log")
    
    plt.step(x, y, where="post")
    plt.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0, 1)
    plt.title('Kaplan-Meier Plot')
    plt.xlabel('PFS(Month)')
    plt.ylabel('Percentage')
    plt.show()
    # plt.savefig(os.path.join(file_path,"Kaplan-Meier-all.png")) #? why
    return

Kaplan_Meier_plot(df_post)