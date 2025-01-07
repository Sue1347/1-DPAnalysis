import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


file_path = "/home/kevin/Downloads/Datasets/DiagProgAnalysis"
file_name = "simple-dataset-v2.csv"

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
    ...I really like this exmaples so I leave it here in case I want to learn...
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
diagnosis_elements = [
    "Stade",
    "PET Global",
    "PET BMI",
    "PET FL",
    "PET Number FLs",
    "PET EMD",
    "PET Number EMD",
    "PET PMD",
    "PET Number EMD",
    "MRI Global",
    "MRI BMI",
    "MRI FL",
    "MRI Number FLs",
    "MRI EMD",
    "MRI Number EMD",
    "MRI PMD",
    "MRI Number EMD",
    ]
column_list = ["Global","FL", "BMI", "EMD", "PMD"] #


df = read_my_csv(os.path.join(file_path,file_name))
mri_idx = df.columns.get_loc('MRI global')

df.rename(columns=rename_dict, inplace=True)
# print(df["Ig"].value_counts(normalize=True))
print("SUVmaxBM: ",df["SUVmaxBM"].mean(),df["SUVmaxBM"].std())
# print(df["SUVmaxBM"])
# print(df.head())
print(df["ADCMeanBMI"].mean(),df["ADCMeanBMI"].std())
# print(df_test.head())

##########################################################################
"""the basic data of the dataset"""
# df.loc[df['P ou R'].isna(), 'PFS'] = np.nan
# df.loc[df['Deces'].isna(), 'OS'] = np.nan
# df.loc[df['PET BMI']== 0, 'SUVmaxBM'] = np.nan
# df.loc[df['MRI BMI']== 0, 'ADCMeanBMI'] = np.nan

df_pre = df[df["Stade"]=="Pre-CAR-T-CELLS"]
df_post = df[df["Stade"]=="Post-CAR-T-CELLS"]

print(df.columns)

# continuous_list = ['Age', 'Ratio k/l', 'PFS', 'OS', 'ISS',
#         'SUVmaxBM',  'SUVmaxFL', 'SUVmaxEMD', 'SUVmaxPMD', 
#         'FF BM', 'FF FL', 
#         'ADCMeanBMI', 'ADCMeanFL', 'ADCMean EMD', 'ADCMean PMD'] 
# #  'T(4 14)', 'del 17p', 'duplication 1q', 'R-ISS', 'Traitement', 

# df_now = df_post[continuous_list]

# for i in df_now.columns:
    
#     print(i,df_now[i].mean(), df_now[i].std())

""" Fisher's Exact Test for diagnosis performance"""
def fishers_exact(df,var_list):
    import scipy.stats as stats
    for var in var_list:
        print(var)
        a = df["PET "+var].value_counts()
        b = df["MRI "+var].value_counts()
        # print(np.asarray(a))
        # print(np.asarray(b))
        data = np.stack((a, b),axis=1) #[::-1]
        print(data)
        
        # performing fishers exact test on the data 
        # if "MRI "+ var == "MRI EMD": continue
        odd_ratio, p_value = stats.fisher_exact(data, alternative="less") 
        print('odd ratio is : ' + str(odd_ratio)) 
        print('p_value is : ' + str(p_value)) 

# print("pre treatment fisher's exact test: ")
# fishers_exact(df_pre, column_list)
# print("post treatment fisher's exact test: ")
# fishers_exact(df_post, column_list) # no EMD



"""Create Box plots for characteristics"""

# Convert the DataFrame to long format for Seaborn
# df_long = df_now[['PFS', 'OS']].melt(var_name='Category', value_name='Value')

# Create the box plot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Category', y='Value', data=df_long)
# plt.title("Box Plot for PFS and OS (post treatment)")
# # plt.xlabel("Categories")
# plt.ylabel("Time")
# plt.grid(axis='y')  # Add a grid for better readability
# plt.show()

"""Percentage of missing Characteristics"""

# nan_percentage = df.isna().mean() * 100
# count_n50 = 0
# for column, percentage in nan_percentage.items():
#     if percentage >= 50: count_n50+=1
#     print(f"{column}: {percentage:.2f}%")
# print(count_n50, " is larger than 50%.")
##################################################################################


####################################################################################
"""choose the small area of the data to do the calculation of diagnosis performance test"""
# Post=='Reponse', 'TEP Global','PET BMI', 'PET FL','PET EMD','PET PMD','MRI global', 'MRI BMI','MRI FL','MRI EMD','MRI PMD',

# if(df1["Stade"].count() != df_pre["Stade"].count() + df_post["Stade"].count()): 
#     print("The number is not correct")
    # print(df1["Stade"].count(), df_pre["Stade"].count(), df_post["Stade"].count())
# print(df_post["Reponse"].value_counts(normalize=False)) # True
# res_1, res_percent1 = make_diagnosis_tables(df_pre, column_list)
# res_2, res_percent2 = make_diagnosis_tables(df_post[df_post["Reponse"]=="CR"], column_list)
# res, res_percent3 = make_diagnosis_tables(df1, column_list)

# arr = np.concatenate((res_1,res_2))
# print(arr)

# save it to csv files
# save_tables_diagnosis = "tables_diagnosis.csv"
# np.savetxt(os.path.join(file_path,save_tables_diagnosis), arr, delimiter=',')
#####################################################################################

#####################################################################################
"""To draw a Kaplan-Meier Plot based on the current data"""
def Kaplan_Meier_plot(df_func):
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator

    
    df_func = df_func[["P ou R", "PFS"]]
    df_func["Event"] = df_func["P ou R"].notna()
    print(df_func.head())
    x, y, conf_int = kaplan_meier_estimator(df_func["Event"], df_func["PFS"], conf_type="log-log")
    print(x,y)

    # Add censored points to the plot
    points_x = []
    points_y = []
    for e in range(df_func["Event"].count()):
        if df_func.iloc[e]["Event"]== False:
            k = df_func.iloc[e]["PFS"]
            points_x.append(k)
            index = np.searchsorted(x, k, side='right') -1
            # print(y[index])
            points_y.append(y[index])
    plt.scatter(points_x, points_y, color='plum', marker="+", zorder=5) #label='Points', 

    
    plt.step(x, y, where="post")
    plt.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.title('Kaplan-Meier Plot')
    plt.xlabel('PFS(Month)')
    plt.ylabel('Percentage')
    # plt.legend(loc="best")
    plt.show()
    # plt.savefig(os.path.join(file_path,"Kaplan-Meier-all.png")) #? why
    return

def Kaplan_Meier_two_plot(df_func):
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator

    df_func["Event"] = df_func["P ou R"].notna()
    for treatment_type in ("Pre-CAR-T-CELLS", "Post-CAR-T-CELLS"):
        mask_treat = df_func["Stade"] == treatment_type
        time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
            df_func["Event"][mask_treat],
            df_func["PFS"][mask_treat],
            conf_type="log-log",
        )
        plt.step(time_treatment, survival_prob_treatment, where="post", label=f"Treatment = {treatment_type}")
        plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")

    plt.ylim(0, 1)
    plt.legend(loc="best")
    
    plt.title('Kaplan-Meier Plot')
    plt.xlabel('PFS(Month)')
    plt.ylabel('Percentage')
    plt.show()
    # plt.savefig(os.path.join(file_path,"Kaplan-Meier-all.png")) #? why
    return

# Kaplan_Meier_plot(df_post)
#################################################################################


###################################################################################
"""the Cox PH survival analysis"""

# transform all data into numeric values
from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def make_one_hot_data(df_func, column_list):
    """Given some list of characteristics that can eveluate the survival """
    # print(column_list)
    # print(df_func.head())
    
    df_func = df_func[column_list]
    df_func = df_func.fillna(0)
    onehot_list = ["Stade", "Ig"]
    # print(df_func.head())
    # print(df_func["Stade"].value_counts())
    # print(df_func["Ig"].value_counts())

    for i in range(len(onehot_list)):
        type_list = list(df_func[onehot_list[i]].unique())
        for j in range(len(type_list)):
            if type_list[j] == 0:
                df_func[onehot_list[i]+"=NaN"]= df_func[onehot_list[i]] == type_list[j]
                df_func[onehot_list[i]+"=NaN"] = df_func[onehot_list[i]+"=NaN"].astype(int)
                column_list.append(onehot_list[i]+"=NaN")
                continue
            # print(onehot_list[i],"=",type_list[j])
            df_func[onehot_list[i]+"="+type_list[j]]= df_func[onehot_list[i]] == type_list[j]
            df_func[onehot_list[i]+"="+type_list[j]] = df_func[onehot_list[i]+"="+type_list[j]].astype(int)
            column_list.append(onehot_list[i]+"="+type_list[j])
        column_list.remove(onehot_list[i])
    
    # print(df_func.head())
    # data_x_numeric = OneHotEncoder().fit_transform(df_func)
    df_func = df_func.drop(columns = onehot_list)
    print(df_func.columns)

    return df_func,column_list

# cox_pre_column_list = ["Age", "Stade", "Ig", "SUVmaxBM", "SUVmaxFL", "ADCMeanBMI", "ADCMeanFL"]
# df_onehot, cox_column_list = make_one_hot_data(df, cox_pre_column_list)

# cox_pre_column_list = ["Age", "Stade", "Ig", "SUVmaxBM", "SUVmaxFL", "ADCMeanBMI", "ADCMeanFL"]
# df_onehot_test, cox_column_list_test = make_one_hot_data(df_test, cox_pre_column_list)
# print("dataframe onehot test:",df_onehot_test)

df_pfs = df_pre[["PFS"]].astype(float)
df_pfs["Event"] = df_pre["P ou R"].notna().astype(int)
df_pfs = df_pfs[["Event","PFS"]]

cox_column_list = ['Age', 
       'PET Global', 'PET BMI', 'SUVmaxBM', 'PET FL', 'SUVmaxFL',
       'PET PMD', 'SUVmaxPMD', 
       'MRI Global', 'MRI BMI', 'ADCMeanBMI', 'MRI FL','ADCMeanFL',
       'MRI PMD', 'ADCMean PMD',] 
# 'PET EMD', 'SUVmaxEMD',
#  'MRI EMD', 'ADCMean EMD',
# 'Ratio k/l', 'ISS', 'FF BM', 'FF FL', 
# maybe because of the high values ValueError: LAPACK reported an illegal value in 5-th argument.
# print(df_pre[cox_column_list].fillna(0).head())

df_pre = df_pre[cox_column_list].fillna(0)

# Columns to standardize
# columns_to_standardize = ["Age", "SUVmaxBM", "SUVmaxFL", 'SUVmaxPMD', "ADCMeanBMI", "ADCMeanFL", 'ADCMean PMD'] # 'SUVmaxEMD','ADCMean EMD',
# Initialize the StandardScaler
# scaler = MinMaxScaler()
# Standardize only the selected columns
# df_pre[columns_to_standardize] = scaler.fit_transform(df_pre[columns_to_standardize])

df_pre = (df_pre-df_pre.min())/(df_pre.max()-df_pre.min())

# print(df_pre.head()) # it is the same, using minmax scaler from sklearn, or calculate by myself

df_test = df_pre.iloc[5:10]


def cox_PH_model(df_onehot, df_pfs, df_onehot_test):

    set_config(display="text")  # displays text representation of estimators
    
    # Define the structured dtype
    dtype = [('Status', '?'), ('Survival_in_days', '<f8')]  # '?' for boolean, '<f8' for float

    # Convert DataFrame to structured array
    structured_array = np.array(list(df_pfs.itertuples(index=False, name=None)), dtype=dtype)
    # print(structured_array)

    estimator = CoxPHSurvivalAnalysis()
    
    estimator.fit(df_onehot, structured_array)
    print("C-index",estimator.score(df_onehot, structured_array))

    print("Hazard Ratio: \n",pd.Series(np.exp(estimator.coef_), index=df_onehot.columns))

    # print("\n",)

    """show what's influence best"""
    n_features = df_onehot.values.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = df_onehot.values[:, j : j + 1]
        m.fit(Xj, structured_array)
        scores[j] = m.score(Xj, structured_array)

    print("scores: \n",pd.Series(scores, index=df_onehot.columns).sort_values(ascending=False))

    """make a test""" 
    # Find columns in B but not in A
    missing_columns = set(df_onehot.columns) - set(df_onehot_test.columns)
    # Add missing columns to A with value 0
    for col in missing_columns:
        df_onehot_test[col] = 0
    # Ensure column order matches B
    df_onehot_test = df_onehot_test[df_onehot.columns]

    pred_surv = estimator.predict_survival_function(df_onehot_test)

    time_points = np.arange(1, 41)
    for i, surv_func in enumerate(pred_surv):
        plt.step(time_points, surv_func(time_points), where="post", label=f"Sample {i + 1}")
    plt.ylabel(r"probability of survival") # $\hat{S}(t)$
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    plt.show()
    
    return 

def RandomForest_model(df_onehot, df_pfs, df_onehot_test):
    dtype = [('Status', '?'), ('Survival_in_days', '<f8')]  # '?' for boolean, '<f8' for float

    # Convert DataFrame to structured array
    structured_array = np.array(list(df_pfs.itertuples(index=False, name=None)), dtype=dtype)
    # print(structured_array)

    # estimator = CoxPHSurvivalAnalysis()
    estimator = RandomSurvivalForest(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=20)
    
    
    estimator.fit(df_onehot, structured_array)
    print("estimator score",estimator.score(df_onehot, structured_array))
    

    # Find columns in B but not in A
    missing_columns = set(df_onehot.columns) - set(df_onehot_test.columns)
    # Add missing columns to A with value 0
    for col in missing_columns:
        df_onehot_test[col] = 0
    # Ensure column order matches B
    df_onehot_test = df_onehot_test[df_onehot.columns]
    
    surv = estimator.predict_survival_function(df_onehot_test, return_array=True)

    for i, s in enumerate(surv):
        plt.step(estimator.unique_times_, s, where="post", label=f"Sample {i + 1}")
    plt.ylabel("Survival probability")
    plt.xlabel("Time in months")
    plt.legend()
    # plt.grid(True)
    plt.show()

    return
 
# cox_PH_model(df_pre, df_pfs, df_test)
# RandomForest_model(df_pre_norm, df_pfs, df_test)


"""save it to csv files"""
# print(res)
# save_tables_cox = "tables_cox.csv"
# np.savetxt(os.path.join(file_path,save_tables_cox), res, delimiter=',')

