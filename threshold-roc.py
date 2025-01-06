import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
import os
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score
from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis

import matplotlib.pyplot as plt


file_path = "/home/kevin/Downloads/Datasets/DiagProgAnalysis"
file_name = "simple-dataset-v2.csv"
df = pd.read_csv(os.path.join(file_path,file_name))

df = df[df["Stade"]=="Pre-CAR-T-CELLS"] # Pre
# df_post = df[df["Stade"]=="Post-CAR-T-CELLS"]

# organize the data into 
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
df.rename(columns=rename_dict, inplace=True)
df["event"] = df["P ou R"].notna()


df = df.fillna(0.0001)
# Create structured array for survival data
survival_data = Surv.from_dataframe(event="event", time="PFS", data=df)


# Define time points of interest
time_points = np.linspace(1, max(df["PFS"])-1, max(df["PFS"])-2)  # Create time points within the range
# print(time_points)


var_list = ["Age", "SUVmaxBM", "SUVmaxFL", "ADCMeanBMI", "ADCMeanFL"]
cox_column_list = ["Age", "SUVmaxBM", "SUVmaxFL",'SUVmaxEMD', 'SUVmaxPMD', "ADCMeanBMI", "ADCMeanFL",'ADCMean EMD', 'ADCMean PMD']
# 'PET EMD',  'PET PMD', 
#  'MRI EMD', 'MRI PMD',
# 'Ratio k/l', 'ISS', 'FF BM', 'FF FL', 
var_list = cox_column_list
thresholds_lists = []


for var in var_list:
    print("Var: ",var)
    # print(len(survival_data), len(df[var]))

    # Compute time-dependent AUC and ROC metrics
    auc_times, auc_scores = cumulative_dynamic_auc(
        survival_data, 
        survival_data, 
        df[var], 
        times=time_points
    )
    print("time average auc_scores: ",auc_scores)
    # print("auc_times: \n",auc_times)

    # Calculate traditional ROC metrics
    fpr, tpr, thresholds = roc_curve(df["event"], df[var])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    thresholds_lists.append(optimal_threshold)

    # Print optimal threshold
    print("Optimal Threshold:", optimal_threshold)

    # Function to plot ROC curve
    def plot_roc_curve(fpr, tpr, auc_score, optimal_idx=None):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        
        # Highlight optimal threshold point
        if optimal_idx is not None:
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label='Optimal Threshold')
            plt.annotate(f"Optimal Threshold\n(FPR={fpr[optimal_idx]:.2f}, TPR={tpr[optimal_idx]:.2f})",
                        (fpr[optimal_idx], tpr[optimal_idx]), 
                        textcoords="offset points", 
                        xytext=(10, -10), 
                        ha='center')
        
        plt.xlabel("False Positive Rate (1-Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve: "+var)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

    # Calculate AUC using scikit-learn's roc_auc_score
    auc_score = roc_auc_score(df["event"], df[var])
    # print("auc_score for scikit-learn's roc_auc_score: ", auc_score) # the same as the above

    # Plot ROC curve
    # plot_roc_curve(fpr, tpr, auc_score, optimal_idx)

    df[var+"_thrshd"] = df[var]>=optimal_threshold

    T1 = df.loc[(df[var] >= optimal_threshold),"PFS"]
    T2 = df.loc[(df[var] < optimal_threshold),"PFS"]
    E1 = df.loc[(df[var] >= optimal_threshold),"event"]
    E2 = df.loc[(df[var] < optimal_threshold),"event"]

    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    # results.print_summary()
    # print("p value: ",results.p_value)        # 
    # print(results.test_statistic) # 

# print(thresholds_lists)

"""Cox PH model based on thresholds"""
df_pfs = df[["PFS"]].astype(float)
df_pfs["Event"] = df["P ou R"].notna().astype(int)
df_pfs = df_pfs[["Event","PFS"]]

cox_column_list = ['Age_thrshd', 
       'PET Global', 'PET BMI','SUVmaxBM_thrshd','PET FL', 'SUVmaxFL_thrshd', 'PET EMD', 'SUVmaxEMD_thrshd', 'PET PMD', 'SUVmaxPMD_thrshd', 
       'MRI Global', 'MRI BMI','ADCMeanBMI_thrshd', 'MRI FL', 'ADCMeanFL_thrshd',
       ] 
#     
#  'MRI EMD', 'ADCMean EMD', 'MRI PMD', 'ADCMean PMD' 
# 'Ratio k/l', 'ISS', 'FF BM', 'FF FL', 
# maybe because of the high values ValueError: LAPACK reported an illegal value in 5-th argument.
# print(df_pre[cox_column_list].fillna(0).head())

df_pre = df[cox_column_list].fillna(0)
# df_pre_norm = (df_pre-df_pre.min())/(df_pre.max()-df_pre.min())

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
    print("estimator score",estimator.score(df_onehot, structured_array))

    print("Coefficients: \n",pd.Series(estimator.coef_, index=df_onehot.columns))

    print("Hazard Ratio: \n",np.exp(estimator.coef_))

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

cox_PH_model(df_pre, df_pfs, df_test)