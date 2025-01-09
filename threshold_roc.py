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
from Preprocess_utils import Kaplan_Meier_plot, cox_PH_model

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

df_ori = df
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
    df_ori[var+"_thrshd"] = df_ori[var]>=optimal_threshold

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

# cox_PH_model(df_pre, df_pfs, df_test)

"""depend on thresholds to draw kaplan meier"""
def Kaplan_Meier_two_plot(df_func, thrshd):
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator

    df_func["Event"] = df_func["P ou R"].notna()
    for treatment_type in (True, False):
        mask_treat = df_func[thrshd] == treatment_type
        x, y, conf_int = kaplan_meier_estimator(
            df_func["Event"][mask_treat],
            df_func["PFS"][mask_treat],
            conf_type="log-log",
        )
        plt.step(x, y, where="post", label=f"Larger than {thrshd} is {treatment_type}")
        plt.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post")

        # Add censored points to the plot
        points_x = []
        points_y = []
        for e in range(df_func["Event"].count()):
            if df_func.iloc[e]["Event"]== False and df_func.iloc[e][thrshd]==treatment_type:
                k = df_func.iloc[e]["PFS"]
                points_x.append(k)
                index = np.searchsorted(x, k, side='right') -1
                # print(y[index])
                points_y.append(y[index])
        plt.scatter(points_x, points_y, color='plum', marker="+", zorder=5) #label='Points', 

    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend(loc="best")
    
    plt.title('Kaplan-Meier Plot')
    plt.xlabel('PFS(Month)')
    plt.ylabel('Percentage')
    plt.show()
    # plt.savefig(os.path.join(file_path,"Kaplan-Meier-all.png")) #? why
    return

Kaplan_Meier_two_plot(df_ori, 'SUVmaxEMD_thrshd') # 'Age_thrshd', 'SUVmaxBM_thrshd','SUVmaxFL_thrshd','SUVmaxEMD_thrshd',