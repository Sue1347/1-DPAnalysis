import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
import os

# Example DataFrame
# df = pd.DataFrame({
#     "time": [5, 10, 15, 20, 25],
#     "event": [1, 1, 0, 1, 0],
#     "SUVmaxBM": [0.2, 0.4, 0.6, 0.8, 1.0]
# })

file_path = "/home/kevin/Downloads/Datasets/DiagProgAnalysis"
file_name = "simple-dataset-v2.csv"
df = pd.read_csv(os.path.join(file_path,file_name))

df = df[df["Stade"]=="Pre-CAR-T-CELLS"]
# df_post = df[df["Stade"]=="Post-CAR-T-CELLS"]

# organize the data into 
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
df.rename(columns=rename_dict, inplace=True)
df["event"] = df["P ou R"].notna()


df = df.fillna(0.0001)
# Create structured array for survival data
survival_data = Surv.from_dataframe(event="event", time="PFS", data=df)



# Define time points of interest
time_points = np.linspace(1, max(df["PFS"])-1, max(df["PFS"])-2)  # Create time points within the range
# print(time_points)


var_list = ["Age", "SUVmaxBM", "SUVmaxFL", "ADCMeanBMI", "ADCMeanFL"]


for var in var_list:
    print("Var: ",var)
    # print(len(survival_data), len(df[var]))
    # print(df[var])

    # Compute time-dependent AUC and ROC metrics
    auc_times, auc_scores = cumulative_dynamic_auc(
        survival_data, 
        survival_data, 
        df[var], 
        times=time_points
    )
    print("auc_scores: \n",auc_scores)
    print("auc_times: \n",auc_times)

    print(df["event"], df[var])
    # Calculate traditional ROC metrics
    fpr, tpr, thresholds = roc_curve(df["event"], df[var])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

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
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

    # Calculate AUC using scikit-learn's roc_auc_score
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(df["event"], df[var])
    print("auc_score for scikit-learn's roc_auc_score: ", auc_score)

    # Plot ROC curve
    plot_roc_curve(fpr, tpr, auc_score, optimal_idx)

    """Plot time-dependent ROC"""
    # plt.figure(figsize=(10, 6))
    # for i, time in enumerate(time_points):
    #     plt.plot(
    #         [0, 1], [0, 1], "k--", label="Random Guess" if i == 0 else ""
    #     )  # Diagonal line
    #     plt.scatter(1 - auc_times[i], auc_scores[i], label=f"Time {time:.1f}")
    # plt.xlabel("1 - Specificity (False Positive Rate)")
    # plt.ylabel("Sensitivity (True Positive Rate)")
    # plt.title("Time-Dependent ROC Curve")
    # plt.legend()
    # plt.grid()
    # plt.show()
