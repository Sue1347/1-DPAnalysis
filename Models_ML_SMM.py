from Preprocess_utils import Kaplan_Meier_plot

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from sklearn import set_config

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM

set_config(display="text")  # displays text representation of estimators

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


file_path = "/home/kevin/Downloads/Datasets/DiagProgAnalysis"
proj_path = "/home/kevin/Documents/SUN Huajun/Projects/1-DPAnalysis-main"
file_name = "simple-dataset-SMM-clean.csv"

df = pd.read_csv(os.path.join(file_path,file_name))
print(df.columns)
print(df.shape)
# print("SUVmaxBM: ",df["SUVmaxBM"].mean(),df["SUVmaxBM"].std())
# print(df["ADCMeanBMI"].mean(),df["ADCMeanBMI"].std())

variate_list = [
        'Pic', 'Plasmocytose', #'Ratio k/l',
        'PET BMI', 'SUVmaxBM', #'PET FL',
        'MRI BMI', 'ADCMeanBMI',# 'MRI FL',
        'PEI', 'TPEI', 'MITR',
        'PFS','Event',
       ]
######### Create the event and time df
df["PFS"] = df["PFS"].astype(float)
df["PEI"] = df["PEI"].astype(float)
df["Event"] = df["P ou R"].notna().astype(int)

########## Kaplan Meier plot
# Kaplan_Meier_plot(df)


df = df[variate_list].dropna()
print(df.shape)

######### Create the structured array of event and time
df_pfs = df[["Event","PFS"]]
# Define the structured dtype
dtype = [('Status', '?'), ('Survival_in_days', '<f8')]  # '?' for boolean, '<f8' for float
# Convert DataFrame to structured array
data_y = np.array(list(df_pfs.itertuples(index=False, name=None)), dtype=dtype)
# print(structured_array)

df = df.drop(columns=["Event","PFS"])

###### standardize the values
df = (df-df.min())/(df.max()-df.min())

random_state = 20

X_train, X_test, y_train, y_test = train_test_split(df,data_y, test_size=0.10, random_state=random_state)
X_train = df
y_train = data_y


############################################### univariate cox ph model
def fit_and_score_features(X, y):
    X=np.asanyarray(X) # make sure X is numpy array
    n_features = X.shape[1]
    scores = np.empty(n_features)
    # print("n_features :",n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        # print("j: ",j)
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

# scores = fit_and_score_features(X_train,y_train) # df, data_y
# print("scores: \n")
# print(pd.Series(scores, index=df.columns).sort_values(ascending=False))

#################### Find the best combination of multivariate cos ph model
pipe = Pipeline(
    [
        ("encode", OneHotEncoder()),
        ("select", SelectKBest(fit_and_score_features, k=3)),
        ("model", RandomSurvivalForest(n_estimators=200, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=20)), 
    ]
)
# CoxPHSurvivalAnalysis()), 
# RandomSurvivalForest(n_estimators=200, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=20)),
# GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=0)

#estimator = RandomSurvivalForest(n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=20)

from sklearn.model_selection import GridSearchCV, KFold

param_grid = {"select__k": np.arange(1, df.shape[1] + 1)}
cv = KFold(n_splits=3, random_state=1, shuffle=True)
gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv)
gcv.fit(X_train,y_train) # df, data_y

results = pd.DataFrame(gcv.cv_results_).sort_values(by="mean_test_score", ascending=False)
print("results table: \n")
print(results.loc[:, ~results.columns.str.endswith("_time")])
results.to_csv(os.path.join(proj_path,"Results_of_3folds.csv"))
# /home/kevin/Documents/SUN Huajun/Projects/1-DPAnalysis-main/

pipe.set_params(**gcv.best_params_)
pipe.fit(X_train,y_train) 


encoder, transformer, final_estimator = (s[1] for s in pipe.steps)
print(encoder.encoded_columns_[transformer.get_support()]) # show which parameters are being selected

# only for cox PH model
# print(pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()])) # RFS doesn't have coef for hazard ratio
# print("HR: \n",pd.Series(np.exp(final_estimator.coef_), index=encoder.encoded_columns_[transformer.get_support()])) # RFS doesn't have coef for hazard ratio


# only for parameters are all the columns
# print(pd.Series(final_estimator.predict(X_test)))

# ################################### plot the survival function
# surv = final_estimator.predict_survival_function(X_test, return_array=True)
# for i, s in enumerate(surv):
#     plt.step(final_estimator.unique_times_, s, where="post", label=str(i))
# plt.ylabel("Survival probability")
# plt.xlabel("Time in days")
# plt.legend()
# plt.grid(True)
# plt.show()

# ########################### plot the cumulative hazard function
# surv = final_estimator.predict_cumulative_hazard_function(X_test, return_array=True)
# for i, s in enumerate(surv):
#     plt.step(final_estimator.unique_times_, s, where="post", label=str(i))
# plt.ylabel("Cumulative hazard")
# plt.xlabel("Time in days")
# plt.legend()
# plt.grid(True)
# plt.show()