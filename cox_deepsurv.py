# Simply used the pycox notebook for reference

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import os
import pandas as pd

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

np.random.seed(1234)
_ = torch.manual_seed(123)

"""the basic data of the dataset"""

file_path = "/home/kevin/Downloads/Datasets/DiagProgAnalysis"
file_name = "simple-dataset-v2.csv"

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
df.rename(columns=rename_dict, inplace=True)

print(df.columns)
print("SUVmaxBM: ",df["SUVmaxBM"].mean(),df["SUVmaxBM"].std())


cox_column_list = [ 'PFS', 'Event',
        'PET Global','PET BMI', 'SUVmaxBM',# 'SUVmaxFL',# 'PET EMD', 'PET PMD',   'PET FL',
        # 'MRI Global', 'MRI BMI', 'ADCMeanBMI', 'MRI FL', 'ADCMeanFL', 'MRI EMD', 'MRI PMD',
        # 'PET Number FLs', 'PET Number PMD', 'MRI Number FLs', 'MRI Number PMD',  
       ]  
df["PFS"] = df[["PFS"]].astype(float)
df["Event"] = df["P ou R"].notna().astype(int)

df_pre = df[df["Stade"]=="Pre-CAR-T-CELLS"]
df_post = df[df["Stade"]=="Post-CAR-T-CELLS"]

df_train = df_pre[cox_column_list] # metabric.read_df()
df_test = df_train.sample(frac=0.1)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

df_train_ori = metabric.read_df()
print(type(df_train))
print(type(df_train_ori))

print(df_train.dtypes)
print(df_train_ori.dtypes)

########### preprocess the characteristics

cols_standardize = ['SUVmaxBM', # 'SUVmaxFL', # 'ADCMeanBMI', 'ADCMeanFL'
                    ]
cols_leave = ['PET Global', 'PET BMI',# 'PET FL',#  'PET EMD', 'PET PMD',
              #'MRI Global', 'MRI BMI', 'MRI FL', 'MRI EMD', 'MRI PMD'
              ]

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

get_target = lambda df: (df['PFS'].values, df['Event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val

print("x train shape:",x_train.shape)
print("x val shape:",x_val.shape)
print("x test shape:",x_test.shape)

################## create neuronal networks
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

print(net.parameters())
# fit the model
model = CoxPH(net, tt.optim.Adam)

batch_size = 8 #256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()
# plt.show()

print(lrfinder.get_best_lr())

model.optimizer.set_lr(0.01)

# Training the model
epochs = 128 #512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
log.plot()
# plt.show()

print("model.partial_log_likelihood(*val).mean()", model.partial_log_likelihood(*val).mean())

##### prediction
_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)

surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
plt.xlabel('Time')
# plt.show()


############evaluation
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print("Concordance index:",ev.concordance_td())

time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
_ = ev.brier_score(time_grid).plot()
# plt.show()

# print("Brier Score: ",ev.integrated_brier_score(time_grid))
# print("nbll: ", ev.integrated_nbll(time_grid))