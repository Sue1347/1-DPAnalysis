# Simply used the pycox notebook for reference
# 巧妇难为无米之炊

# with 70 as training data, with 5 variate I can not even set a [8,8,1] MLP network
# because it has 129 parameters to calculate

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
file_name = "simple-dataset-SMM-clean.csv"

df = pd.read_csv(os.path.join(file_path,file_name))

print(df.columns)
print("SUVmaxBM: ",df["SUVmaxBM"].mean(),df["SUVmaxBM"].std())

# df_train = metabric.read_df()
# df_test = df_train.sample(frac=0.2)
# df_train = df_train.drop(df_test.index)
# df_val = df_train.sample(frac=0.2)
# df_train = df_train.drop(df_val.index)

# cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
# cols_leave = ['x4', 'x5', 'x6', 'x7']

# standardize = [([col], StandardScaler()) for col in cols_standardize]
# leave = [(col, None) for col in cols_leave]

# x_mapper = DataFrameMapper(standardize + leave)

# x_train = x_mapper.fit_transform(df_train).astype('float32')
# x_val = x_mapper.transform(df_val).astype('float32')
# x_test = x_mapper.transform(df_test).astype('float32')

# get_target = lambda df: (df['duration'].values, df['event'].values)
# y_train = get_target(df_train)
# y_val = get_target(df_val)
# durations_test, events_test = get_target(df_test)
# val = x_val, y_val


cox_column_list = [ 'PFS', 'Event',
                   # 'Pic','Plasmocytose',
        #'PET BMI', 'SUVmaxBM',# 'SUVmaxFL',# 'PET EMD', 'PET PMD',   'PET FL',
        'MRI BMI', #'ADCMeanBMI', #'MRI FL', 'ADCMeanFL', 'MRI EMD', 'MRI PMD',
        'PEI', 'MITR' 
       ]  
df["PFS"] = df[["PFS"]].astype(float)
df['PEI'] = df['PEI'].replace(['', ' '], None)
df["PEI"] = df[["PEI"]].astype(float)
df["Event"] = df["P ou R"].notna().astype(int)

df_train = df[cox_column_list] # metabric.read_df()
df_test = df_train.sample(frac=0.1)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

print(df_train.dtypes)

########### preprocess the characteristics

cols_standardize = [#'Pic','Plasmocytose',
                    'PEI', 'MITR' ]
cols_leave = ['MRI BMI']

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

################## create neural networks
in_features = x_train.shape[1]
num_nodes = [8,8] #32, 32
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

for name, param in net.named_parameters():
    print(f"{name}: {param.shape}")
# print(net.named_parameters())
# fit the model
model = CoxPH(net, tt.optim.Adam)

batch_size = 32 #256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()
plt.show()

print("### Best lr: ",lrfinder.get_best_lr())
# exit()

model.optimizer.set_lr(0.1)

# Training the model
epochs = 32 #512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
log.plot()
plt.show()

print("model.partial_log_likelihood(*val).mean()", model.partial_log_likelihood(*val).mean())

##### prediction
_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)

surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
plt.xlabel('Time')
plt.show()


############evaluation
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print("Concordance index:",ev.concordance_td())

time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
_ = ev.brier_score(time_grid).plot()
# plt.show()

# print("Brier Score: ",ev.integrated_brier_score(time_grid))
# print("nbll: ", ev.integrated_nbll(time_grid))