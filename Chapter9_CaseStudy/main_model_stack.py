import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import gc
from functools import reduce
import time
import lightgbm as lgb

from model_train import *
import os

cur_dir = os.getcwd()
results_dir = cur_dir + "//input//"

#results_dir = '../input'

base_models = {
    'lgb1': "model_01",
    'lgb2': "model_02",
    'lm': "model_03",
    'nn': "model_04"
}


cvfiles = {
    'lgb1': 'results-random-state-1001/train_LGBM5x_oof_0770248_201904241604.csv',
    'lgb2': 'results-random-state-1001/train_LGBM5x_oof_076703_201904251334.csv',
    'lm': 'model3lm/train_lm_5x_oof_0684941_201904281603.csv',
    'nn': 'model4nn/train_nn_5x_oof_0744998_201904281626.csv' 
}

subfiles = {
    'lgb1': 'results-random-state-1001/submission_LGBM5x_0770248_201904241604.csv',
    'lgb2': 'results-random-state-1001/submission_LGBM5x_076703_201904251334.csv',
    'lm': 'model3lm/submission_lm_5x_0684941_201904281603.csv',
    'nn': 'model4nn/submission_nn_5x_0744998_201904281626.csv' 
    }

##############################
## Step 1: Prepare for the data for the learning
## (Input: 4 model predcitors; Output: Binary Target Values)
##############################
model_order = [m for m in base_models]
def merge_dataframes(dfs, merge_keys):
    dfs_merged = reduce(lambda left,right: pd.merge(left, right, on=merge_keys), dfs)
    return dfs_merged
## train (oof) data
dfs = [pd.read_csv(os.path.join(results_dir, cvfiles[m])) for m in base_models]
df_train = merge_dataframes(dfs, merge_keys=['SK_ID_CURR','TARGET'])
df_train.columns = ['SK_ID_CURR','TARGET'] + [m for m in base_models]
sel_feas = [m for m in base_models]

# test data
dfs = [pd.read_csv(os.path.join(results_dir, subfiles[m])) for m in base_models]
df_test = merge_dataframes(dfs, merge_keys=['SK_ID_CURR'])
df_test.columns = ['SK_ID_CURR'] + [m for m in base_models]
print(df_test.head())

#usecols = sel_feas
#usecol = 'TARGET'

ids = df_train['SK_ID_CURR']
x_train = df_train[sel_feas]
y_train = df_train['TARGET']


##############################
## Step 2: Learning a meta-model
##############################

params = {'objective':'binary',
          'metric': 'auc',
          'num_threads': 6, 
          'num_iterations': 10000, 
          'learning_rate': 0.1,
          'scale_pos_weight': 1, #0.087, sum(negative cases) / sum(positive cases) 
          'verbose': -1,
          'silent': -1}

#train_model_lgbm(data_train, test_, y_, ids, folds_, algo_params, fit_params)


_default_skfold_params = {'n_splits':5, 'shuffle':True, 'random_state':1001}
_default_file_id_m = str(datetime.now().strftime('%Y%m%d%H%M'))
folds = StratifiedKFold(**_default_skfold_params )


fit_params = {
            # 'num_boost_round':20000,
             'early_stopping_rounds': 200,
             'verbose':100,
            # 'seed' : 5
            # 'show_stdv': True
}
res_models = train_model_lgbm(x_train,df_test, y_train,  ids, folds, params,fit_params)
