# Import Python Packages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

import lightgbm as lgb
#from datetime import datetime
#from functools import lru_cache
import os

  
from preprocessing import *
from model_helper import *
from model_train import *

from glob import glob

#### supress some harmless warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

####

## Set the directory
cur_dir = os.getcwd()
input_dir = cur_dir + "//input//"
output_dir = cur_dir + "//output//"
save_dir = cur_dir +"//output//"

## Step 1: Import data 
print(os.getcwd())
x_train, x_test, y_train, ids = build_model_input(input_dir=input_dir)


## Step 2: Prepare the data for learning (LGDT) 
np.random.seed(1)

_default_params_lgbm = {
        
          'objective': 'binary',
          'metric': 'auc',
          'num_threads': 6,
          'num_iterations': 10000,
          'max_depth': 5,
          # 'num_leaves': 31,
          'learning_rate': 0.03,
          'bagging_fraction': 0.744,
          'feature_fraction': 0.268,
          'lambda_l1': 0.91,
          'lambda_l2': 0.89,
          'min_child_weight': 18.288,
          'min_gain_to_split': 0.0365,
          'verbose': -1,
          'silent': -1}


param_grid = {'max_depth': [5,6]}
sel_feas = x_train.columns

#lgb.cv(_default_params_lgbm,train_data, num_round, nfold=2) 
#lgb_estimator = lgb.LGBMClassifier(_default_params_lgbm) 
train_data = lgb.Dataset(x_train[sel_feas], label=y_train.values.astype(int))

## Step 3: Choose one set of hyper-parameters and conduct cross validation:
#print('---start cross validation')
#cvresult = lgb.cv(_default_algo_params_lgbm,
#                     train_data,
#                     nfold=5,
#                     stratified=True,
#                     num_boost_round=  2000,
#                     early_stopping_rounds=100,
#                     verbose_eval=100,
#                     seed = 5,
#                     show_stdv=True) 

## Step 4: Initialize the LGBM classifier and do cross validation for parameter tuning for the train set;
print('---start grid search')
# lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=2000, learning_rate=0.01, metric='auc')
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=2000, learning_rate=0.01, metric='auc',
 num_threads = 6, num_iterations = 10000,
          # 'num_leaves': 31,
          #learning_rate =  0.03,
          #bagging_fraction =  0.744,
          feature_fraction =  0.268,
          lambda_l1 =  0.91,
          lambda_l2= 0.89,
          min_child_weight = 18.288,
          min_gain_to_split =  0.0365,
          verbose = -1,
          silent= -1)
#lgb_estimator = lgb.LGBMClassifier(_default_params_lgbm ) 
gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=5)
lgb_model = gsearch.fit(X=x_train[sel_feas], y=y_train.values.astype(int))

## Step 5: Visualize the grid search results
print('---start output grid search results')
 
print(lgb_model.best_params_, lgb_model.best_score_)

scores = [x[1] for x in lgb_model.grid_scores_]
#scores = np.array(scores).reshape(len(Cs), len(Gammas))
print(scores)