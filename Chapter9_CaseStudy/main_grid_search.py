# Import Python Packages
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import pandas as pd
import numpy as np
#from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import ParameterGrid
from six import *
import lightgbm as lgb
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from preprocessing import *
from default_parameters_config import _default_fit_params_lgbm, _default_algo_params_lgbm 
from glob import glob


#### supress some harmless warnings
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
####

## Step 1: Import data 
#### Set the directory
cur_dir = os.getcwd()
input_dir = cur_dir + "//input//"
output_dir = cur_dir + "//output//"
save_dir = cur_dir +"//output//"
x_train, x_test, y_train, ids = build_model_input(input_dir=input_dir)


## Step 2: Initialize the LGBM classifier and do cross validation for parameter tuning for the train set;

def GridSearch(x_train, sel_feas, y_train, param_grid, algo_params = _default_algo_params_lgbm, cv = 5, num_boost_round=  2000,
		                    early_stopping_rounds=100,
		                    verbose_eval=1000,
		                    seed = 5):
	max_auc = 0
	best_params = {}
	grid_scores = []
	for i in range(len(param_grid)):
		algo_params.update(param_grid[i])
        # cross valition for one set of parameters
		cvresult = lgb.cv(algo_params,
		                    lgb.Dataset(x_train[sel_feas], label=y_train.values.astype(int)),
		                    nfold=cv,
		                    stratified=True,
		                    num_boost_round=  num_boost_round,
		                    early_stopping_rounds=early_stopping_rounds,
		                    verbose_eval=verbose_eval,
		                    seed = seed,
		                    show_stdv=True) 
		curr_auc = pd.Series(cvresult['auc-mean']).max() # k-fold cross validation accuracy
		curr_std = pd.Series(cvresult['auc-mean']).std()
		cv_res = param_grid[i]
		cv_res.update({'auc': curr_auc, 'std': curr_std})
		grid_scores.append(cv_res) 
		if curr_auc > max_auc:
			best_params = param_grid[i]
	return best_params,  grid_scores

print('---start grid search')
sel_feas = x_train.columns
cv = 5
param_grid = {'max_depth': [5, 6],
              'learning_rate': [ 0.03, 0.01],
              'num_leaves': [20, 31],
              }
Param_grid = list(ParameterGrid(param_grid))
best_params, grid_scores = GridSearch(x_train, sel_feas, y_train, Param_grid, cv = 5)


## Step 3: Visualize the grid search results
print('---start output grid search results')
 
for cvres in grid_scores:
    print(cvres)

print('best parameters')
print(best_params)


