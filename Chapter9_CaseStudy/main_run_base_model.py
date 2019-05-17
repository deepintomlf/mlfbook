# Import Python Packages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from datetime import datetime
from functools import lru_cache
import os

  
from preprocessing import *
from model_helper import *
from model_train import *
from glob import glob

print(os.getcwd())
cur_dir = os.getcwd()
input_dir = cur_dir + "//input//"
output_dir = cur_dir + "//output//"
save_dir = cur_dir +"//output//"
############ dfeault LGBM Model ############

_default_algo_params_lgbm = {'objective': 'binary',
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

_default_fit_params_lgbm = {
    "eval_metric": 'auc',
    'verbose': 1000,
    'early_stopping_rounds': 100
}

_default_model_lgbm = {
                            'model': train_model_lgbm,
                            'algo_params':_default_algo_params_lgbm,
                            'fit_params':_default_fit_params_lgbm
}

############ dfeault Logistic Model ############
_default_algo_params_logistic = {
    'C':0.0001
}

_default_fit_params_logistic = None

_default_model_logistic = {
                            'model': train_model_logistic,
                            'algo_params':_default_algo_params_logistic,
                            'fit_params':_default_fit_params_logistic
}


############ dfeault NeuralNetwork Model ############

_default_algo_params_nn = {
    'units_init':400,
    'units_layers': [160,64,26,12],
    'kernel_initializer':'normal',
    'dropout':.3,
    'activation':'sigmoid',
    'optimizer':'adam',
    'loss':'binary_crossentropy',
    'metric':['acc']
}

_default_fit_params_nn = {
    'epochs':20,
    'batch_size':256,
    'verbose':2,
    'callbacks':[EarlyStopping(monitor='val_loss', patience=5)]
}

_default_model_neuralnetwork = {
                            'model': train_model_neuralnetwork,
                            'algo_params':_default_algo_params_nn,
                            'fit_params':_default_fit_params_nn
}

# LGBM gridsearchCV example: (global) alternatives to default algo params above

#estimator = lgb.LGBMClassifier(**_default_algo_params_lgbm)
#x_train, x_test, y_train, ids = build_model_input(input_dir=input_dir)

#param_grid = {
#'learning_rate': [0.01, 0.03],
#'max_depth':[3, 5]
#}

#grid = grid_search_cv(x_train,y_train,estimator,param_grid,cv=5, res_params_only=False)
#params_opt = grid.best_params_
#print(params_opt)

#-----------------------------------------#
def run_model(model_type,model_map,input_dir, output_dir):
    save_dir = '{}/model_{}'.format(output_dir,model_type)
    print('------- run model: {} '.format(model_type,save_dir))
    if model_type.lower() in ['lgbm']:
      x_train, x_test, y_train, ids = build_model_input(input_dir=input_dir)
    else:
      x_train, x_test, y_train, ids = build_model_input_extended(input_dir=input_dir)
    # get train results
    res = train_results(x_train, x_test, y_train, ids, model_map[model_type])

    if len(glob(save_dir))==0:
      os.mkdir(save_dir)
    print('-------- save results to:{}'.format(model_type,save_dir))
    save_training_results(res,model_type=model_type,save_dir=save_dir)

#------------------------------#
model_map = {
    'logistic': _default_model_logistic,
    'neuralnetwork':_default_model_neuralnetwork,
    'lgbm':_default_model_lgbm}

figs = run_model('logistic',model_map,input_dir,output_dir)

figs = run_model('neuralnetwork', model_map,input_dir,output_dir)

figs = run_model('lgbm',model_map,input_dir,output_dir)

