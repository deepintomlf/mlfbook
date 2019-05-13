from google.colab import drive
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
import os

drive.mount('/content/drive', force_remount=True)
input_dir = "/content/drive/My Drive/Programming/Python/kaggle/home_credit/data/"
output_dir = "/content/drive/My Drive/Programming/Python/kaggle/home_credit/output/"
python_path = '/content/drive/My Drive/Programming/Python/mlfbook/mlfbook/CaseStudy/'
if python_path not in os.sys.path:
    os.sys.path.append(python_path)

from preprocessing import *
from model_helper import *
from model_train import *
from glob import glob
import copy

def train_model_lgbm_short(data_, y_, folds_, algo_params, fit_params):
    '''
    k-folded cross validation
    return K-fold estimated models;
    '''
    oof_models = []
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = lgb.LGBMClassifier(**algo_params)
        fit_params.update({"eval_set": [(trn_x, trn_y), (val_x, val_y)]})
        clf.fit(trn_x, trn_y, **fit_params)
        oof_models.append(copy.deepcopy(clf))
        print('fold.{}'.format(n_fold+1))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds_.split(data_, y_)]
    res_models = {'oof_models': oof_models, 'folds_idx':folds_idx}
    return res_models

def oof_prediction_test(data_, test_, y_, ids, folds_, res_models):
    '''
    funcationality:
    oof_prediction + predicatin on the test set.
    '''
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])

    oof_best_iters = []
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        clf = res_models['oof_models'][n_fold]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(
            test_[feats],
            num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        test_['TARGET'] = sub_preds

        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['TARGET'] = sub_preds
    avg_best_iters = np.mean(oof_best_iters)
    df_oof_preds = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': y_, 'PREDICTION': oof_preds})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'TARGET', 'PREDICTION']]

    folds_idx = res_models['folds_idx']

    res = {
        'y':y_,
        'folds_idx':folds_idx,
        'score':roc_auc_score(y_, oof_preds),
        'test_preds':test_[['SK_ID_CURR', 'TARGET']],
        'df_oof_preds':df_oof_preds,
        'oof_preds':oof_preds,
        'avg_best_iters':avg_best_iters
    }
    return res


def feature_importance_oof(data_, test_, y_, ids, folds_, res_models):
    '''
    functionality: compute the feature importance
    '''
    feature_importance_df = pd.DataFrame()

    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        clf = res_models['oof_models'][n_fold]
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        del clf
        gc.collect()

    folds_idx = res_models['folds_idx']
    return feature_importance_df



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

model_lgbm = {
            'model': train_model_lgbm_short,
            'algo_params':_default_algo_params_lgbm,
            'fit_params':_default_fit_params_lgbm
}

_default_skfold_params = {'n_splits':5, 'shuffle':True, 'random_state':1001}
_default_file_id_m = str(datetime.now().strftime('%Y%m%d%H%M'))

#################### application ###################
x_train, x_test, y_train, ids = build_model_input(input_dir=input_dir)
folds = StratifiedKFold(**_default_skfold_params )

# train model
train_model = model_lgbm['model']
res_models = train_model(x_train, y_train, folds, train_model['algo_params'], train_model['fit_params'])
#
prediction_result = oof_prediction_test(x_train, x_test, y_train, ids, folds,res_models)
feature_importance_df  = feature_importance_oof(x_train, x_test, y_train, ids, folds, res_models)

save_training_prediction_result(prediction_result, model_type= 'lgbm')
figs = save_feature_importance(feature_importance_df , prediction_result, model_type= 'lgbm')