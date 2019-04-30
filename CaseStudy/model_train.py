import gc
import numpy as np
import pandas as pd
import datetime
from model_helper import *
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from functools import lru_cache



_default_skfold_params = {'n_splits':5, 'shuffle':True, 'random_state':1001}
_default_file_id_m = str(datetime.now().strftime('%Y%m%d%H%M'))


############################ Train Models: LGBM,LogisticReg,NeuralNetwork ###############################

def train_model_lgbm(data_, test_, y_, ids, folds_, algo_params, fit_params):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    oof_best_iters = []
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = lgb.LGBMClassifier(**algo_params)
        fit_params.update({"eval_set": [(trn_x, trn_y), (val_x, val_y)]})
        clf.fit(trn_x, trn_y, **fit_params)
        oof_best_iters.append(clf.best_iteration_)
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]

        sub_preds += clf.predict_proba(
            test_[feats],
            num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['TARGET'] = sub_preds
    avg_best_iters = np.mean(oof_best_iters)
    df_oof_preds = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': y_, 'PREDICTION': oof_preds})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'TARGET', 'PREDICTION']]
    #
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds_.split(data_, y_)]
    avg_feature_importance = get_feature_importances(feature_importance_df)
    res = {
        'y':y_,
        'folds_idx':folds_idx,
        'score':roc_auc_score(y_, oof_preds),
        'test_preds':test_[['SK_ID_CURR', 'TARGET']],
        'df_oof_preds':df_oof_preds,
        'oof_preds':oof_preds,
        'importances': feature_importance_df,
        'avg_feature_importance':avg_feature_importance,
        'avg_best_iters':avg_best_iters,
        'algo_params':algo_params,
        'fit_params':fit_params
    }
    return res
    # return oof_preds, df_oof_preds, test_[['SK_ID_CURR', 'TARGET'
    #                                        ]], feature_importance_df, roc_auc_score(y_, oof_preds), avg_best_iters

def train_model_logistic(data_, test_, y_, ids, folds_, algo_params, fit_params):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])

    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]

        # Make the model with the specified regularization parameter
        clf = LogisticRegression(**algo_params)
        # Train on the training data
        if fit_params:
            clf.fit(trn_x, trn_y,**fit_params)
        else:
            clf.fit(trn_x, trn_y)

        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
        sub_preds += clf.predict_proba(test_[feats])[:, 1] / folds_.n_splits
        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['TARGET'] = sub_preds

    df_oof_preds = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': y_, 'PREDICTION': oof_preds})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'TARGET', 'PREDICTION']]
    #
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds_.split(data_, y_)]
    res = {
        'y':y_,
        'folds_idx':folds_idx,
        'score':roc_auc_score(y_, oof_preds),
        'test_preds':test_[['SK_ID_CURR', 'TARGET']],
        'df_oof_preds':df_oof_preds,
        'oof_preds':oof_preds,
        'algo_params':algo_params,
        'fit_params':fit_params
    }
    return res

def train_model_neuralnetwork(data_, test_, y_, ids, folds_,algo_params, fit_params):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])

    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]

        print('Setting up neural network...')
        nn = Sequential()
        nn.add(Dense(units=algo_params['units_init'], kernel_initializer='normal', input_dim=trn_x.shape[1]))
        nn.add(PReLU())
        nn.add(Dropout(algo_params['dropout']))
        for units in algo_params['units_layers']:
            nn.add(Dense(units=units, kernel_initializer=algo_params['kernel_initializer']))
            nn.add(PReLU())
            nn.add(BatchNormalization())
            nn.add(Dropout(algo_params['dropout']))
        nn.add(Dense(1, kernel_initializer=algo_params['kernel_initializer'], activation=algo_params['activation']))
        nn.compile(loss=algo_params['loss'], optimizer=algo_params['optimizer'])
        nn.compile(loss=algo_params['loss'], optimizer=algo_params['optimizer'], metrics=algo_params['metric'])

        print('Fitting neural network...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        # nn.fit(trn_x, trn_y, validation_split=0.1, epochs=20, batch_size = 128, verbose=2, callbacks=[early_stopping])
        nn.fit(trn_x, trn_y, validation_data=(val_x, val_y), **fit_params)
        # nn.fit(trn_x, trn_y, validation_split=0.1, epochs=20, batch_size = 128, verbose=2)

        print('Predicting...')
        oof_preds[val_idx] = nn.predict_proba(val_x).flatten().clip(0, 1)
        sub_preds += nn.predict_proba(test_[feats]).flatten().clip(0, 1) / folds_.n_splits

        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del nn, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['TARGET'] = sub_preds

    df_oof_preds = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': y_, 'PREDICTION': oof_preds})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'TARGET', 'PREDICTION']]
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds_.split(data_, y_)]
    res = {
        'y': y_,
        'folds_idx': folds_idx,
        'score': roc_auc_score(y_, oof_preds),
        'test_preds': test_[['SK_ID_CURR', 'TARGET']],
        'df_oof_preds': df_oof_preds,
        'oof_preds': oof_preds,
        'algo_params': algo_params,
        'fit_params': fit_params
    }
    return res


############################ Train:  Models + StratifiedKFold ###############################


def train_results(data, test, y, ids,train_model,sel_feas=None,skfold_params=_default_skfold_params):
    if sel_feas is None:
        sel_feas = data.columns
    folds = StratifiedKFold(**skfold_params)
    df = data[sel_feas]
    model = train_model['model']
    algo_params = train_model['algo_params']
    fit_params = train_model['fit_params']
    model_train_result = model(df, test, y,ids,folds,algo_params,fit_params)
    return model_train_result


def save_training_results(res,model_type,save_dir='.'):
    # save csv
    sub_file = '{f}/submission_{m}_5x_{s}_{id}.csv'.format(f=save_dir,m=model_type,s=res['score'], id=_default_file_id_m)
    oof_file = '{f}/train_{m}_5x_oof_{s}_{id}.csv'.format(f=save_dir,m=model_type,s=res['score'], id=_default_file_id_m)
    res['test_preds'].to_csv(sub_file, index=False)
    res['df_oof_preds'].to_csv(oof_file, index=False)
    # save figure
    figs = []
    if 'importances' in res.keys():
        figs.append(display_importances(feature_importance_df_=res['importances'],model_type=model_type,save_dir=save_dir))
    figs.append(display_roc_curve(y_=res['y'], oof_preds_=res['oof_preds'], folds_idx_=res['folds_idx'],model_type=model_type,save_dir=save_dir))
    figs.append(display_precision_recall(y_=res['y'], oof_preds_=res['oof_preds'], folds_idx_=res['folds_idx'],model_type=model_type,save_dir=save_dir))
    return figs
