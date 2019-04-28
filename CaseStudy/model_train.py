import gc
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from CaseStudy.model_helper import *


_default_skfold_params = {'n_splits':5, 'shuffle':True, 'random_state':1001}
_default_file_id_m = str(datetime.now().strftime('%Y%m%d%H%M'))


def train_model_LGBM(data_, test_, y_, folds_, ids, algo_params, fit_params_base):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    oof_best_iters = []
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]

        clf = lgb.LGBMClassifier(**algo_params)
        # LightGBM parameters found by Bayesian optimization
        fit_params = fit_params_base.update({"eval_set": [(trn_x, trn_y), (val_x, val_y)]})
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

    return oof_preds, df_oof_preds, test_[['SK_ID_CURR', 'TARGET'
                                           ]], feature_importance_df, roc_auc_score(y_, oof_preds), avg_best_iters


def train_results(data, test, y, train_model,sel_feas=None,skfold_params=_default_skfold_params):
    if sel_feas is None:
        sel_feas = data.columns
    folds = StratifiedKFold(**skfold_params)
    model_train_result = train_model(data[sel_feas], test, y,folds)
    oof_preds, df_oof_preds, test_preds, importances, score, avg_best_iters = model_train_result
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(data[sel_feas], y)]
    score = str(round(score, 6)).replace('.', '')
    avg_feature_importance = get_feature_importances(importances)
    res = {
        'y':y,
        'folds_idx':folds_idx,
        'score':score,
        'importances':importances,
        'test_preds':test_preds,
        'df_oof_preds':df_oof_preds,
        'avg_feature_importance':avg_feature_importance
    }
    return res


def save_training_results(res,model_type='LGBM5x',save_dir='.'):
    # save csv
    sub_file = '{f}/submission_{m}_{s}_{id}.csv'.format(f=save_dir,m=model_type,s=res['score'], id=_default_file_id_m)
    oof_file = '{f}/train_{m}_oof_{s}_{id}.csv'.format(f=save_dir,m=model_type,s=res['score'], id=_default_file_id_m)
    res['test_preds'].to_csv(sub_file, index=False)
    res['df_oof_preds'].to_csv(oof_file, index=False)
    # save figure
    display_importances(feature_importance_df_=res['importances'],model_type=model_type,save_dir=save_dir)
    display_roc_curve(y_=res['y'], oof_preds_=res['oof_preds'], folds_idx_=res['folds_idx'],model_type=model_type,save_dir=save_dir)
    display_precision_recall(y_=res['y'], oof_preds_=res['oof_preds'], folds_idx_=res['folds_idx'],model_type=model_type,save_dir=save_dir)
