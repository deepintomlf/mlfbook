import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from datetime import datetime
from sklearn.model_selection import GridSearchCV


def save_figure(fig,fdir,fname,fid=str(datetime.now().strftime('%Y%m%d%H%M')),fformat='png'):
    fig.savefig(f'{fdir}/{fname}_{fid}.{fformat}')

def get_feature_importances(feature_importance_df_,save_csv=True,save_dir='.'):
    avg_df = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False).reset_index()
    avg_df.columns = ['feature', 'importance']
    if save_csv:
        avg_df.to_csv('{f}/feature_importances_{id}.csv'.format(f=save_dir,id=str(datetime.now().strftime('%Y%m%d%H%M'))), index=False)
    return avg_df


def display_importances(feature_importance_df_,model_type='LightGBM',save_dir=None):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(
        by="importance", ascending=False)[:60].index

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)]

    fig = plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False))
    plt.title('{} Features (avg over folds)'.format(model_type))
    plt.tight_layout()
    plt.savefig('feature_importances_' + str(datetime.now().strftime('%Y%m%d%H%M')) + '.png')
    if save_dir:
        save_figure(fig,save_dir,'feature_importances')
    return fig


def display_roc_curve(y_, oof_preds_, folds_idx_,model_type,save_dir=None):
    # Plot ROC curves
    fig = plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        lw=2,
        color='r',
        label='Luck',
        alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(
        fpr,
        tpr,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(model_type))
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_dir:
        save_figure(fig,save_dir,'roc_curve-01')
    return fig


def display_precision_recall(y_, oof_preds_, folds_idx_, model_type='',save_dir=None):
    # Plot ROC curves
    fig = plt.figure(figsize=(6, 6))

    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(
        precision,
        recall,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('{} Recall / Precision'.format(model_type))
    plt.legend(loc="best")
    plt.tight_layout()
    if save_dir:
        save_figure(fig,save_dir,'recall_precision_curve')
    return fig


def grid_search_cv(x_train,y_train,estimator,param_grid,cv=5, res_params_only=False):
    grid = GridSearchCV(estimator, param_grid, cv=cv)
    grid.fit(x_train, y_train)
    if res_params_only:
        res = grid.best_params_
    else:
        res = grid
    return res