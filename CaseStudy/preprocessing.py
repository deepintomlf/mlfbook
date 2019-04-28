import pandas as pd
import numpy as np
import gc
import os
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler, Imputer



@lru_cache(maxsize=None)
def load_data_application(input_dir, num_rows=None, nan_as_category=True):
    df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=num_rows)
    test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=num_rows)

    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df_all = df.append(test_df).reset_index()

    types_all = df_all.dtypes

    del test_df, df
    gc.collect()
    return df_all


def features_grouping(raw_feature_set):
    # grouping
    sel_feas_EXT_SOURCE = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    sel_feas_AMT = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE']
    sel_feas_DAYS = ['DAYS_EMPLOYED',
                     'DAYS_BIRTH',
                     'DAYS_REGISTRATION',
                     'DAYS_LAST_PHONE_CHANGE',
                     'DAYS_ID_PUBLISH',
                     'OWN_CAR_AGE']

    sel_feas_CNT_FAM = ['CNT_FAM_MEMBERS', 'CNT_CHILDREN']

    sel_feas_OWN = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    sel_feas_REGION = ['REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']

    sel_feas_PERSON = ['CODE_GENDER',
                       'NAME_CONTRACT_TYPE',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'OCCUPATION_TYPE',
                       'ORGANIZATION_TYPE']

    sel_feas_CONTACT = ['FLAG_MOBIL',
                        'FLAG_EMP_PHONE',
                        'FLAG_WORK_PHONE',
                        'FLAG_CONT_MOBILE',
                        'FLAG_PHONE',
                        'FLAG_EMAIL']

    sel_feas_APPR_PROCESS = ['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START']

    sel_feas_ADDRESS_MATCH = ['REG_REGION_NOT_LIVE_REGION',
                              'REG_REGION_NOT_WORK_REGION',
                              'LIVE_REGION_NOT_WORK_REGION',
                              'REG_CITY_NOT_LIVE_CITY',
                              'REG_CITY_NOT_WORK_CITY',
                              'LIVE_CITY_NOT_WORK_CITY']

    sel_feas_SOCIAL_CIRCLE = ['OBS_30_CNT_SOCIAL_CIRCLE',
                              'DEF_30_CNT_SOCIAL_CIRCLE',
                              'OBS_60_CNT_SOCIAL_CIRCLE',
                              'DEF_60_CNT_SOCIAL_CIRCLE']

    sel_feas_DOCUMENT = [_f for _f in raw_feature_set if 'FLAG_DOCUMENT' in _f]
    sel_feas_BUILDING = [_f for _f in raw_feature_set if ('_AVG' in _f)] + \
                        [_f for _f in raw_feature_set if ('_MODE' in _f)] + \
                        [_f for _f in raw_feature_set if ('_MEDI' in _f)]

    # TODO: check sel_feas_BUILDING_object not used?
    sel_feas_BUILDING_object = ['WALLSMATERIAL_MODE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE']
    sel_feas_AMT_REQ = [_f for _f in raw_feature_set if 'AMT_REQ_CREDIT_BUREAU' in _f]

    sel_feas_EXTRA = ['TARGET', 'SK_ID_CURR']
    sel_feas = sel_feas_EXTRA + \
               sel_feas_EXT_SOURCE + \
               sel_feas_AMT + \
               sel_feas_DAYS + \
               sel_feas_CNT_FAM + \
               sel_feas_OWN + \
               sel_feas_REGION + \
               sel_feas_PERSON + \
               sel_feas_CONTACT + \
               sel_feas_APPR_PROCESS + \
               sel_feas_ADDRESS_MATCH + \
               sel_feas_SOCIAL_CIRCLE + \
               sel_feas_DOCUMENT + \
               sel_feas_BUILDING + \
               sel_feas_AMT_REQ

    # add grouping names
    feature_groups = {
        'EXTRA': sel_feas_EXTRA,
        'EXT_SOURCE': sel_feas_EXT_SOURCE,
        'AMT': sel_feas_AMT,
        'DAYS': sel_feas_DAYS,
        'CNT_FAM': sel_feas_CNT_FAM,
        'OWN': sel_feas_OWN,
        'REGION': sel_feas_REGION,
        'PERSON': sel_feas_PERSON,
        'CONTACT': sel_feas_CONTACT,
        'APPR_PROCESS': sel_feas_APPR_PROCESS,
        'ADDRESS_MATCH': sel_feas_ADDRESS_MATCH,
        'SOCIAL_CIRCLE': sel_feas_SOCIAL_CIRCLE,
        'DOCUMENT': sel_feas_DOCUMENT,
        'BUILDING': sel_feas_BUILDING,
        'AMT_REQ': sel_feas_AMT_REQ
    }

    return sel_feas, feature_groups


def features_add_derivative(df_all, feature_groups):
    '''
    derivativeI: features from aggregation
    derivativeII: features from domain knowledge
    '''
    # aggregation
    df = df_all.copy()
    agg_mapping = {
        'DOCUMENT': 'FLAG_DOCUMENT_TOTAL',
        'AMT_REQ': 'AMT_REQ_CREDIT_BUREAU_TOTAL',
        'BUILDING': 'BUILDING_SCORE_TOTAL',
        'CONTACT': 'CONTACT_SCORE_TOTAL',
        'ADDRESS_MATCH': 'ADDRESS_MATCH_SCORE_TOTAL'}

    agg_func = lambda group_name, df: df[feature_groups[group_name]].sum(skipna=True, axis=1)

    for (group_name, new_name) in agg_mapping.items():
        df[new_name] = agg_func(group_name, df)

        # new features: from domain knowledge
        df['AMT_INCOME_TOTAL'].replace(1.170000e+08, np.nan, inplace=True)
        df['ANNUITY_CREDIT_PERC'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        ##df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / (1+df['AMT_INCOME_TOTAL'])
        df['GOODS_PRICE_CREDIT_PERC'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        df['DAYS_CREDIT_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        ##df['OWN_CAR_AGE'].replace(np.nan, 0, inplace= True)
        df['CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
        df['CHILDREN_PER_FAM_MEMBERS'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
        df['INCOME_PER_FAM_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        # df['CODE_GENDER'].replace('XNA', 'M', inplace= True)
        sel_feas_object = [col for col in df.columns if df[col].dtype == 'object']
        # Categorical features: Binary features and One-Hot encoding
        for bin_feature in (sel_feas_object):
            df[bin_feature], uniques = pd.factorize(df[bin_feature])
        # df, cat_cols = one_hot_encoder(df, True)
        # for bin_feature in (sel_feas_APPR_PROCESS):
        #    df[bin_feature], uniques = pd.factorize(df[bin_feature])
        # for bin_feature in (sel_feas_PERSON):
        #    df[bin_feature], uniques = pd.factorize(df[bin_feature])
        print(df.shape)
        return df


def features_clean(df, feature_groups, extra_groups=None, extra_features=None):
    # groups to remove
    r_groups_default = [
        'DOCUMENT',
        'AMT_REQ',
        'BUILDING',
        'CONTACT',
        'ADDRESS_MATCH']
    r_groups = r_groups_default if extra_groups is None else r_groups_default + extra_groups
    r_grp_features = [f for g in r_groups for f in feature_groups[g]]

    # features to remove
    r_features_default = [
        'DAYS_EMPLOYED',
        'CNT_FAM_MEMBERS',
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_GOODS_PRICE',
        'CNT_CHILDREN',
        'FLAG_OWN_CAR',
        'OBS_60_CNT_SOCIAL_CIRCLE']

    r_features = r_features_default if extra_features is None else r_features_default + extra_features

    # drop
    df.drop(labels=r_grp_features + r_features, inplace=True, axis=1)
    return df


def feature_selection(input_dir, num_rows=None):
    data_raw = load_data_application(input_dir, num_rows)
    features_raw = data_raw.columns
    sel_feas, feature_groups = features_grouping(features_raw)
    data = features_add_derivative(data_raw, feature_groups)
    data = features_clean(data, feature_groups)
    return data


def build_model_input(df=None,input_dir=None, num_rows=None):
    print("Process application train and test...")
    if df is None:
        df = feature_selection(input_dir, num_rows)
    tr = df[df['TARGET'].notnull()]
    te = df[df['TARGET'].isnull()]
    del df
    gc.collect()
    feats = [f for f in tr.columns if f not in ['index', 'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']]
    ids = tr['SK_ID_CURR']
    y = tr['TARGET']
    tr = tr[feats]
    return tr, te, y, ids


def build_model_input_extended(df=None,input_dir=None, num_rows=None):
    print("Process application train and test...")
    if df is None:
        print('load feature selection ...')
        df = feature_selection(input_dir, num_rows)
    else:
        print('load data given ...')
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    y = df['TARGET']
    X = df[feats]
    X = X.fillna(X.mean()).clip(-1e11, 1e11)
    scaler = MinMaxScaler()
    scaler.fit(X)
    training = y.notnull()
    testing = y.isnull()
    tr = scaler.transform(X[training])
    te = scaler.transform(X[testing])

    tr = pd.DataFrame(tr, columns=X[training].columns)
    te = pd.DataFrame(te, columns=X[training].columns)

    y = y[training]
    te['SK_ID_CURR'] = df[testing]['SK_ID_CURR']
    te['TARGET'] = df[testing]['TARGET']
    ids = df[training]['SK_ID_CURR']
    return tr, te, y, ids