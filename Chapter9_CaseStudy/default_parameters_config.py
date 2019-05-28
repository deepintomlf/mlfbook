from model_train import *

_default_skfold_params = {'n_splits':5, 'shuffle':True, 'random_state':1001}

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