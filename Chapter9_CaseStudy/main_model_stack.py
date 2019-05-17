results_dir = '../input'

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