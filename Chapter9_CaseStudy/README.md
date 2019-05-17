# Chapter 9: Home Credit Default Risk Project 
## Getting Start
The description of the project can be found in [https://www.kaggle.com/c/home-credit-default-risk](https://www.kaggle.com/c/home-credit-default-risk)

To run the codes, you need to download the following two csv files to /input/ folder from the above link:

- application_train.csv
- application_test.csv


## Structure of Codes
preprocessing.py:
- load input data
- feature selection

model\_helper.py:
- feature importance
- plot: importance, roc_curve, precision

model\_train.py
- train model: lgbm,logistic,neuralnetwork
- train results (pass KFold to model)
- save results

run\_model\_kaggle\_home\_credit.ipynb: 
- construct default model parameters
- example of running model training and save result
