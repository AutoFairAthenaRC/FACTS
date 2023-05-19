import pandas as pd
from pandas import DataFrame
import numpy as np

def clean_dataset(X : DataFrame, dataset : str) -> DataFrame:
    if dataset == 'adult':
        X = X.drop(columns=['fnlwgt','education'])
        cols = list(X.columns)
        X[cols] = X[cols].replace([' ?'],np.nan)
        X = X.dropna()
        X['relationship'] = X['relationship'].replace([' Husband',' Wife'],' Married')
        X['hours-per-week'] = pd.cut(x=X['hours-per-week'], bins=[0.9,25,39,40,55,100],labels=['PartTime','MidTime','FullTime','OverTime','BrainDrain'])
        X.age = pd.qcut(X.age,q=5)
        X['income'] = np.where((X['income'] == ' <=50K') , 0, 1)
    elif dataset == 'SSL':
        X['SSL SCORE'] = np.where((X['SSL SCORE'] >= 345) , 0, 1)
        X = X.replace('WBH','BLK')
        X = X.replace('WWH','WHI')
        X = X.replace('U',np.nan)
        X = X.replace('X',np.nan)
        X = X.dropna()
        X = X[(X['RACE CODE CD'] != 'I') & (X['RACE CODE CD'] != 'API')]
        X['PREDICTOR RAT TREND IN CRIMINAL ACTIVITY'] = pd.qcut(X['PREDICTOR RAT TREND IN CRIMINAL ACTIVITY'],q=6)
        X['PREDICTOR RAT AGE AT LATEST ARREST'] = X['PREDICTOR RAT AGE AT LATEST ARREST'].replace('less than 20', '10-20')
    elif dataset == 'compas':
        X = X.reset_index(drop=True)
        X = X.drop(columns=['age','c_charge_desc'])
        X.target.replace('Recidivated',0,inplace=True)
        X.target.replace('Survived',1,inplace=True)

    return X

