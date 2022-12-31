import pandas as pd
import numpy as np
import sklearn as sk
from typing import List

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform

class customXGB:
    def __init__(self, n_estimators=300, max_depth=5):
        self.clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.transformer = None
        self.cate_columns = None
        self.target_column = None

    def fit(self, data: pd.DataFrame, cate_columns: List[str] = None, target_column: str = "label"):
        self.cate_columns = cate_columns if cate_columns is not None else []
        self.target_column = target_column
        tabular_data = Tabular(
            data,
            categorical_columns=self.cate_columns,
            target_column=self.target_column
        )
        self.transformer = TabularTransform().fit(tabular_data)
        x = self.transformer.transform(tabular_data)
        train, test, train_labels, test_labels = train_test_split(x[:, :-1], x[:, -1], train_size=.8)
        self.clf.fit(train, train_labels)
        return self
    
    def predict(self, data: pd.DataFrame):
        data_tab = Tabular(
            data,
            categorical_columns=self.cate_columns,
            target_column=self.target_column
        )
        x = self.transformer.transform(data_tab)
        return self.clf.predict(x[:, :-1])
    

