import pandas as pd
from typing import List, Optional, Protocol
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform

class ModelAPI(Protocol):
    def predict(self, X: ArrayLike) -> ArrayLike:
        ...

def _instances_labels_2tab(X: pd.DataFrame, y: pd.Series, cate_columns: Optional[List[str]] = None, target_column: str = "label") -> Tabular:
    cate_columns = cate_columns if cate_columns is not None else []
    for_tabular = X.assign(**{target_column: y})
    tabular_data = Tabular(
        for_tabular,
        categorical_columns=cate_columns,
        target_column=target_column
    )
    return tabular_data

def _instances_2tab(X: pd.DataFrame, cate_columns: Optional[List[str]] = None) -> Tabular:
    cate_columns = cate_columns if cate_columns is not None else []
    return Tabular(
        X,
        categorical_columns=cate_columns,
    )

class customXGB:
    def __init__(self, n_estimators=300, max_depth=5):
        self.clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.transformer = TabularTransform()
        self.cate_columns = []
    
    def reset(self, cate_columns: Optional[List[str]] = None, target_column: str = "label"):
        self.cate_columns = cate_columns if cate_columns is not None else []
        self.target_column = target_column
    
    def fit(self, X: pd.DataFrame, y: pd.Series, cate_columns: Optional[List[str]] = None, target_column: str = "label"):
        self.reset(cate_columns=cate_columns, target_column=target_column)

        tabular_data = _instances_labels_2tab(X, y, cate_columns=self.cate_columns, target_column=self.target_column)
        self.transformer.fit(tabular_data)
        x = self.transformer.transform(tabular_data)
        train, train_labels = x[:, :-1], x[:, -1]
        self.clf.fit(train, train_labels)
        return self

    def predict(self, X: pd.DataFrame):
        tabular_data = _instances_2tab(X, cate_columns=self.cate_columns)
        x = self.transformer.transform(tabular_data)
        return self.clf.predict(x)
    
#    def accuracy(self, X, y: pd.Series) -> float:
#        pass

class customLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.clf = LogisticRegression(**kwargs)
        self.transformer = TabularTransform()
        self.cate_columns = []
    
    def reset(self, cate_columns: Optional[List[str]] = None, target_column: str = "label"):
        self.cate_columns = cate_columns if cate_columns is not None else []
        self.target_column = target_column
    
    def fit(self, X: pd.DataFrame, y: pd.Series, cate_columns: Optional[List[str]] = None, target_column: str = "label"):
        self.reset(cate_columns=cate_columns, target_column=target_column)

        tabular_data = _instances_labels_2tab(X, y, cate_columns=self.cate_columns, target_column=self.target_column)
        self.transformer.fit(tabular_data)
        x = self.transformer.transform(tabular_data)
        train, train_labels = x[:, :-1], x[:, -1]
        self.clf.fit(train, train_labels)
        return self

    def predict(self, X: pd.DataFrame):
        tabular_data = _instances_2tab(X, cate_columns=self.cate_columns)
        x = self.transformer.transform(tabular_data)
        return self.clf.predict(x)