import numpy as np
import pandas as pd

from pandas import DataFrame
from typing import List, Tuple

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

from .predicate import Predicate

def preprocessDataset(data: DataFrame) -> DataFrame:
    d = data.copy()
    for col in d:
        if isinstance(d[col].dtype, pd.CategoricalDtype):
            d[col] = np.asarray(d[col])
        d[col] = d[col].map(lambda x: (col, x))
    return d

def aprioriout2predicateList(apriori_out: DataFrame) -> Tuple[List[Predicate], List[float]]:
    predicate_set = []
    for itemset in apriori_out["itemsets"].to_numpy():
        pred = {feature: value for feature, value in list(itemset)}
        pred = Predicate.from_dict(pred)
        predicate_set.append(pred)
    
    return predicate_set, apriori_out["support"].to_numpy().tolist()

def runApriori(data: DataFrame, min_support: float = 0.001) -> DataFrame:
    sets = data.to_numpy().tolist()
    te = TransactionEncoder()
    sets_onehot = te.fit_transform(sets)
    df = DataFrame(sets_onehot, columns=te.columns_) # type: ignore
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets.sort_values(['support'], ascending =[False])
