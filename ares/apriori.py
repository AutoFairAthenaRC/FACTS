from pandas import DataFrame
from typing import List, Any
from numpy.typing import NDArray
from lib2 import Predicate
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def preprocessDataset(data: DataFrame) -> DataFrame:
    d = data.copy()
    for col in d:
        d[col] = d[col] + "+" + col
    return d

def aprioriout2predicateList(apriori_out: DataFrame) -> List[Predicate]:
    predicate_set = []
    for itemset in apriori_out["itemsets"].to_numpy():
        feature_value_splits = map(lambda item: item.split("+"), list(itemset))
        pred = {feature: value for value, feature in feature_value_splits}
        pred = Predicate.from_dict(pred)
        predicate_set.append(pred)
    
    return predicate_set

def runApriori(data: DataFrame, min_support: float = 0.001) -> DataFrame:
    sets = data.to_numpy().tolist()
    te = TransactionEncoder()
    sets_onehot = te.fit_transform(sets)
    df = DataFrame(sets_onehot, columns=te.columns_) # type: ignore
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets.sort_values(['support'], ascending =[False])
