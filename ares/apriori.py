from pandas import DataFrame
from typing import List, Any
from numpy.typing import NDArray
from lib import Predicate
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def aprioriout2predicateList(apriori_out: DataFrame, data: DataFrame) -> List[Predicate]:
    value2featureMap = {}
    for feature in data.columns:
        for value in data[feature].unique():
            value2featureMap[value] = feature
    
    predicate_set = []
    for itemset in apriori_out["itemsets"].to_numpy():
        pred = {value2featureMap[item]: item for item in itemset}
        pred = Predicate.from_dict_categorical(pred)
        predicate_set.append(pred)
    
    return predicate_set

def runApriori(data: DataFrame, min_support: float = 0.001) -> DataFrame:
    sets = data.to_numpy().tolist()
    te = TransactionEncoder()
    sets_onehot = te.fit_transform(sets)
    df = DataFrame(sets_onehot, columns=te.columns_) # type: ignore
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets.sort_values(['support'], ascending =[False])
