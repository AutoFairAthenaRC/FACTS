from typing import Dict
from collections import defaultdict

import numpy as np
from pandas import DataFrame

from parameters import *
from models import ModelAPI
from apriori import runApriori, preprocessDataset, aprioriout2predicateList
from recourse_sets import TwoLevelRecourseSet

## Re-exporting
from optimization import optimize
from predicate import Predicate
## Re-exporting



def split_dataset(X: DataFrame, attr: str):
    vals = X[attr].unique()
    grouping = X.groupby(attr)
    return {val: grouping.get_group(val) for val in vals}

def global_counterfactuals(X: DataFrame, model: ModelAPI, sensitive_attribute: str, subsample_size=400):
    X_aff_idxs = np.where(model.predict(X) == 0)[0]
    X_aff = X.iloc[X_aff_idxs, :]

    d = X.drop([sensitive_attribute], axis=1)
    freq_itemsets = runApriori(preprocessDataset(d), min_support=0.03)
    freq_itemsets.reset_index()

    RL = aprioriout2predicateList(freq_itemsets)

    SD = list(map(Predicate.from_dict, [
        {sensitive_attribute: val} for val in X[sensitive_attribute].unique()
    ]))

    ifthen_triples = np.random.choice(RL, subsample_size, replace=False) # type: ignore
    affected_sample = X_aff.iloc[np.random.choice(X_aff.shape[0], size=subsample_size, replace=False), :]
    final_rules = optimize(SD, ifthen_triples, affected_sample, model)

    return TwoLevelRecourseSet.from_triples(final_rules[0])

