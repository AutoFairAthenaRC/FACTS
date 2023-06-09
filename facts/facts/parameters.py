from typing import Dict, Callable, Any, List, Optional
import functools
from collections import defaultdict
from dataclasses import dataclass, field

from pandas import DataFrame

def make_default_featureCosts():
    return defaultdict(lambda: 1)

def default_change(v1, v2):
    return 0 if v1 == v2 else 1

def make_default_featureChanges():
    return defaultdict(lambda: default_change)

def naive_feature_change_builder(
    num_cols: List[str],
    cate_cols: List[str],
    feature_weights: Dict[str, int],
) -> Dict[str, Callable[[Any, Any], int]]:
    def feature_change_cate(v1, v2, weight):
        return (0 if v1 == v2 else 1) * weight
    def feature_change_num(v1, v2, weight):
        return abs(v1 - v2) * weight
    
    ret_cate = {col: functools.partial(feature_change_cate, weight=feature_weights.get(col, 1)) for col in cate_cols}
    ret_num = {col: functools.partial(feature_change_num, weight=feature_weights.get(col, 1)) for col in num_cols}
    return {**ret_cate, **ret_num}

def feature_change_builder(
    X: DataFrame,
    num_cols: List[str],
    cate_cols: List[str],
    ord_cols: List[str],
    feature_weights: Dict[str, int],
    num_normalization: bool = False,
    feats_to_normalize: Optional[List[str]] = None,
) -> Dict[str, Callable[[Any, Any], int]]:
    def feature_change_cate(v1, v2, weight):
        return (0 if v1 == v2 else 1) * weight

    def feature_change_num(v1, v2, weight):
        return abs(v1 - v2) * weight
    
    def feature_change_ord(v1, v2, weight, t):
        return abs(t[v1] - t[v2]) * weight

    ### normalization of numeric features
    max_vals = X.max(axis=0)
    min_vals = X.min(axis=0)
    weight_multipliers = {}
    for col in num_cols:
        weight_multipliers[col] = 1
    for col in cate_cols:
        weight_multipliers[col] = 1
    if num_normalization:
        if feats_to_normalize is not None:
            for col in feats_to_normalize:
                weight_multipliers[col] = 1 / (max_vals[col] - min_vals[col])
        else:
            for col in num_cols:
                weight_multipliers[col] = 1 / (max_vals[col] - min_vals[col])

    ret_cate = {
        col: functools.partial(feature_change_cate, weight=feature_weights.get(col, 1))
        for col in cate_cols
    }
    ret_num = {
        col: functools.partial(
            feature_change_num,
            weight=weight_multipliers[col] * feature_weights.get(col, 1),
        )
        for col in num_cols
    }
    ret_ord = {
        col: functools.partial(
            feature_change_ord,
            weight=feature_weights.get(col, 1),
            t={name: code for code, name in enumerate(X[col].cat.categories)}
        )
        for col in ord_cols
    }
    return {**ret_cate, **ret_num, **ret_ord}

@dataclass
class ParameterProxy:
    featureCosts: Dict[str, int] = field(default_factory=make_default_featureCosts)
    featureChanges: Dict[str, Callable[[Any, Any], int]] = field(default_factory=make_default_featureChanges)
    lambda_cover: int = 1
    lambda_correctness: int = 1
    lambda_featureCost: int = 1
    lambda_featureChange: int = 1

    ##### Utility methods for setting the parameters
    def setFeatureCost(self, fc: Dict):
        self.featureCosts.update(fc)
    def setFeatureChange(self, fc: Dict):
        self.featureChanges.update(fc)

    def set_lambdas(self, l1=1, l2=1, l3=1, l4=1):
        self.lambda_cover = l1
        self.lambda_correctness = l2
        self.lambda_featureCost = l3
        self.lambda_featureChange = l4
    ##### Utility methods for setting the parameters

##### Unused parameters
epsilon1 = 20
epsilon2 = 7
epsilon3 = 10
C_max = 1 # max(featureCosts.values())
M_max = 1
##### Unused parameters