from typing import Dict, Callable, Any, List
import functools
from collections import defaultdict
from dataclasses import dataclass, field

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