from typing import Dict
from collections import defaultdict

featureCosts = defaultdict(lambda: 1)
def default_cost(v1, v2):
    return 0 if v1 == v2 else 1
featureChanges = defaultdict(lambda: default_cost)

lambda_cover = 1
lambda_correctness = 1
lambda_featureCost = 1
lambda_featureChange = 2

##### Utility functions for setting the parameters
def setFeatureCost(fc: Dict):
    global featureCosts
    featureCosts.update(fc)
def setFeatureChange(fc: Dict):
    global featureChanges
    featureChanges.update(fc)

def set_lambdas(l1=1, l2=1, l3=1, l4=1):
    global lambda_cover, lambda_correctness, lambda_featureCost, lambda_featureChange
    lambda_cover = l1
    lambda_correctness = l2
    lambda_featureCost = l3
    lambda_featureChange = l4
##### Utility functions for setting the parameters

##### Unused parameters
epsilon1 = 20
epsilon2 = 7
epsilon3 = 10
C_max = max(featureCosts.values(), default=1)
M_max = 1
##### Unused parameters