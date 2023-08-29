from typing import List, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from .predicate import Predicate
from .models import ModelAPI

##### Incorrect recourses counting.

def incorrectRecoursesIfThen(ifclause: Predicate, thenclause: Predicate, X_aff: DataFrame, model: ModelAPI) -> int:
    X_aff_covered_bool = (X_aff[ifclause.features] == ifclause.values).all(axis=1)
    X_aff_covered = X_aff[X_aff_covered_bool].copy()
    if X_aff_covered.shape[0] == 0:
        raise ValueError("Assuming non-negative frequent itemset threshold, total absence of covered instances should be impossible!")
    
    X_aff_covered[thenclause.features] = thenclause.values

    preds = model.predict(X_aff_covered)
    return np.shape(preds)[0] - np.sum(preds)

def incorrectRecoursesIfThen_bins(ifclause: Predicate, thenclause: Predicate, X_aff: DataFrame, model: ModelAPI, num_features: List[str]) -> int:
    if_nonnumeric_feats = [feat for feat in ifclause.features if feat not in num_features]
    if_nonnumeric_vals = [val for feat, val in zip(ifclause.features, ifclause.values) if feat not in num_features]
    if_numeric_feats = [feat for feat in ifclause.features if feat in num_features]
    if_numeric_vals = [val for feat, val in zip(ifclause.features, ifclause.values) if feat in num_features]

    X_aff_covered_bool_nonnumeric = (X_aff[if_nonnumeric_feats] == if_nonnumeric_vals).all(axis=1)
    X_aff_covered_bool_numeric = X_aff_covered_bool_nonnumeric.map(lambda x: True)
    if_dict = ifclause.to_dict()
    for feat in if_numeric_feats:
        interval = if_dict[feat]
        assert isinstance(interval, pd.Interval)
        indicator = (X_aff[feat] > interval.left) & (X_aff[feat] <= interval.right)
        X_aff_covered_bool_numeric &= indicator
    X_aff_covered = X_aff[X_aff_covered_bool_nonnumeric & X_aff_covered_bool_numeric].copy()

    if X_aff_covered.shape[0] == 0:
        raise ValueError("Assuming non-negative frequent itemset threshold, total absence of covered instances should be impossible!")
    
    then_nonnumeric_feats = [feat for feat in thenclause.features if feat not in num_features]
    assert then_nonnumeric_feats == if_nonnumeric_feats
    then_nonnumeric_vals = [val for feat, val in zip(thenclause.features, thenclause.values) if feat not in num_features]
    then_numeric_feats = [feat for feat in thenclause.features if feat in num_features]
    assert then_numeric_feats == if_numeric_feats
    then_numeric_vals = [val for feat, val in zip(thenclause.features, thenclause.values) if feat in num_features]
    
    X_aff_covered[then_nonnumeric_feats] = then_nonnumeric_vals
    then_dict = thenclause.to_dict()
    for feat in then_numeric_feats:
        if_interval = if_dict[feat]
        assert isinstance(if_interval, pd.Interval)
        then_interval = then_dict[feat]
        assert isinstance(then_interval, pd.Interval)

        slope = (then_interval.right - then_interval.left) / (if_interval.right - if_interval.left)
        lin_map = lambda x: slope * (x - if_interval.left) + then_interval.left

        X_aff_covered[feat] = lin_map(X_aff_covered[feat])

    preds = model.predict(X_aff_covered)
    return np.shape(preds)[0] - np.sum(preds)

##### Cost metrics for a group of one if (i.e. one subpopulation) and several recourses

def if_group_cost_min_change_correctness_threshold(
    ifclause: Predicate,
    then_corrs_costs: List[Tuple[Predicate, float, float]],
    cor_thres: float = 0.5
) -> float:
    costs = np.array([
        cost for _then, cor, cost in then_corrs_costs if cor >= cor_thres
        ])
    if costs.size > 0:
        ret = costs.min()
    else:
        ret = np.inf
    return ret

def if_group_cost_recoursescount_correctness_threshold(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float, float]],
    cor_thres: float = 0.5
) -> float:
    feature_changes = np.array([
        cost for thenclause, cor, cost in thenclauses if cor >= cor_thres
        ])
    return -feature_changes.size

def if_group_cost_recoursescount_correctness_threshold_bins(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float, float]],
    cor_thres: float = 0.5
) -> float:
    feature_changes = np.array([
        cost for thenclause, cor, cost in thenclauses if cor >= cor_thres
        ])
    return -feature_changes.size

def if_group_maximum_correctness(
    ifclause: Predicate,
    then_corrs_costs: List[Tuple[Predicate, float, float]]
) -> float:
    return max(cor for _then, cor, _cost in then_corrs_costs)

def if_group_cost_max_correctness_cost_budget(
    ifclause: Predicate,
    then_corrs_costs: List[Tuple[Predicate, float, float]],
    cost_thres: float = 0.5
) -> float:
    corrs = np.array([
        cor for _then, cor, cost in then_corrs_costs if cost <= cost_thres
        ])
    if corrs.size > 0:
        ret = corrs.max()
    else:
        ret = np.inf
    return ret

def if_group_average_recourse_cost_conditional(
    ifclause: Predicate,
    thens: List[Tuple[Predicate, float, float]]
) -> float:
    mincost_cdf = np.array([corr for then, corr, cost in thens])
    costs = np.array([cost for then, corr, cost in thens])

    mincost_pmf = np.diff(mincost_cdf, prepend=0)

    total_prob = np.sum(mincost_pmf)
    if total_prob > 0:
        return np.dot(mincost_pmf, costs) / np.sum(mincost_pmf)
    else:
        return np.inf

##### Aggregations of if-group cost for all protected values and for all subgroups in a list

if_group_cost_f_t = Callable[[Predicate, List[Tuple[Predicate, float, float]]], float]

def calculate_if_subgroup_costs(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]],
    group_calculator: if_group_cost_f_t
) -> Dict[str, float]:
    return {sg: group_calculator(ifclause, thens) for sg, (_cov, thens) in thenclauses.items()}

def calculate_all_if_subgroup_costs(
    ifclauses: List[Predicate],
    all_thenclauses: List[Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    group_calculator: if_group_cost_f_t
) -> Dict[Predicate, Dict[str, float]]:
    ret: Dict[Predicate, Dict[str, float]] = {}
    for ifclause, thenclauses in zip(ifclauses, all_thenclauses):
        ret[ifclause] = calculate_if_subgroup_costs(ifclause, thenclauses, group_calculator=group_calculator)
    return ret

##### Calculations of discrepancies between the costs of different subgroups (for the same if-group)

def max_intergroup_cost_diff(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]],
    group_calculator: if_group_cost_f_t
) -> float:
    group_costs = list(calculate_if_subgroup_costs(ifclause, thenclauses, group_calculator).values())
    return max(group_costs) - min(group_costs)
