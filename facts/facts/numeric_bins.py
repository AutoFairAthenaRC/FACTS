from typing import List, Tuple, Dict, Callable
import functools
from tqdm import tqdm

import numpy as np
import pandas as pd
from pandas import DataFrame

from .parameters import ParameterProxy
from .predicate import Predicate
from .models import ModelAPI
from .metrics import (
    if_group_cost_recoursescount_correctness_threshold_bins,
    if_group_total_correctness,
    if_group_cost_min_change_correctness_cumulative_threshold,
    if_group_cost_change_cumulative_threshold,
    if_group_average_recourse_cost_conditional,
    calculate_all_if_subgroup_costs_cumulative
)
from .optimization import (
    sort_triples_by_max_costdiff_generic_cumulative
)
from .rule_filters import (
    filter_contained_rules_keep_max_bias_cumulative,
    filter_by_correctness_cumulative,
    filter_by_cost_cumulative,
    keep_cheapest_rules_above_cumulative_correctness_threshold,
    keep_only_minimum_change_bins,
    delete_fair_rules_cumulative
)

def add_cost_to_rules(
    valid_if_thens: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    X_aff: DataFrame,
    num_features: List[str],
    sensitive_attribute: str,
    params: ParameterProxy = ParameterProxy()
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = dict()
    for ifclause, all_thens in valid_if_thens.items():
        ret[ifclause] = dict()
        for sg, (cov, sg_thens) in all_thens.items():
            group = X_aff[X_aff[sensitive_attribute] == sg]
            covered_individuals = find_covered_individuals(ifclause, group, num_features)
            preds_with_costs = []
            for thenclause, cor in sg_thens:
                cost = calculate_cost_of_change(ifclause, thenclause, covered_individuals, num_features, params)
                preds_with_costs.append((thenclause, cor, cost))
            ret[ifclause][sg] = (cov, preds_with_costs)
    
    return ret

def find_covered_individuals(ifclause: Predicate, X_aff: DataFrame, num_features: List[str]) -> DataFrame:
    if_nonnumeric_feats = [feat for feat in ifclause.features if feat not in num_features]
    if_nonnumeric_vals = [val for feat, val in zip(ifclause.features, ifclause.values) if feat not in num_features]
    if_numeric_feats = [feat for feat in ifclause.features if feat in num_features]
    # if_numeric_vals = [val for feat, val in zip(ifclause.features, ifclause.values) if feat in num_features]

    X_aff_covered_bool_nonnumeric = (X_aff[if_nonnumeric_feats] == if_nonnumeric_vals).all(axis=1)
    X_aff_covered_bool_numeric = X_aff_covered_bool_nonnumeric.map(lambda x: True)
    if_dict = ifclause.to_dict()
    for feat in if_numeric_feats:
        interval = if_dict[feat]
        assert isinstance(interval, pd.Interval)
        indicator = (X_aff[feat] > interval.left) & (X_aff[feat] <= interval.right)
        X_aff_covered_bool_numeric &= indicator
    X_aff_covered = X_aff[X_aff_covered_bool_nonnumeric & X_aff_covered_bool_numeric].copy()

    return X_aff_covered

def move_to_counterfactuals(X: DataFrame, ifclause: Predicate, thenclause: Predicate, num_features: List[str]):
    if_nonnumeric_feats = [feat for feat in ifclause.features if feat not in num_features]
    if_nonnumeric_vals = [val for feat, val in zip(ifclause.features, ifclause.values) if feat not in num_features]
    if_numeric_feats = [feat for feat in ifclause.features if feat in num_features]
    if_numeric_vals = [val for feat, val in zip(ifclause.features, ifclause.values) if feat in num_features]

    then_nonnumeric_feats = [feat for feat in thenclause.features if feat not in num_features]
    assert then_nonnumeric_feats == if_nonnumeric_feats
    then_nonnumeric_vals = [val for feat, val in zip(thenclause.features, thenclause.values) if feat not in num_features]
    then_numeric_feats = [feat for feat in thenclause.features if feat in num_features]
    assert then_numeric_feats == if_numeric_feats
    then_numeric_vals = [val for feat, val in zip(thenclause.features, thenclause.values) if feat in num_features]

    X[then_nonnumeric_feats] = then_nonnumeric_vals
    if_dict = ifclause.to_dict()
    then_dict = thenclause.to_dict()
    for feat in then_numeric_feats:
        if_interval = if_dict[feat]
        assert isinstance(if_interval, pd.Interval)
        then_interval = then_dict[feat]
        assert isinstance(then_interval, pd.Interval)

        slope = (then_interval.right - then_interval.left) / (if_interval.right - if_interval.left)
        lin_map = lambda x: slope * (x - if_interval.left) + then_interval.left

        X[feat] = lin_map(X[feat])


def calculate_cost_of_change(
    ifclause: Predicate,
    thenclause: Predicate,
    covered_individuals: DataFrame,
    num_features: List[str],
    params: ParameterProxy = ParameterProxy()
) -> float:
    total = 0.
    for i, f in enumerate(ifclause.features):
        val1 = ifclause.values[i]
        val2 = thenclause.values[i]
        if f not in num_features:
            costChange = params.featureChanges[f](val1, val2)
        else:
            assert isinstance(val1, pd.Interval) and isinstance(val2, pd.Interval)
            if_interval = val1
            then_interval = val2
            slope = (then_interval.right - then_interval.left) / (if_interval.right - if_interval.left)
            lin_map = lambda x: slope * (x - if_interval.left) + then_interval.left
            start_vals = covered_individuals[f]
            target_vals = lin_map(start_vals)
            costChange = np.mean(params.featureChanges[f](start_vals, target_vals))
        total += float(costChange)
    return total

def select_rules_subset_bins(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    metric: str = "num-above-thr",
    sort_strategy: str = "abs-diff-decr",
    top_count: int = 10,
    filter_sequence: List[str] = [],
    cor_threshold: float = 0.5,
    cost_threshold: float = 0.5,
    secondary_sorting_objectives: List[str] = [],
    params: ParameterProxy = ParameterProxy(),
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    Dict[Predicate, Dict[str, float]],
]:
    # step 1: sort according to metric
    metrics: Dict[
        str, Callable[[Predicate, List[Tuple[Predicate, float, float]], ParameterProxy], float]
    ] = {
        "min-above-thr": functools.partial(
            if_group_cost_min_change_correctness_cumulative_threshold, cor_thres=cor_threshold
        ),
        "num-above-thr": functools.partial(
            if_group_cost_recoursescount_correctness_threshold_bins, cor_thres=cor_threshold
        ),
        "total-correctness": if_group_total_correctness,
        "min-above-corr": functools.partial(
            if_group_cost_min_change_correctness_cumulative_threshold, cor_thres=cor_threshold
        ),
        "max-upto-cost": functools.partial(
            if_group_cost_change_cumulative_threshold, cost_thres=cost_threshold
        ),
        "fairness-of-mean-recourse-conditional": if_group_average_recourse_cost_conditional
    }
    sorting_functions = {
        "generic-sorting": functools.partial(
            sort_triples_by_max_costdiff_generic_cumulative,
            ignore_nans=False,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "generic-sorting-ignore-forall-subgroups-empty": functools.partial(
            sort_triples_by_max_costdiff_generic_cumulative,
            ignore_nans=True,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "generic-sorting-ignore-exists-subgroup-empty": functools.partial(
            sort_triples_by_max_costdiff_generic_cumulative,
            ignore_nans=True,
            ignore_infs=True,
            secondary_objectives=secondary_sorting_objectives,
        ),
    }
    metric_fn = metrics[metric]
    sort_fn = sorting_functions[sort_strategy]
    rules_sorted = sort_fn(rulesbyif, group_calculator=metric_fn, params=params)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    # keep also the aggregate costs of the then-blocks of the top rules
    costs = calculate_all_if_subgroup_costs_cumulative(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        group_calculator=metric_fn,
        params=params,
    )

    # step 3 (optional): filtering
    filters: Dict[
        str,
        Callable[
            [Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]],
            Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
        ],
    ] = {
        "remove-contained": functools.partial(
            filter_contained_rules_keep_max_bias_cumulative, subgroup_costs=costs
        ),
        "remove-below-thr": functools.partial(
            filter_by_correctness_cumulative, threshold=cor_threshold
        ),
        "remove-fair-rules": functools.partial(delete_fair_rules_cumulative, subgroup_costs=costs),
        "keep-only-min-change": keep_only_minimum_change_bins,
        "remove-above-thr-cost": functools.partial(
            filter_by_cost_cumulative, threshold=cost_threshold
        ),
        "keep-cheap-rules-above-thr-cor": functools.partial(
            keep_cheapest_rules_above_cumulative_correctness_threshold, threshold=cor_threshold
        ),
    }
    for single_filter in filter_sequence:
        top_rules = filters[single_filter](top_rules)

    return top_rules, costs

def cum_corr_costs(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float, float]],
    X: DataFrame,
    model: ModelAPI,
    num_features: List[str]
) -> List[Tuple[Predicate, float, float]]:
    # sort by increasing cost
    thens_sorted_by_cost = sorted(thenclauses, key=lambda c: (c[2], c[1]))

    X_covered = find_covered_individuals(ifclause, X, num_features=num_features)
    covered_count = X_covered.shape[0]

    cumcorrs = []
    for thenclause, _cor, _cost in thens_sorted_by_cost:
        if X_covered.shape[0] == 0:
            cumcorrs.append(0)
            continue
        X_temp = X_covered.copy()
        move_to_counterfactuals(X_temp, ifclause, thenclause, num_features)
        preds = model.predict(X_temp)

        corrected_count = np.sum(preds)
        cumcorrs.append(corrected_count)
        X_covered = X_covered[~preds.astype(bool)]  # type: ignore

    cumcorrs = np.array(cumcorrs).cumsum() / covered_count
    updated_thens = [
        (thenclause, cumcor, float(cost))
        for (thenclause, _cor, cost), cumcor in zip(thens_sorted_by_cost, cumcorrs)
    ]

    return updated_thens

def cum_corr_costs_all(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    num_features: List[str]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    X_affected: DataFrame = X[model.predict(X) == 0]  # type: ignore
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = {}
    for ifclause, all_thens in tqdm(rulesbyif.items()):
        all_thens_new: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] = {}
        for sg, (cov, thens) in all_thens.items():
            subgroup_affected = X_affected[X_affected[sensitive_attribute] == sg]
            all_thens_new[sg] = (cov, cum_corr_costs(
                ifclause, thens, subgroup_affected, model, num_features
            ))
        ret[ifclause] = all_thens_new
    return ret

