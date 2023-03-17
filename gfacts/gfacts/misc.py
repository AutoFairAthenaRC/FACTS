from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import functools

import numpy as np
from pandas import DataFrame, Series

from .parameters import *
from .models import ModelAPI
from .predicate import Predicate, recIsValid, featureChangePred
from .frequent_itemsets import runApriori, preprocessDataset, aprioriout2predicateList
from .recourse_sets import TwoLevelRecourseSet
from .optimization import optimize_vanilla
from .metrics import incorrectRecoursesIfThen

import matplotlib.pyplot as plt

## Re-exporting
from .optimization import (
    sort_triples_by_max_costdiff,
    sort_triples_by_max_costdiff_ignore_nans,
    sort_triples_by_max_costdiff_ignore_nans_infs
)
from .metrics import (
    calculate_all_if_subgroup_costs,
    if_group_cost_mean_with_correctness,
    if_group_cost_min_change_correctness_threshold,
    if_group_cost_sum_change_correctness_threshold,
    if_group_cost_recoursescount_correctness_threshold
)
from .rule_filters import filter_by_correctness, filter_contained_rules
## Re-exporting


def split_dataset(X: DataFrame, attr: str):
    vals = X[attr].unique()
    grouping = X.groupby(attr)
    return {val: grouping.get_group(val) for val in vals}

def global_counterfactuals_ares(X: DataFrame, model: ModelAPI, sensitive_attribute: str, subsample_size=400, freqitem_minsupp=0.01):
    X_aff_idxs = np.where(model.predict(X) == 0)[0]
    X_aff = X.iloc[X_aff_idxs, :]

    d = X.drop([sensitive_attribute], axis=1)
    freq_itemsets = runApriori(preprocessDataset(d), min_support=freqitem_minsupp)
    freq_itemsets.reset_index()

    RL = aprioriout2predicateList(freq_itemsets)

    SD = list(map(Predicate.from_dict, [
        {sensitive_attribute: val} for val in X[sensitive_attribute].unique()
    ]))

    ifthen_triples = np.random.choice(RL, subsample_size, replace=False) # type: ignore
    affected_sample = X_aff.iloc[np.random.choice(X_aff.shape[0], size=subsample_size, replace=False), :]
    final_rules = optimize_vanilla(SD, ifthen_triples, affected_sample, model)

    return TwoLevelRecourseSet.from_triples(final_rules[0])

def global_counterfactuals_threshold(
    X: DataFrame, model: ModelAPI,
    sensitive_attribute: str,
    threshold_coverage=0.7,
    threshold_correctness=0.8
) -> Dict[str, List[Tuple[Predicate, Predicate, float, float]]]:
    # call function to calculate all valid triples along with coverage and correctness metrics
    ifthens_with_correctness = valid_ifthens_with_coverage_correctness(X, model, sensitive_attribute)

    # all we need now is which are the subgroups (e.g. Male-Female)
    subgroups = np.unique(X[sensitive_attribute])

    # finally, keep triples whose coverage and correct recourse percentage is at least a given threshold
    ifthens_filtered = {sg: [] for sg in subgroups}
    for h, s, ifsupps, thencorrs in ifthens_with_correctness:
        for sg in subgroups:
            if ifsupps[sg] >= threshold_coverage and thencorrs[sg] >= threshold_correctness:
                ifthens_filtered[sg].append((h, s, ifsupps[sg], thencorrs[sg]))
    
    return ifthens_filtered









def intersect_predicate_lists(acc: List[Tuple[Dict[Any, Any], Dict[str, float]]], l2: List[Tuple[Dict[Any, Any], float]], l2_sg: str):
    ret = []
    for i, (pred1, supps) in enumerate(acc):
        for j, (pred2, supp2) in enumerate(l2):
            if pred1 == pred2:
                supps[l2_sg] = supp2
                ret.append((pred1, supps))
    return ret

def affected_unaffected_split(
    X: DataFrame,
    model: ModelAPI
) -> Tuple[DataFrame, DataFrame]:
    # get model predictions
    preds = model.predict(X)
    # find affected individuals
    X_aff_idxs = np.where(preds == 0)[0]
    X_aff = X.iloc[X_aff_idxs, :]

    # find unaffected individuals
    X_unaff_idxs = np.where(preds == 1)[0]
    X_unaff = X.iloc[X_unaff_idxs, :]
    
    return X_aff, X_unaff

def freqitemsets_with_supports(
    X: DataFrame,
    min_support: float = 0.001
) -> Tuple[List[Predicate], List[float]]:
    ret = aprioriout2predicateList(runApriori(preprocessDataset(X), min_support=min_support))
    return ret

def calculate_correctnesses(
        ifthens_withsupp: List[Tuple[Predicate, Predicate, Dict[str, float]]],
        affected_by_subgroup: Dict[str, DataFrame],
        sensitive_attribute: str,
        model: ModelAPI
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    subgroup_names = list(affected_by_subgroup.keys())
    ifthens_with_correctness = []
    for h, s, ifsupps in tqdm(ifthens_withsupp):
        recourse_correctness = {}
        for sg in subgroup_names:
            incorrect_recourses_for_sg = incorrectRecoursesIfThen(h, s, affected_by_subgroup[sg].assign(**{sensitive_attribute: sg}), model)
            covered_sg = ifsupps[sg] * affected_by_subgroup[sg].shape[0]
            inc_sg = incorrect_recourses_for_sg / covered_sg
            recourse_correctness[sg] = 1 - inc_sg

        ifthens_with_correctness.append((h, s, ifsupps, recourse_correctness))
    
    return ifthens_with_correctness

def valid_ifthens_with_coverage_correctness(
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    freqitem_minsupp: float = 0.01,
    missing_subgroup_val: str = "N/A"
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    # throw out all individuals for whom the value of the sensitive attribute is unknown
    X = X[X[sensitive_attribute] != missing_subgroup_val]

    # split into affected-unaffected
    X_aff, X_unaff = affected_unaffected_split(X, model)

    # find descriptors of all sensitive subgroups
    subgroups = np.unique(X[sensitive_attribute])

    # split affected individuals into subgroups
    affected_subgroups = {sg: X_aff[X_aff[sensitive_attribute] == sg].drop([sensitive_attribute], axis=1) for sg in subgroups}

    # calculate frequent itemsets for each subgroup and turn them into predicates
    print("Computing frequent itemsets for each subgroup of the affected instances.",flush=True)
    RLs_and_supports = {sg: freqitemsets_with_supports(affected_sg, min_support=freqitem_minsupp) for sg, affected_sg in tqdm(affected_subgroups.items())}

    # turn RLs into dictionaries for easier comparison
    RLs_supports_dict = {sg: [(dict(zip(p.features, p.values)), supp) for p, supp in zip(*RL_sup)] for sg, RL_sup in RLs_and_supports.items()}

    # intersection of frequent itemsets of all sensitive subgroups
    print("Computing the intersection between the frequent itemsets of each subgroup of the affected instances.",flush=True)
    if len(RLs_supports_dict) < 1:
        raise ValueError("There must be at least 2 subgroups.")
    else:
        sg = subgroups[0]
        RLs_supports = RLs_supports_dict[sg]
        aff_intersection = [(d, {sg: supp}) for d, supp in RLs_supports]
    for sg, RLs in tqdm(RLs_supports_dict.items()):
        if sg == subgroups[0]:
            continue

        aff_intersection = intersect_predicate_lists(aff_intersection, RLs, sg)
    
    aff_intersection = [(Predicate.from_dict(d), supps) for d, supps in aff_intersection]
    
    # Frequent itemsets for the unaffacted (to be used in the then clauses)
    freq_unaffected, _ = freqitemsets_with_supports(X_unaff, min_support=freqitem_minsupp)

    # Filter all if-then pairs to keep only valid
    print("Computing all valid if-then pairs between the common frequent itemsets of each subgroup of the affected instances and the frequent itemsets of the unaffacted instances.",flush=True)
    ifthens = [(h, s, ifsupps) for h, ifsupps in tqdm(aff_intersection) for s in freq_unaffected if recIsValid(h, s)]

    # Calculate incorrectness percentages
    print("Computing correctenesses for all valid if-thens.",flush=True)
    ifthens_with_correctness = calculate_correctnesses(ifthens, affected_subgroups, sensitive_attribute, model)

    return ifthens_with_correctness

def rules2rulesbyif(rules: List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    # group rules based on If clauses, instead of protected subgroups!
    rules_by_if: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]] = {}
    for h, s, covs, cors in rules:
        if h not in rules_by_if:
            rules_by_if[h] = {sg: (cov, []) for sg, cov in covs.items()}
        
        for sg, (_cov, sg_thens) in rules_by_if[h].items():
            sg_thens.append((s, cors[sg]))
    
    return rules_by_if



def select_rules_subset(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    metric: str = "weighted-average",
    sort_strategy: str = "abs-diff-decr",
    top_count: int = 10,
    filter_sequence: Optional[List[str]] = None,
    cor_threshold: float = 0.5,
    params: ParameterProxy = ParameterProxy()
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    # step 1: sort according to metric
    metrics = {
        "weighted-average": if_group_cost_mean_with_correctness,
        "min-above-thr": functools.partial(if_group_cost_min_change_correctness_threshold, cor_thres=cor_threshold),
        "total-above-thr": functools.partial(if_group_cost_sum_change_correctness_threshold, cor_thres=cor_threshold),
        "num-above-thr": functools.partial(if_group_cost_recoursescount_correctness_threshold, cor_thres=cor_threshold)
    }
    sorting_functions = {
        "abs-diff-decr": sort_triples_by_max_costdiff,
        "abs-diff-decr-ignore-forall-subgroups-empty": sort_triples_by_max_costdiff_ignore_nans,
        "abs-diff-decr-ignore-exists-subgroup-empty": sort_triples_by_max_costdiff_ignore_nans_infs
    }
    metric_fn = metrics[metric]
    sort_fn = sorting_functions[sort_strategy]
    rules_sorted = sort_fn(rulesbyif, group_calculator=metric_fn, params=params)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    # step 3 (optional): filtering
    filters = {
        "remove-contained": filter_contained_rules,
        "remove-below-thr": functools.partial(filter_by_correctness, threshold=cor_threshold)
    }
    if filter_sequence is not None:
        for filter in filter_sequence:
            top_rules = filters[filter](top_rules)

    return top_rules








def sort_thens_by_cost(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    params: ParameterProxy = ParameterProxy()
) -> List[Tuple[Predicate, float, float]]:
    withcosts = [(thenclause, cor, float(featureChangePred(ifclause, thenclause, params))) for thenclause, cor in thenclauses]
    thens_sorted_by_cost = sorted(withcosts, key=lambda c: (c[2], -c[1]))
    return thens_sorted_by_cost

def cumcorr(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    X: DataFrame,
    model: ModelAPI,
    params: ParameterProxy = ParameterProxy()
) -> List[Tuple[Predicate, float, float]]:
    withcosts = [(thenclause, cor, featureChangePred(ifclause, thenclause, params)) for thenclause, cor in thenclauses]
    thens_sorted_by_cost = sorted(withcosts, key=lambda c: c[2])

    X_covered_bool = (X[ifclause.features] == ifclause.values).all(axis=1)
    X_covered = X[X_covered_bool]
    covered_count = X_covered.shape[0]

    cumcorrs = []
    for thenclause, _cor, _cost in thens_sorted_by_cost:
        X_temp = X_covered.copy()
        X_temp[thenclause.features] = thenclause.values
        preds = model.predict(X_temp)

        corrected_count = np.sum(preds)
        cumcorrs.append(corrected_count)
        X_covered = X_covered[~ preds.astype(bool)] # type: ignore

    cumcorrs = np.array(cumcorrs).cumsum() / covered_count
    print(f"{cumcorrs=}")
    updated_thens = [(thenclause, cumcor, float(cost)) for (thenclause, _cor, cost), cumcor in zip(thens_sorted_by_cost, cumcorrs)]

    return updated_thens

def create_correctness_plot(
    costs: List[float],
    correctnesses: List[float]
):
    fig, ax = plt.subplots()
    ax.plot(costs, correctnesses)
    ax.set_xlabel("Cost of change")
    ax.set_ylabel("Correctness percentage")
    return fig
