from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import functools

import numpy as np
from pandas import DataFrame

from mlxtend.preprocessing import minmax_scaling

from .parameters import *
from .models import ModelAPI
from .predicate import Predicate, recIsValid, featureChangePred , drop_two_above
from .frequent_itemsets import runApriori, preprocessDataset, aprioriout2predicateList
from .recourse_sets import TwoLevelRecourseSet
from .metrics import (
    incorrectRecoursesIfThen,
    if_group_cost_mean_with_correctness,
    if_group_cost_min_change_correctness_threshold,
    if_group_cost_mean_change_correctness_threshold,
    if_group_cost_recoursescount_correctness_threshold
)
from .optimization import (
    optimize_vanilla,
    sort_triples_by_max_costdiff,
    sort_triples_by_max_costdiff_ignore_nans,
    sort_triples_by_max_costdiff_ignore_nans_infs
)
from .rule_filters import filter_by_correctness, filter_contained_rules, delete_fair_rules, keep_only_minimum_change

## Re-exporting
from .metrics import calculate_all_if_subgroup_costs
from .formatting import plot_aggregate_correctness, print_recourse_report
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
    min_support: float = 0.01
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


def aff_intersection_version_1(RLs_and_supports, subgroups):
    RLs_supports_dict = {sg: [(dict(zip(p.features, p.values)), supp) for p, supp in zip(
        *RL_sup)] for sg, RL_sup in RLs_and_supports.items()}

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

    aff_intersection = [(Predicate.from_dict(d), supps)
                        for d, supps in aff_intersection]

    return aff_intersection


def aff_intersection_version_2(RLs_and_supports, subgroups):
    RLs_supports_dict = {sg: {tuple(sorted(zip(p.features, p.values))): supp for p, supp in zip(
        *RL_sup)} for sg, RL_sup in RLs_and_supports.items()}

    if len(RLs_supports_dict) < 1:
        raise ValueError("There must be at least 2 subgroups.")
    else:

        aff_intersection = []

        for i in tqdm(range(len(subgroups) - 1)):
            sg1 = subgroups[i]
            for value, supp in RLs_supports_dict[sg1].items():
                in_all = True
                supp_dict = {sg1: supp}
                for j in range(i + 1, len(subgroups)):
                    sg2 = subgroups[j]
                    if value not in RLs_supports_dict[sg2]:
                        in_all = False
                    supp_dict[sg2] = RLs_supports_dict[sg2].get(value)

                if in_all == True:
                    feat_dict = {}
                    for feat in value:
                        feat_dict[feat[0]] = feat[1]
                    aff_intersection.append((feat_dict, supp_dict))

    aff_intersection = [(Predicate.from_dict(d), supps)
                        for d, supps in aff_intersection]

    return aff_intersection


def valid_ifthens_with_coverage_correctness(
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    freqitem_minsupp: float = 0.01,
    missing_subgroup_val: str = "N/A",
    drop_infeasible: bool = True,
    drop_above: bool = True
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
    print("Computing frequent itemsets for each subgroup of the affected instances.", flush=True)
    RLs_and_supports = {sg: freqitemsets_with_supports(affected_sg, min_support=freqitem_minsupp) for sg, affected_sg in tqdm(affected_subgroups.items())}

    # aff_intersection_1 = aff_intersection_version_1(
    #     RLs_and_supports, subgroups)
    aff_intersection_2 = aff_intersection_version_2(
        RLs_and_supports, subgroups)

    # print(len(aff_intersection_1), len(aff_intersection_2))
    # if aff_intersection_1 != aff_intersection_2:
    #     print("ERRRROOROROROROROROROROROR")

    aff_intersection = aff_intersection_2

    # intersection of frequent itemsets of all sensitive subgroups
    
    # Frequent itemsets for the unaffacted (to be used in the then clauses)
    freq_unaffected, _ = freqitemsets_with_supports(X_unaff, min_support=freqitem_minsupp)

    # Filter all if-then pairs to keep only valid
    print("Computing all valid if-then pairs between the common frequent itemsets of each subgroup of the affected instances and the frequent itemsets of the unaffacted instances.",flush=True)
    ifthens = [(h, s, ifsupps) for h, ifsupps in tqdm(aff_intersection) for s in freq_unaffected if recIsValid(h, s, affected_subgroups[subgroups[0]], drop_infeasible)]

    #keep ifs that have change on features of max value 2
    if drop_above == True:
        age = [val.left for val in X.age.unique()]
        age.sort()
        ifthens = [(ifs,then,cov) for ifs,then,cov in ifthens if drop_two_above(ifs,then,age)]

    # Calculate incorrectness percentages
    print("Computing correctenesses for all valid if-thens.", flush=True)
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

def rulesbyif2rules(rules_by_if: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    rules: List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]] = []
    for ifclause, thenclauses in rules_by_if.items():
        then_covs = dict()
        then_cors = dict()
        for sg, (cov, thens) in thenclauses.items():
            for then, cor in thens:
                if then in then_covs:
                    then_covs[then][sg] = cov
                else:
                    then_covs[then] = {sg: cov}
                if then in then_cors:
                    then_cors[then][sg] = cor
                else:
                    then_cors[then] = {sg: cor}
        
        for sg, (_cov, thens) in thenclauses.items():
            for then, _cor in thens:
                rules.append((ifclause, then, then_covs[then], then_cors[then]))
    return rules




def select_rules_subset(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    metric: str = "weighted-average",
    sort_strategy: str = "abs-diff-decr",
    top_count: int = 10,
    filter_sequence: Optional[List[str]] = None,
    cor_threshold: float = 0.5,
    secondary_sorting: bool = False,
    params: ParameterProxy = ParameterProxy()
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    Dict[Predicate, Dict[str, float]]
]:
    # step 1: sort according to metric
    metrics: Dict[str, Callable[[Predicate, List[Tuple[Predicate, float]], ParameterProxy], float]] = {
        "weighted-average": if_group_cost_mean_with_correctness,
        "min-above-thr": functools.partial(if_group_cost_min_change_correctness_threshold, cor_thres=cor_threshold),
        "mean-above-thr": functools.partial(if_group_cost_mean_change_correctness_threshold, cor_thres=cor_threshold),
        "num-above-thr": functools.partial(if_group_cost_recoursescount_correctness_threshold, cor_thres=cor_threshold)
    }
    sorting_functions = {
        "abs-diff-decr": sort_triples_by_max_costdiff,
        "abs-diff-decr-ignore-forall-subgroups-empty": functools.partial(sort_triples_by_max_costdiff_ignore_nans, use_secondary_objective=secondary_sorting),
        "abs-diff-decr-ignore-exists-subgroup-empty": functools.partial(sort_triples_by_max_costdiff_ignore_nans_infs, use_secondary_objective=secondary_sorting)
    }
    metric_fn = metrics[metric]
    sort_fn = sorting_functions[sort_strategy]
    rules_sorted = sort_fn(rulesbyif, group_calculator=metric_fn, params=params)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    # keep also the aggregate costs of the then-blocks of the top rules
    costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        group_calculator=metric_fn,
        params=params
    )

    # step 3 (optional): filtering
    filters: Dict[str, Callable[[Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]], Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]] = {
        "remove-contained": filter_contained_rules,
        "remove-below-thr": functools.partial(filter_by_correctness, threshold=cor_threshold),
        "remove-fair-rules": functools.partial(delete_fair_rules, subgroup_costs=costs),
        "keep-only-min-change": functools.partial(keep_only_minimum_change, params=params)
    }
    if filter_sequence is not None:
        for filter in filter_sequence:
            top_rules = filters[filter](top_rules)

    return top_rules, costs








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
) -> List[Tuple[float, float]]:
    withcosts = [(thenclause, cor, featureChangePred(ifclause, thenclause, params)) for thenclause, cor in thenclauses]
    thens_sorted_by_cost = sorted(withcosts, key=lambda c: (c[2], c[1]))

    X_covered_bool = (X[ifclause.features] == ifclause.values).all(axis=1)
    X_covered = X[X_covered_bool]
    covered_count = X_covered.shape[0]

    cumcorrs = []
    for thenclause, _cor, _cost in thens_sorted_by_cost:
        if X_covered.shape[0] == 0:
            cumcorrs.append(0)
            continue
        X_temp = X_covered.copy()
        X_temp[thenclause.features] = thenclause.values
        preds = model.predict(X_temp)

        corrected_count = np.sum(preds)
        cumcorrs.append(corrected_count)
        X_covered = X_covered[~ preds.astype(bool)] # type: ignore

    cumcorrs = np.array(cumcorrs).cumsum() / covered_count
    updated_thens = [(cumcor, float(cost)) for (_thenclause, _cor, cost), cumcor in zip(thens_sorted_by_cost, cumcorrs)]

    return updated_thens

def cumcorr_all(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    params: ParameterProxy = ParameterProxy()
) -> Dict[Predicate, Dict[str, List[Tuple[float, float]]]]:
    X_affected: DataFrame = X[model.predict(X) == 0] # type: ignore
    ret = {}
    for ifclause, all_thens in rulesbyif.items():
        all_thens_new = {}
        for sg, (_cov, thens) in all_thens.items():
            subgroup_affected = X_affected[X_affected[sensitive_attribute] == sg]
            all_thens_new[sg] = cumcorr(ifclause, thens, subgroup_affected, model, params=params)
        ret[ifclause] = all_thens_new
    return ret


def feature_change_builder(
    X: DataFrame,
    num_cols: List[str],
    cate_cols: List[str],
    feature_weights: Dict[str, int],
    num_normalization: bool = False,
    feats_to_normalize: Optional[List[str]] = None
) -> Dict[str, Callable[[Any, Any], int]]:
    def feature_change_cate(v1, v2, weight):
        return (0 if v1 == v2 else 1) * weight
    def feature_change_num(v1, v2, weight):
        return abs(v1 - v2) * weight

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

    ret_cate = {col: functools.partial(feature_change_cate, weight=feature_weights.get(col, 1)) for col in cate_cols}
    ret_num = {col: functools.partial(feature_change_num, weight=weight_multipliers[col] * feature_weights.get(col, 1)) for col in num_cols}
    return {**ret_cate, **ret_num}


