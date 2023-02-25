from typing import List, Tuple, Dict
import functools

import numpy as np
from pandas import DataFrame, Series

from .parameters import *
from .models import ModelAPI
from .frequent_itemsets import runApriori, preprocessDataset, aprioriout2predicateList
from .recourse_sets import TwoLevelRecourseSet
from .metrics import incorrectRecoursesIfThen
from .formatting import to_bold_str

## Re-exporting
from .optimization import optimize_vanilla
from .predicate import Predicate, recIsValid, featureChangePred
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

def intersect_predicate_lists(acc: List[Tuple[Dict[Any, Any], Dict[str, float]]], l2: List[Tuple[Dict[Any, Any], float]], l2_sg: str):
    ret = []
    for i, (pred1, supps) in enumerate(acc):
        for j, (pred2, supp2) in enumerate(l2):
            if pred1 == pred2:
                supps[l2_sg] = supp2
                ret.append((pred1, supps))
    return ret

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
    RLs_and_supports = {sg: freqitemsets_with_supports(affected_sg, min_support=freqitem_minsupp) for sg, affected_sg in affected_subgroups.items()}

    # turn RLs into dictionaries for easier comparison
    RLs_supports_dict = {sg: [(dict(zip(p.features, p.values)), supp) for p, supp in zip(*RL_sup)] for sg, RL_sup in RLs_and_supports.items()}

    # intersection of frequent itemsets of all sensitive subgroups
    if len(RLs_supports_dict) < 1:
        raise ValueError("There must be at least 2 subgroups.")
    else:
        sg = subgroups[0]
        RLs_supports = RLs_supports_dict[sg]
        aff_intersection = [(d, {sg: supp}) for d, supp in RLs_supports]
    for sg, RLs in RLs_supports_dict.items():
        if sg == subgroups[0]:
            continue

        aff_intersection = intersect_predicate_lists(aff_intersection, RLs, sg)
    
    aff_intersection = [(Predicate.from_dict(d), supps) for d, supps in aff_intersection]
    
    # Frequent itemsets for the unaffacted (to be used in the then clauses)
    freq_unaffected, _ = freqitemsets_with_supports(X_unaff, min_support=freqitem_minsupp)

    # Filter all if-then pairs to keep only valid
    ifthens = [(h, s, ifsupps) for h, ifsupps in aff_intersection for s in freq_unaffected if recIsValid(h, s)]

    # Calculate incorrectness percentages
    from tqdm import tqdm
    ifthens_with_correctness = []
    for h, s, ifsupps in tqdm(ifthens):
        recourse_correctness = {}
        for sg in subgroups:
            incorrect_recourses_for_sg = incorrectRecoursesIfThen(h, s, affected_subgroups[sg].assign(**{sensitive_attribute: sg}), model)
            covered_sg = ifsupps[sg] * affected_subgroups[sg].shape[0]
            inc_sg = incorrect_recourses_for_sg / covered_sg
            recourse_correctness[sg] = 1 - inc_sg

        ifthens_with_correctness.append((h, s, ifsupps, recourse_correctness))
    
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






def if_group_cost_mean_with_correctness(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    params: ParameterProxy = ParameterProxy()
) -> float:
    return np.mean([cor * featureChangePred(ifclause, thenclause, params=params) for thenclause, cor in thenclauses]).astype(float)

def if_group_cost_mean_correctness_weighted(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    params: ParameterProxy = ParameterProxy()
) -> float:
    feature_changes = np.array([featureChangePred(ifclause, thenclause, params=params) for thenclause, _ in thenclauses])
    corrs = np.array([cor for _, cor in thenclauses])
    return np.average(feature_changes, weights=corrs).astype(float)

def if_group_cost_recoursesnum_correctness_threshold(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy()
) -> float:
    feature_changes = np.array([
        featureChangePred(ifclause, thenclause, params=params) for thenclause, cor in thenclauses if cor >= cor_thres
        ])
    try:
        ret = feature_changes.min()
    except ValueError:
        ret = np.inf
    return ret

def if_group_cost_max_change_correctness_threshold(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy()
) -> float:
    feature_changes = np.array([
        featureChangePred(ifclause, thenclause, params=params) for thenclause, cor in thenclauses if cor >= cor_thres
        ])
    return feature_changes.size

if_group_cost_f_t = Callable[[Predicate, List[Tuple[Predicate, float]]], float]

def calculate_if_subgroup_costs(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    group_calculator: if_group_cost_f_t = if_group_cost_mean_with_correctness,
    **kwargs
) -> Dict[str, float]:
    return {sg: group_calculator(ifclause, thens, **kwargs) for sg, (_cov, thens) in thenclauses.items()}

def calculate_all_if_subgroup_costs(
    ifclauses: List[Predicate],
    all_thenclauses: List[Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    **kwargs
) -> Dict[Predicate, Dict[str, float]]:
    ret: Dict[Predicate, Dict[str, float]] = {}
    for ifclause, thenclauses in zip(ifclauses, all_thenclauses):
        ret[ifclause] = calculate_if_subgroup_costs(ifclause, thenclauses, **kwargs)
    return ret

def calculate_cost_difference_2groups(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    group1: str = "0",
    group2: str = "1",
    params: ParameterProxy = ParameterProxy()
) -> float:
    group_costs = calculate_if_subgroup_costs(ifclause, thenclauses, params=params)
    return abs(group_costs[group1] - group_costs[group2])

def sort_triples_by_costdiff_2groups(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    group1: str = "0",
    group2: str = "1",
    params: ParameterProxy = ParameterProxy()
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    def apply_calc(ifthens):
        return calculate_cost_difference_2groups(ifthens[0], ifthens[1], group1, group2, params)
    ret = sorted(rulesbyif.items(), key=apply_calc, reverse=True)
    return ret

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

def max_intergroup_cost_diff(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    **kwargs
) -> float:
    group_costs = list(calculate_if_subgroup_costs(ifclause, thenclauses, **kwargs).values())
    return max(group_costs) - min(group_costs)

def sort_triples_by_max_costdiff(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    **kwargs
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    def apply_calc(ifthens):
        ifclause = ifthens[0]
        thenclauses = ifthens[1]
        return max_intergroup_cost_diff(ifclause, thenclauses, **kwargs)
    ret = sorted(rulesbyif.items(), key=apply_calc, reverse=True)
    return ret

def sort_triples_by_max_costdiff_ignore_nans(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    **kwargs
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    def apply_calc(ifthens):
        ifclause = ifthens[0]
        thenclauses = ifthens[1]
        max_costdiff = max_intergroup_cost_diff(ifclause, thenclauses, **kwargs)
        if np.isnan(max_costdiff):
            return -np.inf
        else:
            return max_costdiff
    ret = sorted(rulesbyif.items(), key=apply_calc, reverse=True)
    return ret

def sort_triples_by_max_costdiff_ignore_nans_infs(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    **kwargs
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    def apply_calc(ifthens):
        ifclause = ifthens[0]
        thenclauses = ifthens[1]
        max_costdiff = max_intergroup_cost_diff(ifclause, thenclauses, **kwargs)
        if np.isnan(max_costdiff) or np.isinf(max_costdiff):
            return -np.inf
        else:
            return max_costdiff
    max_diffs = np.array([apply_calc(ifthens) for ifthens in rulesbyif.items()])

    # TODO: change this to something more helpful
    if np.isinf(max_diffs).all():
        print(to_bold_str("Dommage monsieur!"))
    ret = sorted(rulesbyif.items(), key=apply_calc, reverse=True)
    return ret

def filter_by_correctness(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    threshold: float = 0.5
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    # ret = {ifclause: {sg: (cov, [(then, cor) for then, cor in thens if cor >= threshold]) for sg, (cov, thens) in thenclauses.items()} for ifclause, thenclauses in rulesbyif.items()}
    ret = dict()
    for ifclause, thenclauses in rulesbyif.items():
        filtered_thenclauses = dict()
        for sg, (cov, sg_thens) in thenclauses.items():
            filtered_thens = [(then, cor) for then, cor in sg_thens if cor >= threshold]
            filtered_thenclauses[sg] = (cov, filtered_thens)
        ret[ifclause] = filtered_thenclauses
    return ret

def filter_contained_rules(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    threshold: float = 0.5
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    ret = dict()
    for ifclause, thenclauses in rulesbyif.items():
        flag_keep = True
        allthens = [then for _sg, (_cov, sg_thens) in thenclauses.items() for then, _cor in sg_thens]
        for otherif, _ in rulesbyif.items():
            if not ifclause.contains(otherif):
                continue
            extra_features = list(set(ifclause.features) - set(otherif.features))
            if len(extra_features) == 0:
                continue
            allthens_relevant_values = [tuple(then.to_dict()[feat] for feat in extra_features) for then in allthens]
            if Series(allthens_relevant_values).unique().size == 1:
                flag_keep = False

        if flag_keep:
            ret[ifclause] = thenclauses
    return ret