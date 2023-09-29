from tqdm import tqdm
from typing import List, Tuple, Dict, Sequence
import functools

import numpy as np
import pandas as pd
from pandas import DataFrame

# from mlxtend.preprocessing import minmax_scaling

from .parameters import *
from .models import ModelAPI
from .predicate import Predicate, recIsValid, featureChangePred, drop_two_above
from .frequent_itemsets import runApriori, preprocessDataset, aprioriout2predicateList
from .metrics import (
    incorrectRecoursesIfThen,
    incorrectRecoursesIfThen_bins,
    if_group_cost_min_change_correctness_threshold,
    if_group_cost_recoursescount_correctness_threshold,
    if_group_maximum_correctness,
    if_group_cost_max_correctness_cost_budget,
    calculate_all_if_subgroup_costs,
    if_group_average_recourse_cost_conditional
)
from .optimization import (
    sort_triples_by_max_costdiff,
    sort_triples_KStest
)
from .rule_filters import (
    remove_rules_below_correctness_threshold,
    filter_contained_rules_simple,
    filter_contained_rules_keep_max_bias,
    delete_fair_rules,
    keep_only_minimum_change,
    remove_rules_above_cost_budget,
    keep_cheapest_rules_above_cumulative_correctness_threshold
)

# Re-exporting
from .formatting import print_recourse_report
# Re-exporting


def split_dataset(X: DataFrame, attr: str):
    vals = X[attr].unique()
    grouping = X.groupby(attr)
    return {val: grouping.get_group(val) for val in vals}

def affected_unaffected_split(
    X: DataFrame, model: ModelAPI
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
    X: DataFrame, min_support: float = 0.01
) -> Tuple[List[Predicate], List[float]]:
    ret = aprioriout2predicateList(
        runApriori(preprocessDataset(X), min_support=min_support)
    )
    return ret


def calculate_correctnesses(
    ifthens_withsupp: List[Tuple[Predicate, Predicate, Dict[str, float]]],
    affected_by_subgroup: Dict[str, DataFrame],
    sensitive_attribute: str,
    model: ModelAPI,
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    subgroup_names = list(affected_by_subgroup.keys())
    ifthens_with_correctness = []
    for h, s, ifsupps in tqdm(ifthens_withsupp):
        recourse_correctness = {}
        for sg in subgroup_names:
            incorrect_recourses_for_sg = incorrectRecoursesIfThen(
                h,
                s,
                affected_by_subgroup[sg].assign(**{sensitive_attribute: sg}),
                model,
            )
            covered_sg = ifsupps[sg] * affected_by_subgroup[sg].shape[0]
            inc_sg = incorrect_recourses_for_sg / covered_sg
            recourse_correctness[sg] = 1 - inc_sg

        ifthens_with_correctness.append((h, s, ifsupps, recourse_correctness))

    return ifthens_with_correctness

def calculate_correctnesses_bins(
    ifthens_withsupp: List[Tuple[Predicate, Predicate, Dict[str, float]]],
    affected_by_subgroup: Dict[str, DataFrame],
    sensitive_attribute: str,
    num_features: List[str],
    model: ModelAPI,
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    subgroup_names = list(affected_by_subgroup.keys())
    ifthens_with_correctness = []
    for h, s, ifsupps in tqdm(ifthens_withsupp):
        recourse_correctness = {}
        for sg in subgroup_names:
            incorrect_recourses_for_sg = incorrectRecoursesIfThen_bins(
                h,
                s,
                affected_by_subgroup[sg].assign(**{sensitive_attribute: sg}),
                model,
                num_features
            )
            covered_sg = ifsupps[sg] * affected_by_subgroup[sg].shape[0]
            inc_sg = incorrect_recourses_for_sg / covered_sg
            recourse_correctness[sg] = 1 - inc_sg

        ifthens_with_correctness.append((h, s, ifsupps, recourse_correctness))

    return ifthens_with_correctness

def intersect_predicate_lists(
    acc: List[Tuple[Dict[Any, Any], Dict[str, float]]],
    l2: List[Tuple[Dict[Any, Any], float]],
    l2_sg: str,
):
    ret = []
    for i, (pred1, supps) in enumerate(acc):
        for j, (pred2, supp2) in enumerate(l2):
            if pred1 == pred2:
                supps[l2_sg] = supp2
                ret.append((pred1, supps))
    return ret

def aff_intersection_version_1(RLs_and_supports, subgroups):
    RLs_supports_dict = {
        sg: [(dict(zip(p.features, p.values)), supp) for p, supp in zip(*RL_sup)]
        for sg, RL_sup in RLs_and_supports.items()
    }

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

    aff_intersection = [
        (Predicate.from_dict(d), supps) for d, supps in aff_intersection
    ]

    return aff_intersection


def aff_intersection_version_2(RLs_and_supports, subgroups):
    RLs_supports_dict = {
        sg: {tuple(sorted(zip(p.features, p.values))): supp for p, supp in zip(*RL_sup)}
        for sg, RL_sup in RLs_and_supports.items()
    }

    if len(RLs_supports_dict) < 1:
        raise ValueError("There must be at least 2 subgroups.")
    else:
        aff_intersection = []

        _, sg1 = min((len(RLs_supports_dict[sg]), sg) for sg in subgroups)

        for value, supp in tqdm(RLs_supports_dict[sg1].items()):
            in_all = True
            supp_dict = {sg1: supp}
            for sg2 in subgroups:
                if sg2 == sg1:
                    continue
                if value not in RLs_supports_dict[sg2]:
                    in_all = False
                    break
                supp_dict[sg2] = RLs_supports_dict[sg2].pop(value)

            if in_all == True:
                feat_dict = {}
                for feat in value:
                    feat_dict[feat[0]] = feat[1]
                aff_intersection.append((feat_dict, supp_dict))

    aff_intersection = [
        (Predicate.from_dict(d), supps) for d, supps in aff_intersection
    ]

    return aff_intersection

def aff_intersection(RLs_and_supports: Dict[str, Tuple[List[Predicate], List[float]]], subgroups: Sequence[Any]):
    raise NotImplementedError

def valid_ifthens_with_coverage_correctness(
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    freqitem_minsupp: float = 0.01,
    missing_subgroup_val: str = "N/A",
    drop_infeasible: bool = True,
    drop_above: bool = True,
) -> Tuple[List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]], Any]:
    # throw out all individuals for whom the value of the sensitive attribute is unknown
    X = X[X[sensitive_attribute] != missing_subgroup_val]

    # split into affected-unaffected
    X_aff, X_unaff = affected_unaffected_split(X, model)

    # find descriptors of all sensitive subgroups
    subgroups = np.unique(X[sensitive_attribute])
    # split affected individuals into subgroups
    affected_subgroups = {
        sg: X_aff[X_aff[sensitive_attribute] == sg].drop([sensitive_attribute], axis=1)
        for sg in subgroups
    }

    # calculate frequent itemsets for each subgroup and turn them into predicates
    print(
        "Computing frequent itemsets for each subgroup of the affected instances.",
        flush=True,
    )
    RLs_and_supports = {
        sg: freqitemsets_with_supports(affected_sg, min_support=freqitem_minsupp)
        for sg, affected_sg in tqdm(affected_subgroups.items())
    }
    lens = {sg: len(rls[0]) for sg, rls in RLs_and_supports.items()}
    print(f"Number of frequent itemsets for affected: {lens}", flush=True)
    rest_ret: Dict[str, Any] = {"freq-itemsets-no": lens}

    # intersection of frequent itemsets of all sensitive subgroups
    print(
        "Computing the intersection between the frequent itemsets of each subgroup of the affected instances.",
        flush=True,
    )

    aff_intersection_2 = aff_intersection_version_2(RLs_and_supports, subgroups)

    aff_intersection = aff_intersection_2
    print(f"Number of subgroups in the intersection: {len(aff_intersection)}", flush=True)
    rest_ret["inter-groups-no"] = len(aff_intersection)

    # Frequent itemsets for the unaffacted (to be used in the then clauses)
    freq_unaffected, _ = freqitemsets_with_supports(
        X_unaff, min_support=freqitem_minsupp
    )
    print(f"Number of frequent itemsets for the unaffected: {len(freq_unaffected)}", flush=True)
    rest_ret["unaff-freq-itemsets-no"] = len(freq_unaffected)

    # Filter all if-then pairs to keep only valid
    print(
        "Computing all valid if-then pairs between the common frequent itemsets of each subgroup of the affected instances and the frequent itemsets of the unaffacted instances.",
        flush=True,
    )

    # we want to create a dictionary for freq_unaffected key: features in tuple, value: list(values)
    # for each Predicate in aff_intersection we loop through the list from dictionary
    # create dictionary:

    freq_unaffected_dict = {}
    for predicate_ in freq_unaffected:
        if tuple(predicate_.features) in freq_unaffected_dict:
            freq_unaffected_dict[tuple(predicate_.features)].append(predicate_.values)
        else:
            freq_unaffected_dict[tuple(predicate_.features)] = [predicate_.values]

    ifthens_2 = []
    for predicate_, supps_dict in tqdm(aff_intersection):
        candidates = freq_unaffected_dict.get(tuple(predicate_.features))
        if candidates == None:
            continue
        for candidate_values in candidates:
            # resIsValid can be changed to avoid checking if features are the same
            if recIsValid(
                predicate_,
                Predicate(predicate_.features, candidate_values),
                affected_subgroups[subgroups[0]],
                drop_infeasible,
            ):
                ifthens_2.append(
                    (
                        predicate_,
                        Predicate(predicate_.features, candidate_values),
                        supps_dict,
                    )
                )

    ifthens = ifthens_2
    # keep ifs that have change on features of max value 2
    if drop_above == True:
        age = [val.left for val in X.age.unique()]
        age.sort()
        ifthens = [
            (ifs, then, cov)
            for ifs, then, cov in ifthens
            if drop_two_above(ifs, then, age)
        ]

    # Calculate incorrectness percentages
    print("Computing correctenesses for all valid if-thens.", flush=True)
    ifthens_with_correctness = calculate_correctnesses(
        ifthens, affected_subgroups, sensitive_attribute, model
    )

    return ifthens_with_correctness, rest_ret

def valid_ifthens_with_coverage_correctness_bins(
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    num_features: List[str],
    nbins_affected: int,
    nbins_unaffected: int,
    freqitem_minsupp: float = 0.01,
    drop_infeasible: bool = True,
    drop_above: bool = True,
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    # split into affected-unaffected
    X_aff, X_unaff = affected_unaffected_split(X, model)
    X_aff = X_aff.copy()
    X_unaff = X_unaff.copy()
    for col in num_features:
        bin_edges = np.linspace(X_unaff[col].min(), X_unaff[col].max(), nbins_unaffected)
        X_unaff[col] = pd.cut(X_unaff[col], bins=bin_edges, include_lowest=True)

        bin_edges = np.linspace(X_aff[col].min(), X_aff[col].max(), nbins_affected)
        X_aff[col] = pd.cut(X_aff[col], bins=bin_edges, include_lowest=True)

    # find descriptors of all sensitive subgroups
    subgroups = np.unique(X[sensitive_attribute])
    # split affected individuals into subgroups
    affected_subgroups = {
        sg: X_aff[X_aff[sensitive_attribute] == sg].drop([sensitive_attribute], axis=1)
        for sg in subgroups
    }

    # calculate frequent itemsets for each subgroup and turn them into predicates
    print(
        "Computing frequent itemsets for each subgroup of the affected instances.",
        flush=True,
    )
    RLs_and_supports = {
        sg: freqitemsets_with_supports(affected_sg, min_support=freqitem_minsupp)
        for sg, affected_sg in tqdm(affected_subgroups.items())
    }
    lens = {sg: len(rls[0]) for sg, rls in RLs_and_supports.items()}
    print(f"Number of frequent itemsets for affected: {lens}", flush=True)

    # intersection of frequent itemsets of all sensitive subgroups
    print(
        "Computing the intersection between the frequent itemsets of each subgroup of the affected instances.",
        flush=True,
    )

    aff_intersection_2 = aff_intersection_version_2(RLs_and_supports, subgroups)

    aff_intersection = aff_intersection_2
    print(f"Number of groups from the intersection: {len(aff_intersection)}", flush=True)

    # Frequent itemsets for the unaffacted (to be used in the then clauses)
    freq_unaffected, _ = freqitemsets_with_supports(
        X_unaff, min_support=freqitem_minsupp
    )
    print(f"Number of frequent itemsets for the unaffected: {len(freq_unaffected)}", flush=True)

    # Filter all if-then pairs to keep only valid
    print(
        "Computing all valid if-then pairs between the common frequent itemsets of each subgroup of the affected instances and the frequent itemsets of the unaffacted instances.",
        flush=True,
    )

    # we want to create a dictionary for freq_unaffected key: features in tuple, value: list(values)
    # for each Predicate in aff_intersection we loop through the list from dictionary
    # create dictionary:

    freq_unaffected_dict = {}
    for predicate_ in freq_unaffected:
        if tuple(predicate_.features) in freq_unaffected_dict:
            freq_unaffected_dict[tuple(predicate_.features)].append(predicate_.values)
        else:
            freq_unaffected_dict[tuple(predicate_.features)] = [predicate_.values]

    ifthens_2 = []
    for predicate_, supps_dict in tqdm(aff_intersection):
        candidates = freq_unaffected_dict.get(tuple(predicate_.features))
        if candidates == None:
            continue
        for candidate_values in candidates:
            # resIsValid can be changed to avoid checking if features are the same
            if recIsValid(
                predicate_,
                Predicate(predicate_.features, candidate_values),
                affected_subgroups[subgroups[0]],
                drop_infeasible,
            ):
                ifthens_2.append(
                    (
                        predicate_,
                        Predicate(predicate_.features, candidate_values),
                        supps_dict,
                    )
                )

    ifthens = ifthens_2
    # keep ifs that have change on features of max value 2
    if drop_above == True:
        age = [val.left for val in X.age.unique()]
        age.sort()
        ifthens = [
            (ifs, then, cov)
            for ifs, then, cov in ifthens
            if drop_two_above(ifs, then, age)
        ]

    # Calculate incorrectness percentages
    print("Computing correctenesses for all valid if-thens.", flush=True)
    X_aff, _ = affected_unaffected_split(X, model)
    affected_subgroups = {
        sg: X_aff[X_aff[sensitive_attribute] == sg].drop([sensitive_attribute], axis=1)
        for sg in subgroups
    }
    ifthens_with_correctness = calculate_correctnesses_bins(
        ifthens, affected_subgroups, sensitive_attribute, num_features, model
    )

    return ifthens_with_correctness

def rules2rulesbyif(
    rules: List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    # group rules based on If clauses, instead of protected subgroups!
    rules_by_if: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]
    ] = {}
    for h, s, covs, cors in rules:
        if h not in rules_by_if:
            rules_by_if[h] = {sg: (cov, []) for sg, cov in covs.items()}

        for sg, (_cov, sg_thens) in rules_by_if[h].items():
            sg_thens.append((s, cors[sg]))

    return rules_by_if

def calc_costs(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    params: ParameterProxy = ParameterProxy()
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = dict()
    for ifclause, thenclauses in rules.items():
        newthenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] = dict()
        for sg, (cov, thens) in thenclauses.items():
            # TODO: make featureChangePred return a float, if possible
            newthens = [(then, cor, float(featureChangePred(ifclause, then, params))) for then, cor in thens]
            newthenclauses[sg] = (cov, newthens)
        ret[ifclause] = newthenclauses
    return ret

def cost_update_inplace(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    params: ParameterProxy = ParameterProxy()
) -> None:
    for ifclause, thenclauses in rules.items():
        for sg, (cov, thens) in thenclauses.items():
            for i, (then, cor, _cost) in enumerate(thens):
                thens[i] = (then, cor, featureChangePred(ifclause, then, params))
    return

def rulesbyif2rules(
    rules_by_if: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    rules: List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]] = []
    for ifclause, thenclauses in rules_by_if.items():
        then_covs_cors = dict()
        for sg, (cov, thens) in thenclauses.items():
            for then, cor in thens:
                if then in then_covs_cors:
                    then_covs_cors[then][0][sg] = cov
                    then_covs_cors[then][1][sg] = cor
                else:
                    then_covs_cors[then] = ({sg: cov}, {sg: cor})

        for then, covs_cors in then_covs_cors.items():
            rules.append((ifclause, then, covs_cors[0], covs_cors[1]))
    return rules

def select_rules_subset(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    metric: str = "total-correctness",
    sort_strategy: str = "max-cost-diff-decr",
    top_count: int = 10,
    filter_sequence: List[str] = [],
    cor_threshold: float = 0.5,
    cost_threshold: float = 0.5,
    secondary_sorting_objectives: List[str] = []
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    Dict[Predicate, Dict[str, float]],
]:
    # step 1: sort according to metric
    metrics: Dict[
        str, Callable[[Predicate, List[Tuple[Predicate, float, float]]], float]
    ] = {
        "min-above-corr": functools.partial(
            if_group_cost_min_change_correctness_threshold, cor_thres=cor_threshold
        ),
        "num-above-thr": functools.partial(if_group_cost_recoursescount_correctness_threshold, cor_thres=cor_threshold),
        "total-correctness": if_group_maximum_correctness,
        "max-upto-cost": functools.partial(
            if_group_cost_max_correctness_cost_budget, cost_thres=cost_threshold
        ),
        "fairness-of-mean-recourse-conditional": if_group_average_recourse_cost_conditional
    }
    sorting_functions = {
        "max-cost-diff-decr": functools.partial(
            sort_triples_by_max_costdiff,
            ignore_nans=False,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "max-cost-diff-decr-ignore-forall-subgroups-empty": functools.partial(
            sort_triples_by_max_costdiff,
            ignore_nans=True,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "max-cost-diff-decr-ignore-exists-subgroup-empty": functools.partial(
            sort_triples_by_max_costdiff,
            ignore_nans=True,
            ignore_infs=True,
            secondary_objectives=secondary_sorting_objectives,
        ),
    }
    metric_fn = metrics[metric]
    sort_fn = sorting_functions[sort_strategy]
    rules_sorted = sort_fn(rulesbyif, group_calculator=metric_fn)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    # keep also the aggregate costs of the then-blocks of the top rules
    costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        group_calculator=metric_fn
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
            filter_contained_rules_keep_max_bias, subgroup_costs=costs
        ),
        "remove-below-thr-corr": functools.partial(
            remove_rules_below_correctness_threshold, threshold=cor_threshold
        ),
        "remove-above-thr-cost": functools.partial(
            remove_rules_above_cost_budget, threshold=cost_threshold
        ),
        "keep-cheap-rules-above-thr-cor": functools.partial(
            keep_cheapest_rules_above_cumulative_correctness_threshold, threshold=cor_threshold
        ),
        "remove-fair-rules": functools.partial(delete_fair_rules, subgroup_costs=costs),
        "keep-only-min-change": keep_only_minimum_change
    }
    for single_filter in filter_sequence:
        top_rules = filters[single_filter](top_rules)

    return top_rules, costs

def select_rules_subset_KStest(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    affected_population_sizes: Dict[str, int],
    top_count: int = 10,
    filter_contained: bool = False
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    Dict[Predicate, float]
]:
    # step 1: sort according to metric
    rules_sorted, unfairness = sort_triples_KStest(rulesbyif, affected_population_sizes)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    if filter_contained:
        top_rules = filter_contained_rules_simple(top_rules)

    return top_rules, unfairness

def cum_corr_costs(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    X: DataFrame,
    model: ModelAPI,
    params: ParameterProxy = ParameterProxy(),
) -> List[Tuple[Predicate, float, float]]:
    withcosts = [
        (thenclause, cor, featureChangePred(ifclause, thenclause, params))
        for thenclause, cor in thenclauses
    ]
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
        X_covered = X_covered[~preds.astype(bool)]  # type: ignore

    cumcorrs = np.array(cumcorrs).cumsum() / covered_count
    updated_thens = [
        (thenclause, cumcor, float(cost))
        for (thenclause, _cor, cost), cumcor in zip(thens_sorted_by_cost, cumcorrs)
    ]

    return updated_thens


def cum_corr_costs_all(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    params: ParameterProxy = ParameterProxy(),
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    X_affected: DataFrame = X[model.predict(X) == 0]  # type: ignore
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = {}
    for ifclause, all_thens in tqdm(rulesbyif.items()):
        all_thens_new: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] = {}
        for sg, (cov, thens) in all_thens.items():
            subgroup_affected = X_affected[X_affected[sensitive_attribute] == sg]
            all_thens_new[sg] = (cov, cum_corr_costs(
                ifclause, thens, subgroup_affected, model, params=params
            ))
        ret[ifclause] = all_thens_new
    return ret

def cum_corr_costs_all_minimal(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    params: ParameterProxy = ParameterProxy(),
) -> Dict[Predicate, Dict[str, List[Tuple[float, float]]]]:
    full_rules = cum_corr_costs_all(rulesbyif, X, model, sensitive_attribute, params)
    ret: Dict[Predicate, Dict[str, List[Tuple[float, float]]]] = {}
    for ifclause, all_thens in full_rules.items():
        ret[ifclause] = {}
        for sg, (cov, thens) in all_thens.items():
            thens_plain = [(corr, cost) for _then, corr, cost in thens]
            ret[ifclause][sg] = thens_plain
    return ret


