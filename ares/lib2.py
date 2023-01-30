from typing import List, Tuple, Dict

import numpy as np
from pandas import DataFrame

from parameters import *
from models import ModelAPI
from frequent_itemsets import runApriori, preprocessDataset, aprioriout2predicateList
from recourse_sets import TwoLevelRecourseSet
from metrics import incorrectRecoursesSingle

## Re-exporting
from optimization import optimize_vanilla
from predicate import Predicate, recIsValid
## Re-exporting


def split_dataset(X: DataFrame, attr: str):
    vals = X[attr].unique()
    grouping = X.groupby(attr)
    return {val: grouping.get_group(val) for val in vals}

def global_counterfactuals_ares(X: DataFrame, model: ModelAPI, sensitive_attribute: str, subsample_size=400):
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
    ifthens_with_correctness = valid_triples_with_coverage_correctness(X, model, sensitive_attribute)

    # all we need now is which are the subgroups (e.g. Male-Female)
    subgroups = np.unique(X[sensitive_attribute])

    # finally, keep triples whose coverage and correct recourse percentage is at least a given threshold
    ifthens_filtered = {sg: [] for sg in subgroups}
    for h, s, ifsupps, thencorrs in ifthens_with_correctness:
        for sg in subgroups:
            if ifsupps[sg] >= threshold_coverage and thencorrs[sg] >= threshold_correctness:
                ifthens_filtered[sg].append((h, s, ifsupps[sg], thencorrs[sg]))
    
    return ifthens_filtered

def valid_triples_with_coverage_correctness(
    X: DataFrame, model: ModelAPI,
    sensitive_attribute: str
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    # get model predictions
    preds = model.predict(X)
    # find affected individuals
    X_aff_idxs = np.where(preds == 0)[0]
    X_aff = X.iloc[X_aff_idxs, :]

    # find unaffected individuals
    X_unaff_idxs = np.where(preds == 1)[0]
    X_unaff = X.iloc[X_unaff_idxs, :]

    # find descriptors of all sensitive subgroups
    subgroups = np.unique(X[sensitive_attribute])

    # split affected individuals into subgroups
    affected_subgroups = {sg: X_aff[X_aff[sensitive_attribute] == sg].drop([sensitive_attribute], axis=1) for sg in subgroups}

    # calculate frequent itemsets for each subgroup and turn them into predicates
    freq_itemsets = {sg: runApriori(preprocessDataset(affected_sg), min_support=0.03) for sg, affected_sg in affected_subgroups.items()}
    RLs_and_supports = {sg: aprioriout2predicateList(freq) for sg, freq in freq_itemsets.items()}

    # turn RLs into dictionaries for easier comparison
    RLs_supports_dict = {sg: [(dict(zip(p.features, p.values)), supp) for p, supp in zip(*RL_sup)] for sg, RL_sup in RLs_and_supports.items()}

    # intersection of frequent itemsets of all sensitive subhgroups
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
    freq_unaffected, _ = aprioriout2predicateList(runApriori(preprocessDataset(X_unaff), min_support=0.03))

    # Filter all if-then pairs to keep only valid
    ifthens = [(h, s, ifsupps) for h, ifsupps in aff_intersection for s in freq_unaffected if recIsValid(h, s)]

    # Calculate incorrectness percentages
    from tqdm import tqdm
    ifthens_with_correctness = []
    for h, s, ifsupps in tqdm(ifthens):
        recourse_correctness = {}
        for sg in subgroups:
            sd = Predicate.from_dict({sensitive_attribute: sg})
            incorrect_recourses_for_sg = incorrectRecoursesSingle(sd, h, s, X_aff, model)
            covered_sg = ifsupps[sg] * affected_subgroups[sg].shape[0]
            inc_sg = incorrect_recourses_for_sg / covered_sg
            recourse_correctness[sg] = 1 - inc_sg

        ifthens_with_correctness.append((h, s, ifsupps, recourse_correctness))
    
    return ifthens_with_correctness

def rules2rulesbyif(rules: List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]
) -> Dict[Predicate, Dict[str, List[Tuple[Predicate, float, float]]]]:
    subgroups = list(rules[0][2].keys())

    # group rules based on If clauses, instead of protected subgroups!
    rules_by_if = {}
    for h, s, covs, cors in rules:
        block = rules_by_if.get(h, {sg: [] for sg in subgroups})
        for sg, thens in block.items():
            thens.append((s, covs[sg], cors[sg]))
        rules_by_if[h] = block
    
    return rules_by_if


