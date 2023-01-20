import numpy as np
from pandas import DataFrame

from parameters import *
from models import ModelAPI
from apriori import runApriori, preprocessDataset, aprioriout2predicateList
from recourse_sets import TwoLevelRecourseSet
from metrics import cover, incorrectRecoursesSingle, incorrectRecoursesSubmodular, incorrectRecourses, featureCost, featureChange

## Re-exporting
from optimization import optimize_vanilla
from predicate import Predicate
## Re-exporting


def recourse_report(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI) -> str:
    ret = []
    # first, print the statistics of the whole recourse set
    incorrects_additive = incorrectRecourses(R, X_aff, model)
    incorrects_at_least_one = incorrectRecoursesSubmodular(R, X_aff, model)
    coverage = cover(R, X_aff)
    feature_cost = featureCost(R)
    feature_change = featureChange(R)

    ret.append(f"Total coverage: {coverage / X_aff.shape[0]:.3%} (over all affected).\n")
    ret.append(f"Total incorrect recourses: {incorrects_additive / coverage:.3%} (over all those covered).\n")
    if incorrects_at_least_one != incorrects_additive:
        ret.append(f"\tAttention! If measured as at-least-one-correct, it changes to {incorrects_at_least_one / coverage:.3%}!\n")
    ret.append(f"Total feature cost: {feature_cost}.\n")
    ret.append(f"Total feature change: {feature_change}.\n")

    # then, print the rules with the statistics for each rule separately
    sensitive = R.feature
    for val in R.values:
        ret.append(f"If {sensitive} = {val}:\n")
        rules = R.rules[val]
        for h, s in zip(rules.hypotheses, rules.suggestions):
            ret.append(f"\tIf {h},\n\tThen {s}.\n")

            sd = Predicate.from_dict({sensitive: val})
            degenerate_two_level_set = TwoLevelRecourseSet.from_triples([(sd, h, s)])

            coverage = cover(degenerate_two_level_set, X_aff)
            inc_original = incorrectRecoursesSingle(sd, h, s, X_aff, model) / coverage
            inc_submodular = incorrectRecoursesSubmodular(degenerate_two_level_set, X_aff, model) / coverage
            coverage /= X_aff.shape[0]

            ret.append(f"\t\tCoverage: {coverage:.3%} over all affected.\n")
            ret.append(f"\t\tIncorrect recourses additive: {inc_original:.3%} over all individuals covered by this rule.\n")
            ret.append(f"\t\tIncorrect recourses at-least-one: {inc_submodular:.3%} over all individuals covered by this rule.\n")
    return "".join(ret)

def split_dataset(X: DataFrame, attr: str):
    vals = X[attr].unique()
    grouping = X.groupby(attr)
    return {val: grouping.get_group(val) for val in vals}

def global_counterfactuals(X: DataFrame, model: ModelAPI, sensitive_attribute: str, subsample_size=400):
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

