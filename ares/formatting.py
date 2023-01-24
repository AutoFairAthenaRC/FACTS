from typing import List, Dict, Tuple

from pandas import DataFrame

from models import ModelAPI
from recourse_sets import TwoLevelRecourseSet
from metrics import incorrectRecourses, incorrectRecoursesSubmodular, cover, featureChange, featureCost, incorrectRecoursesSingle
from predicate import Predicate

def report_base(outer: List[Predicate], blocks: List) -> str:
    ret = []
    for p in outer:
        ret.append(f"If {p}:\n")
        for b in blocks:
            ret.append(f"\t{b}")
    return "".join(ret)

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

def recourse_report_preprocessed(groups: List[str], rules: Dict[str, List[Tuple[Predicate, Predicate, float, float]]]) -> str:
    ret = []
    for val in groups:
        ret.append(f"For subgroup '{val}':\n")
        rules_subgroup = rules[val]
        for h, s, coverage, incorrectness in rules_subgroup:
            ret.append(f"\tIf {h},\n\tThen {s}.\n")

            ret.append(f"\t\tCoverage: {coverage:.3%} of those in the subgroup that are affected.\n")
            ret.append(f"\t\tIncorrect recourses: {incorrectness:.3%} over all individuals covered by this rule.\n")
    return "".join(ret)

def recourse_report_reverse(rules: List[Tuple[Predicate, Dict[str, List[Tuple[Predicate, float, float]]]]]) -> str:
    ret = []
    for ifclause, all_thens in rules:
        ret.append(f"If {ifclause}:\n")
        for subgroup, thens in all_thens.items():
            ret.append(f"\tSubgroup '{subgroup}'\n")
            for then, coverage, correctness in thens:
                ret.append(f"\t\tMake {then} with coverage {coverage} and correctness {correctness}.")

    return "".join(ret)
