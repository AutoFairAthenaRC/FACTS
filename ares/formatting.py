from typing import List, Dict, Tuple, Any, Optional

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

def to_bold_str(s: Any) -> str:
    return f"\033[1m{s}\033[0m"

def to_blue_str(s: Any) -> str:
    return f"\033[0;34m{s}\033[0m"

def to_green_str(s: Any) -> str:
    return f"\033[0;32m{s}\033[0m"

def recourse_report_reverse(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A"
) -> str:
    ret = []
    for ifclause, sg_thens in rules.items():
        ret.append(f"If {to_bold_str(ifclause)}:\n")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            cov_str = to_blue_str(f"{cov:.4%}")
            ret.append(f"\tSubgroup '{to_bold_str(subgroup)}', {cov_str} covered")
            if population_sizes is not None:
                if subgroup in population_sizes:
                    ret.append(f" out of {population_sizes[subgroup]}")
                else:
                    ret.append(" (protected subgroup population size not given)")
            ret.append("\n")

            # print each available recourse together with the respective correctness
            for then, correctness in thens:
                cor_str = to_green_str(f"{correctness:.4%}")
                ret.append(f"\t\tMake {to_bold_str(then)} with correctness {cor_str}.\n")

    return "".join(ret)
