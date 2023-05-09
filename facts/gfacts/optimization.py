import functools

from typing import List, Set, Tuple, Dict
from collections import defaultdict

import numpy as np
from pandas import DataFrame

from .predicate import Predicate, featureChangePred, featureCostPred, recIsValid
from .models import ModelAPI
from .metrics import (
    incorrectRecoursesSingle,
    calculate_cost_difference_2groups,
    max_intergroup_cost_diff,
    calculate_all_if_subgroup_costs,
    calculate_all_if_subgroup_costs_cumulative
)
from .parameters import ParameterProxy

##### Submodular optimization as described in AReS paper.

def ground_set_generation_vanilla(SD: List[Predicate], RL: List[Predicate], X_aff: DataFrame, model: ModelAPI):
    valid_triples = [(sd, h, s) for sd in SD for h in RL for s in RL if recIsValid(h, s, X_aff, False)]
    return valid_triples

def optimizer(modulars: List[int], covers: List[Set[int]], N_aff: int, params: ParameterProxy = ParameterProxy()):
    assert len(modulars) == len(covers)
    lcov = params.lambda_cover

    modulars_covers_zipped = zip(modulars, covers)
    sorted_zipped = sorted(modulars_covers_zipped, key=lambda pair: pair[0], reverse=True)

    modulars = [e[0] for e in sorted_zipped]
    covers = [e[1] for e in sorted_zipped]

    N = len(modulars)
    singleton_rewards = [mod + lcov * len(cov) for mod, cov in zip(modulars, covers)]

    argmax_singleton = np.argmax(singleton_rewards)
    subset = set([argmax_singleton])
    excluded = set(np.arange(N))
    excluded.remove(argmax_singleton)

    curr_modular = modulars[argmax_singleton]
    curr_cover = lcov * len(covers[argmax_singleton])
    ref_counts = [0] * N_aff
    for idx in covers[argmax_singleton]:
        ref_counts[idx] += 1
    
    flag_continue = True
    while flag_continue:
        flag_continue = False

        # try delete
        for idx in subset:
            updated_modular = curr_modular - modulars[idx]
            updated_cover = curr_cover
            for obj in covers[idx]:
                ref_counts[obj] -= 1
                if ref_counts[obj] < 0:
                    raise IndexError("Something went wrong. Reference count negative.")
                elif ref_counts[obj] == 0:
                    updated_cover -= lcov
            
            if updated_modular + updated_cover > curr_modular + curr_cover:
                curr_modular = updated_modular
                curr_cover = updated_cover
                subset.remove(idx)
                excluded.add(idx)
                flag_continue = True
                break
            else:
                for j in covers[idx]:
                    ref_counts[j] += 1
        
        # try add
        for idx in excluded:
            updated_modular = curr_modular + modulars[idx]
            updated_cover = curr_cover
            for obj in covers[idx]:
                ref_counts[obj] += 1
                if ref_counts[obj] == 1:
                    updated_cover += lcov
            
            if updated_modular + updated_cover > curr_modular + curr_cover:
                curr_modular = updated_modular
                curr_cover = updated_cover
                subset.add(idx)
                excluded.remove(idx)
                flag_continue = True
                break
            else:
                for j in covers[idx]:
                    ref_counts[j] -= 1
                    if ref_counts[j] < 0:
                        raise IndexError("Something went wrong. Reference count negative.")

        # try exchange
        for idx1 in subset:
            for idx2 in excluded:
                updated_modular = curr_modular - modulars[idx1] + modulars[idx2]
                updated_cover = curr_cover
                for obj in covers[idx1]:
                    ref_counts[obj] -= 1
                    if ref_counts[obj] < 0:
                        raise IndexError("Something went wrong. Reference count negative.")
                    elif ref_counts[obj] == 0:
                        updated_cover -= lcov
                for obj in covers[idx2]:
                    ref_counts[obj] += 1
                    if ref_counts[obj] == 1:
                        updated_cover += lcov
                
                if updated_modular + updated_cover > curr_modular + curr_cover:
                    curr_modular = updated_modular
                    curr_cover = updated_cover
                    subset.remove(idx1)
                    excluded.add(idx1)
                    subset.add(idx2)
                    excluded.remove(idx2)
                    flag_continue = True
                    break
                else:
                    for j in covers[idx2]:
                        ref_counts[j] -= 1
                        if ref_counts[j] < 0:
                            raise IndexError("Something went wrong. Reference count negative.")
                    for j in covers[idx1]:
                        ref_counts[j] += 1

    return subset

def optimize_vanilla(SD: List[Predicate], RL: List[Predicate], X_aff: DataFrame, model: ModelAPI) -> Tuple[List[Tuple[Predicate, Predicate, Predicate]], int, int, int, int]:
    d = defaultdict(lambda: RL, {})
    return optimize(SD, d, d, X_aff, model)

def optimize(
    SD: List[Predicate],
    ifs: Dict[str, List[Predicate]],
    thens: Dict[str, List[Predicate]],
    X_aff: DataFrame,
    model: ModelAPI,
    params: ParameterProxy = ParameterProxy()
) -> Tuple[List[Tuple[Predicate, Predicate, Predicate]], int, int, int, int]:
    all_triples = [(sd, h, s) for sd in SD for h in ifs[sd.values[0]] for s in thens[sd.values[0]] if recIsValid(h, s, X_aff, False)]
    triples_no = len(all_triples)
    print(f"Total triples = {triples_no}")
    all_incorrects = list(-params.lambda_correctness * incorrectRecoursesSingle(sd, h, s, X_aff, model) for sd, h, s in all_triples)
    print("Calculated incorrect recourse for each triple")
    all_feature_costs = list(-params.lambda_featureCost * featureCostPred(h, s) for _, h, s in all_triples)
    print("Calculated feature costs for each triple")
    all_feature_changes = list(-params.lambda_featureChange * featureChangePred(h, s) for _, h, s in all_triples)
    print("Calculated feature changes for each feature")
    
    triples_covers: List = [set() for i in range(triples_no)]
    for i, (sd, h, s) in enumerate(all_triples):
        X_aff_covered = X_aff.apply(lambda x: h.satisfies(x) and sd.satisfies(x), axis=1).to_numpy()
        nonzeros = X_aff_covered.nonzero()
        nonzeros_first = nonzeros[0]
        triples_covers[i] = set(nonzeros_first)
    print("Calculated covers for each triple")
    
    almost_all = [inc + cost + change for inc, cost, change in zip(all_incorrects, all_feature_costs, all_feature_changes)]
    best_subset = optimizer(almost_all, triples_covers, X_aff.shape[0], params=params)
    
    final_incorrects = sum([all_incorrects[i] for i in best_subset])
    final_coverage = len(set().union(*[triples_covers[i] for i in best_subset]))
    final_feature_cost = sum([all_feature_costs[i] for i in best_subset])
    final_feature_change = sum([all_feature_changes[i] for i in best_subset])
    return [all_triples[i] for i in best_subset], final_incorrects, final_coverage, final_feature_cost, final_feature_change


##### Rankings of if-groups

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
    use_secondary_objective: bool = False,
    **kwargs
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    subgroup_costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        **kwargs
    )

    max_intergroup_cost_diffs = {
        ifclause: max(subgroup_costs[ifclause].values()) - min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    min_group_costs = {
        ifclause: min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }

    def simple_objective_fn(ifthens):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if np.isnan(max_costdiff):
            max_costdiff = -np.inf
        return max_costdiff
    
    def double_objective_fn(ifthens):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if np.isnan(max_costdiff):
            max_costdiff = -np.inf
        return (max_costdiff, -min_group_costs[ifclause])
    
    if use_secondary_objective:
        ret = sorted(rulesbyif.items(), key=double_objective_fn, reverse=True)
    else:
        ret = sorted(rulesbyif.items(), key=simple_objective_fn, reverse=True)
    return ret

def sort_triples_by_max_costdiff_ignore_nans_infs(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    use_secondary_objective: bool = False,
    **kwargs
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    subgroup_costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        **kwargs
    )

    max_intergroup_cost_diffs = {
        ifclause: max(subgroup_costs[ifclause].values()) - min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    min_group_costs = {
        ifclause: min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }

    def simple_objective_fn(ifthens):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if np.isnan(max_costdiff) or np.isinf(max_costdiff):
            max_costdiff = -np.inf
        return max_costdiff
    
    def double_objective_fn(ifthens):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if np.isnan(max_costdiff) or np.isinf(max_costdiff):
            max_costdiff = -np.inf
        return (max_costdiff, -min_group_costs[ifclause])
    
    if use_secondary_objective:
        ret = sorted(rulesbyif.items(), key=double_objective_fn, reverse=True)
    else:
        ret = sorted(rulesbyif.items(), key=simple_objective_fn, reverse=True)
    return ret



def sort_triples_by_max_costdiff_generic(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    ignore_nans: bool = False,
    ignore_infs: bool = False,
    secondary_objectives: List[str] = [],
    **kwargs
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    subgroup_costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        **kwargs
    )

    max_intergroup_cost_diffs = {
        ifclause: max(subgroup_costs[ifclause].values()) - min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    min_group_costs = {
        ifclause: min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    max_group_correctness = {
        ifclause: max(cor for _sg, (_cov, thens) in thenclauses.items() for _then, cor in thens)
        for ifclause, thenclauses in rulesbyif.items()
    }

    def objective_fn(ifthens, ignore_nan, ignore_inf, return_indicator):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if ignore_nan and np.isnan(max_costdiff):
            max_costdiff = -np.inf
        if ignore_inf and np.isinf(max_costdiff):
            max_costdiff = -np.inf
        
        optional_rets = {
            "min-group-cost": -min_group_costs[ifclause],
            "max-group-corr": max_group_correctness[ifclause]
        }
        ret = (max_costdiff,)
        for i in return_indicator:
            ret = ret + (optional_rets[i],)
        return ret
    
    return sorted(rulesbyif.items(), key=functools.partial(objective_fn, ignore_nan=ignore_nans, ignore_inf=ignore_infs, return_indicator=secondary_objectives), reverse=True)

def sort_triples_by_max_costdiff_generic_cumulative(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    ignore_nans: bool = False,
    ignore_infs: bool = False,
    secondary_objectives: List[str] = [],
    **kwargs
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]]:
    subgroup_costs = calculate_all_if_subgroup_costs_cumulative(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        **kwargs
    )

    max_intergroup_cost_diffs = {
        ifclause: max(subgroup_costs[ifclause].values()) - min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    min_group_costs = {
        ifclause: min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    max_group_correctness = {
        ifclause: max(cor for _sg, (_cov, thens) in thenclauses.items() for _then, cor, _cost in thens)
        for ifclause, thenclauses in rulesbyif.items()
    }

    def objective_fn(ifthens, ignore_nan, ignore_inf, return_indicator):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if ignore_nan and np.isnan(max_costdiff):
            max_costdiff = -np.inf
        if ignore_inf and np.isinf(max_costdiff):
            max_costdiff = -np.inf
        
        optional_rets = {
            "min-group-cost": -min_group_costs[ifclause],
            "max-group-corr": max_group_correctness[ifclause]
        }
        ret = (max_costdiff,)
        for i in return_indicator:
            ret = ret + (optional_rets[i],)
        return ret
    
    return sorted(rulesbyif.items(), key=functools.partial(objective_fn, ignore_nan=ignore_nans, ignore_inf=ignore_infs, return_indicator=secondary_objectives), reverse=True)


def sort_triples_KStest(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    affected_population_sizes: Dict[str, int],
    alpha: float = 0.95
) -> Tuple[
    List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]],
    Dict[Predicate, float]
]:
    def calculate_test(ifclause: Predicate, thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]):
        if len(thenclauses) != 2:
            raise NotImplementedError("Definitions only for two protected subgroups")
        
        sgs = list(thenclauses.keys())
        sg1 = sgs[0]
        sg2 = sgs[1]
        corrs1 = np.array([corr for then, corr, cost in thenclauses[sg1][1]])
        corrs2 = np.array([corr for then, corr, cost in thenclauses[sg2][1]])
        lhs: float = abs(corrs1 - corrs2).max()

        cov1 = thenclauses[sg1][0]
        cov2 = thenclauses[sg2][0]
        affected_sg1 = cov1 * affected_population_sizes[sg1]
        affected_sg2 = cov2 * affected_population_sizes[sg2]
        rhs: float = np.sqrt(- np.log(alpha / 2) * (1/2) * ((1 / affected_sg1) + (1 / affected_sg2)))

        return lhs, rhs

    unfair_rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = {}
    unfairness: Dict[Predicate, float] = {}
    for ifclause, thenclauses in rulesbyif.items():
        lhs, rhs = calculate_test(ifclause, thenclauses)
        if lhs < rhs:
            continue
        unfair_rules[ifclause] = thenclauses
        unfairness[ifclause] = lhs - rhs
    
    return sorted(unfair_rules.items(), key=lambda ifthens: unfairness[ifthens[0]], reverse=True), unfairness