from typing import Dict, Tuple, List
from pandas import Series
import numpy as np

from .parameters import ParameterProxy
from .predicate import Predicate, featureChangePred

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

def filter_contained_rules_simple(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]
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
            if_and_allthens_relevant_values = [tuple(then.to_dict()[feat] for feat in extra_features) for then in allthens]
            if_and_allthens_relevant_values.append(tuple(ifclause.to_dict()[feat] for feat in extra_features))
            if Series(if_and_allthens_relevant_values).unique().size == 1:
                flag_keep = False
                break
            
        if flag_keep:
            ret[ifclause] = thenclauses
    
    return ret

# TODO: implementation is slightly incorrect. Should create partition of ifs where each if has a "subsumes" relationship with at least another.
# essentially, if "subsumes" is a graph, it is transient, and we want weakly connected components
def filter_contained_rules_keep_max_bias(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    subgroup_costs: Dict[Predicate, Dict[str, float]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    ret = dict()
    cost_values = {subgroup: costs.values() for subgroup, costs in subgroup_costs.items()}
    bias_measures = {subgroup: max(costs) - min(costs) for subgroup, costs in cost_values.items()}
    flags_keep = {subgroup: True for subgroup, _thens in rulesbyif.items()}
    for ifclause, thenclauses in rulesbyif.items():
        allthens = [then for _sg, (_cov, sg_thens) in thenclauses.items() for then, _cor in sg_thens]
        for otherif, _ in rulesbyif.items():
            if not ifclause.contains(otherif):
                continue
            extra_features = list(set(ifclause.features) - set(otherif.features))
            if len(extra_features) == 0:
                continue
            if_and_allthens_relevant_values = [tuple(then.to_dict()[feat] for feat in extra_features) for then in allthens]
            if_and_allthens_relevant_values.append(tuple(ifclause.to_dict()[feat] for feat in extra_features))
            if Series(if_and_allthens_relevant_values).unique().size == 1:
                if bias_measures[ifclause] > bias_measures[otherif]:
                    flags_keep[otherif] = False
                else:
                    flags_keep[ifclause] = False
    
    for ifclause, thenclauses in rulesbyif.items():
        if flags_keep[ifclause]:
            ret[ifclause] = thenclauses
    return ret

def delete_fair_rules(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    subgroup_costs: Dict[Predicate, Dict[str, float]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        curr_subgroup_costs = list(subgroup_costs[ifclause].values())
        max_intergroup_cost_diff = max(curr_subgroup_costs) - min(curr_subgroup_costs)
        if max_intergroup_cost_diff == 0 or np.isnan(max_intergroup_cost_diff):
            continue
        ret[ifclause] = thenclauses
    return ret

def keep_only_minimum_change(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    params: ParameterProxy = ParameterProxy()
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        ret[ifclause] = dict()
        for sg, (cov, thens) in thenclauses.items():
            min_change = min((featureChangePred(ifclause, then, params=params) for then, _ in thens), default=np.inf)
            newthens = [(then, cor) for then, cor in thens if featureChangePred(ifclause, then, params=params) <= min_change]
            ret[ifclause][sg] = (cov, newthens)
    return ret




# same for metrics definitions of the "micro" viewpoint
def filter_by_correctness_cumulative(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    threshold: float = 0.5
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    # ret = {ifclause: {sg: (cov, [(then, cor) for then, cor in thens if cor >= threshold]) for sg, (cov, thens) in thenclauses.items()} for ifclause, thenclauses in rulesbyif.items()}
    ret = dict()
    for ifclause, thenclauses in rulesbyif.items():
        filtered_thenclauses = dict()
        for sg, (cov, sg_thens) in thenclauses.items():
            filtered_thens = [(then, cor, cost) for then, cor, cost in sg_thens if cor >= threshold]
            filtered_thenclauses[sg] = (cov, filtered_thens)
        ret[ifclause] = filtered_thenclauses
    return ret

def keep_cheapest_rules_above_cumulative_correctness_threshold(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    threshold: float = 0.5
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]= dict()
    for ifclause, thenclauses in rulesbyif.items():
        filtered_thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] = dict()
        for sg, (cov, sg_thens) in thenclauses.items():
            filtered_thens: List[Tuple[Predicate, float, float]] = []
            for then, cor, cost in sg_thens:
                filtered_thens.append((then, cor, cost))
                if cor > threshold:
                    break
            filtered_thenclauses[sg] = (cov, filtered_thens)
        ret[ifclause] = filtered_thenclauses
    return ret

def filter_by_cost_cumulative(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    threshold: float = 0.5
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    # ret = {ifclause: {sg: (cov, [(then, cor) for then, cor in thens if cor >= threshold]) for sg, (cov, thens) in thenclauses.items()} for ifclause, thenclauses in rulesbyif.items()}
    ret = dict()
    for ifclause, thenclauses in rulesbyif.items():
        filtered_thenclauses = dict()
        for sg, (cov, sg_thens) in thenclauses.items():
            filtered_thens = [(then, cor, cost) for then, cor, cost in sg_thens if cost <= threshold]
            filtered_thenclauses[sg] = (cov, filtered_thens)
        ret[ifclause] = filtered_thenclauses
    return ret

def filter_contained_rules_simple_cumulative(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret = dict()
    for ifclause, thenclauses in rulesbyif.items():
        flag_keep = True
        allthens = [then for _sg, (_cov, sg_thens) in thenclauses.items() for then, _cor, _cost in sg_thens]
        for otherif, _ in rulesbyif.items():
            if not ifclause.contains(otherif):
                continue
            extra_features = list(set(ifclause.features) - set(otherif.features))
            if len(extra_features) == 0:
                continue
            if_and_allthens_relevant_values = [tuple(then.to_dict()[feat] for feat in extra_features) for then in allthens]
            if_and_allthens_relevant_values.append(tuple(ifclause.to_dict()[feat] for feat in extra_features))
            if Series(if_and_allthens_relevant_values).unique().size == 1:
                flag_keep = False
                break
        
        if flag_keep:
            ret[ifclause] = thenclauses
    
    return ret

# TODO: implementation is slightly incorrect. Should create partition of ifs where each if has a "subsumes" relationship with at least another.
# essentially, if "subsumes" is a graph, it is transient, and we want weakly connected components
def filter_contained_rules_keep_max_bias_cumulative(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    subgroup_costs: Dict[Predicate, Dict[str, float]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret = dict()
    cost_values = {subgroup: costs.values() for subgroup, costs in subgroup_costs.items()}
    bias_measures = {subgroup: max(costs) - min(costs) for subgroup, costs in cost_values.items()}
    flags_keep = {subgroup: True for subgroup, _thens in rulesbyif.items()}
    for ifclause, thenclauses in rulesbyif.items():
        allthens = [then for _sg, (_cov, sg_thens) in thenclauses.items() for then, _cor, _cost in sg_thens]
        for otherif, _ in rulesbyif.items():
            if not ifclause.contains(otherif):
                continue
            extra_features = list(set(ifclause.features) - set(otherif.features))
            if len(extra_features) == 0:
                continue
            if_and_allthens_relevant_values = [tuple(then.to_dict()[feat] for feat in extra_features) for then in allthens]
            if_and_allthens_relevant_values.append(tuple(ifclause.to_dict()[feat] for feat in extra_features))
            if Series(if_and_allthens_relevant_values).unique().size == 1:
                if bias_measures[ifclause] > bias_measures[otherif]:
                    flags_keep[otherif] = False
                else:
                    flags_keep[ifclause] = False
    
    for ifclause, thenclauses in rulesbyif.items():
        if flags_keep[ifclause]:
            ret[ifclause] = thenclauses
    return ret

def delete_fair_rules_cumulative(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    subgroup_costs: Dict[Predicate, Dict[str, float]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        curr_subgroup_costs = list(subgroup_costs[ifclause].values())
        max_intergroup_cost_diff = max(curr_subgroup_costs) - min(curr_subgroup_costs)
        if max_intergroup_cost_diff == 0 or np.isnan(max_intergroup_cost_diff):
            continue
        ret[ifclause] = thenclauses
    return ret

def keep_only_minimum_change_cumulative(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    params: ParameterProxy = ParameterProxy()
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        ret[ifclause] = dict()
        for sg, (cov, thens) in thenclauses.items():
            min_change = min((featureChangePred(ifclause, then, params=params) for then, _, _ in thens), default=np.inf)
            newthens = [(then, cor, cost) for then, cor, cost in thens if featureChangePred(ifclause, then, params=params) <= min_change]
            ret[ifclause][sg] = (cov, newthens)
    return ret