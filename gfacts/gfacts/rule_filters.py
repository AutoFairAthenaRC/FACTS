from typing import Dict, Tuple, List
from pandas import Series

from .predicate import Predicate

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
            allthens_relevant_values = [tuple(then.to_dict()[feat] for feat in extra_features) for then in allthens]
            if Series(allthens_relevant_values).unique().size == 1:
                flag_keep = False

        if flag_keep:
            ret[ifclause] = thenclauses
    return ret