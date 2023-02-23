import itertools

from typing import List, Set, Tuple, Dict
from collections import defaultdict

import numpy as np
from pandas import DataFrame

from .predicate import Predicate, featureChangePred, featureCostPred, recIsValid
from .models import ModelAPI
from .metrics import incorrectRecoursesSingle
from .parameters import ParameterProxy

def ground_set_generation_vanilla(SD: List[Predicate], RL: List[Predicate], X_aff: DataFrame, model: ModelAPI):
    valid_triples = [(sd, h, s) for sd in SD for h in RL for s in RL if recIsValid(h, s)]
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
    all_triples = [(sd, h, s) for sd in SD for h in ifs[sd.values[0]] for s in thens[sd.values[0]] if recIsValid(h, s)]
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

def optimize_sameif(
    SD: List[Predicate],
    ifs: List[Predicate],
    thens: Dict[str, List[Predicate]],
    X_aff: DataFrame,
    model: ModelAPI,
    params: ParameterProxy = ParameterProxy()
) -> List[Tuple[Predicate, Predicate, Predicate]]:
    sensitive_attr_values = [sd.values[0] for sd in SD]
    atoms = []
    print(thens.items())
    for ifclause in ifs:
        for thenlist in itertools.product(*list(thens.items())):
            print(thenlist)
            if all(recIsValid(ifclause, thenclause) for _, thenclause in thenlist):
                atoms.append((ifclause, dict(thenlist)))
    print(len(atoms))

    raise NotImplementedError


