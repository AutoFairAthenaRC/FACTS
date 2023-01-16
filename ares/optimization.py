from typing import List, Set, Tuple
import itertools
import operator

import numpy as np
from pandas import DataFrame

from predicate import Predicate, featureChangePred, featureCostPred, recIsValid
from models import ModelAPI
from metrics import incorrectRecoursesSingle

def optimizer(modulars: List[int], covers: List[Set[int]], N_aff: int):
    assert len(modulars) == len(covers)

    N = len(modulars)
    singleton_rewards = [mod + len(cov) for mod, cov in zip(modulars, covers)]

    argmax_singleton = np.argmax(singleton_rewards)
    subset = set([argmax_singleton])
    excluded = set(np.arange(N))
    excluded.remove(argmax_singleton)

    curr_modular = modulars[argmax_singleton]
    curr_cover = len(covers[argmax_singleton])
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
                    updated_cover -= 1
            
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
                    updated_cover += 1
            
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
                        updated_cover -= 1
                for obj in covers[idx2]:
                    ref_counts[obj] += 1
                    if ref_counts[obj] == 1:
                        updated_cover += 1
                
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

def optimize(SD: List[Predicate], RL: List[Predicate], X_aff: DataFrame, model: ModelAPI) -> Tuple[List[Tuple[Predicate, Predicate, Predicate]], int, int, int, int]:
    all_triples = [(sd, h, s) for sd, h, s in itertools.product(SD, RL, RL) if recIsValid(h, s)]
    triples_no = len(all_triples)
    print(f"Total triples = {triples_no}")
    affected_no = X_aff.shape[0]
    print(f"X_aff shape before: {X_aff.shape}")
    all_incorrects = list(-incorrectRecoursesSingle(sd, h, s, X_aff, model) for sd, h, s in all_triples)
    print("Calculated incorrect recourse for each triple")
    all_feature_costs = list(-featureCostPred(h, s) for _, h, s in all_triples)
    print("Calculated feature costs for each triple")
    all_feature_changes = list(-featureChangePred(h, s) for _, h, s in all_triples)
    print("Calculated feature changes for each feature")
    print(f"X_aff shape after: {X_aff.shape}")
    
    # triples_covers: List[set] = [set() for i in range(triples_no)]
    triples_covers: List = [set() for i in range(triples_no)]
    for i, (sd, h, s) in enumerate(all_triples):
        X_aff_covered = X_aff.apply(lambda x: h.satisfies(x) and sd.satisfies(x), axis=1).to_numpy()
        nonzeros = X_aff_covered.nonzero()
        nonzeros_first = nonzeros[0]
        triples_covers[i] = set(nonzeros_first)
        # triples_covers[i] = set(X_aff_covered.nonzero()[0])
    ref_counts: List[int] = [0] * affected_no
    print("Calculated covers for each triple")
    
    # print(triples_covers)
    almost_all = [inc + cost + change for inc, cost, change in zip(all_incorrects, all_feature_costs, all_feature_changes)]
    all_single = list(map(operator.add, almost_all, [len(cover) for cover in triples_covers]))

    argmax_total = np.argmax(all_single)
    print(argmax_total)

    my_subset = set([argmax_total])
    excluded = set(np.array([i for i in range(triples_no) if i != argmax_total]))
    curr_modular = almost_all[argmax_total]
    curr_cover = len(triples_covers[argmax_total])
    for j in triples_covers[argmax_total]:
        ref_counts[j] += 1
    flag_continue = True
    while flag_continue:
        flag_continue = False

        # try delete
        for idx in my_subset:
            updated_modular = curr_modular - almost_all[idx]
            updated_cover = curr_cover
            for j in triples_covers[idx]:
                ref_counts[j] -= 1
                if ref_counts[j] < 0:
                    raise IndexError("Something went wrong. Reference count negative.")
                elif ref_counts[j] == 0:
                    updated_cover -= 1
            if updated_modular + updated_cover > curr_modular + curr_cover:
                curr_modular = updated_modular
                curr_cover = updated_cover
                my_subset.remove(idx)
                excluded.add(idx)
                flag_continue = True
                break
            else:
                for j in triples_covers[idx]:
                    ref_counts[j] += 1
        
        # try add
        for idx in excluded:
            updated_modular = curr_modular + almost_all[idx]
            updated_cover = curr_cover
            for j in triples_covers[idx]:
                ref_counts[j] += 1
                if ref_counts[j] == 1:
                    updated_cover += 1
            if updated_modular + updated_cover > curr_modular + curr_cover:
                curr_modular = updated_modular
                curr_cover = updated_cover
                my_subset.add(idx)
                excluded.remove(idx)
                flag_continue = True
                break
            else:
                for j in triples_covers[idx]:
                    ref_counts[j] -= 1
                    if ref_counts[j] < 0:
                        raise IndexError("Something went wrong. Reference count negative.")
        
        # try exchange
        for idx1 in my_subset:
            for idx2 in excluded:
                updated_modular = curr_modular + almost_all[idx2] - almost_all[idx1]
                updated_cover = curr_cover
                for j in triples_covers[idx1]:
                    ref_counts[j] -= 1
                    if ref_counts[j] < 0:
                        raise IndexError("Something went wrong. Reference count negative.")
                    elif ref_counts[j] == 0:
                        updated_cover -= 1
                for j in triples_covers[idx2]:
                    ref_counts[j] += 1
                    if ref_counts[j] == 1:
                        updated_cover += 1
                if updated_modular + updated_cover > curr_modular + curr_cover:
                    curr_modular = updated_modular
                    curr_cover = updated_cover
                    my_subset.add(idx2)
                    my_subset.remove(idx1)
                    excluded.remove(idx2)
                    excluded.add(idx1)
                    flag_continue = True
                    break
                else:
                    for j in triples_covers[idx2]:
                        ref_counts[j] -= 1
                        if ref_counts[j] < 0:
                            raise IndexError("Something went wrong. Reference count negative.")
                    for j in triples_covers[idx1]:
                        ref_counts[j] += 1
    
    final_incorrects = sum([all_incorrects[i] for i in my_subset])
    final_coverage = len(set().union(*[triples_covers[i] for i in my_subset]))
    final_feature_cost = sum([all_feature_costs[i] for i in my_subset])
    final_feature_change = sum([all_feature_changes[i] for i in my_subset])
    return [all_triples[i] for i in my_subset], final_incorrects, final_coverage, final_feature_cost, final_feature_change