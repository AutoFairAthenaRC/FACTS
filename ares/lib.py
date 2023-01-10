from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar, List, Dict, Tuple, Protocol
import operator
from collections import defaultdict
import itertools
from functools import partial

import pandas as pd
import numpy as np

from numpy.typing import ArrayLike
from pandas.core.series import Series
from pandas import DataFrame

from pprint import pprint

##### Parameters
# CLASS1 = "MALE"
# CLASS2 = "FEMALE"
epsilon1 = 20
epsilon2 = 7
epsilon3 = 10
featureCosts = defaultdict(lambda: 1, {"Sex": 100})
C_max = max(featureCosts.values())
M_max = 1

lincorrect = 1
lcover = 1
lfcost = 5
lfchange = 5
##### Parameters

class Operator(Enum):
    EQ = "="
    GEQ = ">="
    LEQ = "<="

    def eval(self, x, y) -> bool:
        ops = {
            Operator.EQ: operator.eq,
            Operator.GEQ: operator.ge,
            Operator.LEQ: operator.le
            }
        return ops[self](x, y)

# Self = TypeVar("Self", bound="Predicate")
@dataclass
class Predicate:
    features: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    operators: List[Operator] = field(default_factory=list, repr=False)

    @staticmethod
    def from_dict_categorical(d: Dict[str, str]) -> "Predicate":
        p = Predicate()
        p.features = list(d.keys())
        p.values = list(d.values())
        p.operators = [Operator.EQ for _ in range(len(d))]
        return p
    
    def satisfies(self, x) -> bool:
        return all(op.eval(x[feat], val) for feat, val, op in zip(self.features, self.values, self.operators))
    
    def width(self):
        return len(self.features)
    
def featureCostPred(p1: Predicate, p2: Predicate):
    ret = 0
    for i, f in enumerate(p1.features):
        if p1.values[i] != p2.values[i]:
            ret += featureCosts[f]
    return ret

def featureChangePred(p1: Predicate, p2: Predicate):
    return sum(v1 != v2 for v1, v2 in zip(p1.values, p2.values))

def recIsValid(p1: Predicate, p2: Predicate) -> bool:
    n1 = len(p1.features)
    n2 = len(p2.features)
    if n1 != n2:
        return False
    featuresMatch = all(map(operator.eq, p1.features, p2.features))
    existsChange = any(map(operator.ne, p1.values, p2.values))
    return featuresMatch and existsChange

@dataclass
class RecourseSet:
    hypotheses: List[Predicate] = field(default_factory=list)
    suggestions: List[Predicate] = field(default_factory=list)

    @staticmethod
    def fromList(l: List[Tuple[Dict[str,str], Dict[str, str]]]):
        R = RecourseSet()
        hs = [Predicate.from_dict_categorical(rule[0]) for rule in l]
        ss = [Predicate.from_dict_categorical(rule[1]) for rule in l]
        R.hypotheses = hs
        R.suggestions = ss
        return R
    
    def __post_init__(self):
        try:
            assert len(self.hypotheses) == len(self.suggestions)
        except AssertionError:
            print("--> Number of antecedents and consequents should be equal.")
            raise
    
    def suggest(self, x: Series):
        for h, s in zip(self.hypotheses, self.suggestions):
            if h.satisfies(x):
                yield s

@dataclass
class TwoLevelRecourseSet:
    feature: str
    values: List[str]
    rules: Dict[str, RecourseSet] = field(default_factory=dict)

    def addRules(self, val: str, l: List[Tuple[Dict[str,str], Dict[str, str]]]) -> None:
        self.rules[val] = RecourseSet.fromList(l)
    
    def suggest(self, x):
        x_belongs_to = x[self.feature]
        return self.rules[x_belongs_to].suggest(x)
    
    def addRule(self, subgroup: str, h: Predicate, s: Predicate):
        self.rules[subgroup].addRecourse(h, s)

class ModelAPI(Protocol):
    def predict(self, X: ArrayLike) -> ArrayLike:
        ...

def incorrectRecoursesSingle(sd: Predicate, h: Predicate, s: Predicate, X_aff: DataFrame, model: ModelAPI) -> int:
    assert recIsValid(h, s)
    # X_aff_subgroup = X_aff[[h.satisfies(x) for i, x in X_aff.iterrows()]]
    new_rows = []
    for _, x in X_aff.iterrows():
        if h.satisfies(x) and sd.satisfies(x):
            x_corrected = x.copy()
            x_corrected[s.features] = s.values
            new_rows.append(x_corrected.to_frame().T)
    
    if len(new_rows) == 0:
        return 0
    X_changed = pd.concat(new_rows, ignore_index=True)
    preds = model.predict(X_changed)
    return np.shape(preds)[0] - np.sum(preds)

def incorrectRecourses(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI):
    new_rows = []
    for _, x in X_aff.iterrows():
        for s in R.suggest(x):
            x_corrected = x.copy()
            x_corrected[s.features] = s.values
            new_rows.append(x_corrected.to_frame().T)
    X_changed = pd.concat(new_rows, ignore_index=True)
    preds = model.predict(X_changed)
    return np.shape(preds)[0] - np.sum(preds)

def cover(R: TwoLevelRecourseSet, X_aff: DataFrame):
    suggestions = [list(R.suggest(x)) for _, x in X_aff.iterrows()]
    return len([s for s in suggestions if s != []])

def featureCost(R: TwoLevelRecourseSet):
    return sum(featureCostPred(h, s) for r in R.rules.values() for h, s in zip(r.hypotheses, r.suggestions))

def featureChange(R: TwoLevelRecourseSet):
    return sum(featureChangePred(h, s) for r in R.rules.values() for h, s in zip(r.hypotheses, r.suggestions))

def size(R: TwoLevelRecourseSet):
    return sum(len(r.hypotheses) for r in R.rules.values())

def maxwidth(R: TwoLevelRecourseSet):
    return max(p.width() for r in R.rules.values() for p in r.hypotheses)

def numrsets(R: TwoLevelRecourseSet):
    return len(R.values)



def reward1(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI) -> float:
    U_1 = len(X_aff) * epsilon1
    return U_1 - incorrectRecourses(R, X_aff, model)

reward2 = cover

def reward3(R: TwoLevelRecourseSet) -> float:
    U_3 = C_max * epsilon1 * epsilon2
    return U_3 - featureCost(R)

def reward4(R: TwoLevelRecourseSet) -> float:
    U_4 = M_max * epsilon1 * epsilon2
    return U_4 - featureChange(R)



def optimize(SD: List[Predicate], RL: List[Predicate], X_aff: DataFrame, model: ModelAPI) -> Tuple[List[Tuple[Predicate, Predicate, Predicate]], int, int, int, int]:
    all_triples = list(((sd, h, s) for sd, h, s in itertools.product(SD, RL, RL) if recIsValid(h, s)))
    triples_no = len(all_triples)
    affected_no = X_aff.shape[0]
    all_incorrects = list(-lincorrect * incorrectRecoursesSingle(sd, h, s, X_aff, model) for sd, h, s in all_triples)
    all_feature_costs = list(-lfcost * featureCostPred(h, s) for _, h, s in all_triples)
    all_feature_changes = list(-lfchange * featureChangePred(h, s) for _, h, s in all_triples)

    triples_covers: Dict[int, set] = {i: set() for i in range(triples_no)}
    for i, (sd, h, s) in enumerate(all_triples):
        for j, x in X_aff.iterrows():
            if sd.satisfies(x) and h.satisfies(x):
                triples_covers[i].add(j)
    
    # print(triples_covers)
    almost_all = list(map(operator.add, all_incorrects, map(operator.add, all_feature_costs, all_feature_changes)))
    all_single = list(map(operator.add, almost_all, [len(cover) for cover in triples_covers.values()]))

    argmax_total = np.argmax(all_single)
    my_subset = set([argmax_total])
    excluded = set(np.array([i for i in range(triples_no) if i != argmax_total]))
    curr_score = all_single[argmax_total]
    flag_continue = True
    while flag_continue:
        flag_continue = False

        # try delete
        for idx in my_subset:
            cand_subset = my_subset - {idx}
            new_score = sum(almost_all[i] for i in cand_subset) + lcover * len(set().union(*[triples_covers[i] for i in cand_subset]))
            if new_score > curr_score:
                curr_score = new_score
                my_subset.remove(idx)
                excluded.add(idx)
                flag_continue = True
                break
        
        # try add
        for idx in excluded:
            cand_subset = my_subset | {idx}
            new_score = sum(almost_all[i] for i in cand_subset) + lcover * len(set().union(*[triples_covers[i] for i in cand_subset]))
            if new_score > curr_score:
                curr_score = new_score
                my_subset.add(idx)
                excluded.remove(idx)
                flag_continue = True
                break
        
        # try exchange
#        for idx1 in my_subset:
#            for idx2 in excluded:
#                cand_subset = (my_subset | {idx2}) - {idx1}
#                new_score = sum(almost_all[i] for i in cand_subset) + len(set().union(*[triples_covers[i] for i in cand_subset]))
#                if new_score > curr_score:
#                    curr_score = new_score
#                    my_subset.add(idx2)
#                    my_subset.remove(idx1)
#                    excluded.add(idx1)
#                    excluded.remove(idx2)
#                    flag_continue = False
#                    break
    
    final_incorrects = sum([all_incorrects[i] for i in my_subset])
    final_coverage = len(set().union(*[triples_covers[i] for i in my_subset]))
    final_feature_cost = sum([all_feature_costs[i] for i in my_subset])
    final_feature_change = sum([all_feature_changes[i] for i in my_subset])
    return [all_triples[i] for i in my_subset], final_incorrects, final_coverage, final_feature_cost, final_feature_change


