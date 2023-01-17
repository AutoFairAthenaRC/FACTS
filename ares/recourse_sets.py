from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
from pandas import Series

from predicate import Predicate

@dataclass
class RecourseSet:
    hypotheses: List[Predicate] = field(default_factory=list)
    suggestions: List[Predicate] = field(default_factory=list)

    @staticmethod
    def fromList(l: List[Tuple[Dict[str,str], Dict[str, str]]]):
        R = RecourseSet()
        hs = [Predicate.from_dict(rule[0]) for rule in l]
        ss = [Predicate.from_dict(rule[1]) for rule in l]
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

    def to_pairs(self) -> List[Tuple[Predicate, Predicate]]:
        return [(h, s) for h, s in zip(self.hypotheses, self.suggestions)]

@dataclass
class TwoLevelRecourseSet:
    feature: str
    values: List[str]
    rules: Dict[str, RecourseSet] = field(default_factory=dict)

    def __str__(self) -> str:
        ret = []
        for val in self.values:
            ret.append(f"If {self.feature} = {val}:\n")
            rules = self.rules[val]
            for h, s in zip(rules.hypotheses, rules.suggestions):
                ret.append(f"\tIf {h},\n\tThen {s}.\n")
        return "".join(ret)
    
    @staticmethod
    def from_triples(l: List[Tuple[Predicate, Predicate, Predicate]]):
        feat = l[0][0].features[0]
        values = np.unique([t[0].values[0] for t in l]).tolist()
        rules = {val: RecourseSet() for val in values}
        for sd, h, s in l:
            rules[sd.values[0]].hypotheses.append(h)
            rules[sd.values[0]].suggestions.append(s)
        return TwoLevelRecourseSet(feat, values, rules)
    
    def to_triples(self) -> List[Tuple[Predicate, Predicate, Predicate]]:
        l = []
        for val, ifthens in self.rules.items():
            sd = Predicate.from_dict({self.feature: val})
            l.extend([(sd, h, s) for h, s in ifthens.to_pairs()])
        return l

    def addRules(self, val: str, l: List[Tuple[Dict[str,str], Dict[str, str]]]) -> None:
        self.rules[val] = RecourseSet.fromList(l)
    
    def suggest(self, x):
        x_belongs_to = x[self.feature]
        if x_belongs_to in self.rules:
            return self.rules[x_belongs_to].suggest(x)
        else:
            return []