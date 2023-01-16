from dataclasses import dataclass, field
from typing import List, Any, Dict
from enum import Enum
import operator

from lib2 import featureChanges, featureCosts

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

@dataclass
class Predicate:
    features: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    operators: List[Operator] = field(default_factory=list, repr=False)

    def __post_init__(self, operators=None):
        if operators is None:
            self.operators = [Operator.EQ for _ in range(len(self.features))]
        else:
            self.operators = operators
    
    @staticmethod
    def from_dict(d: Dict[str, str]) -> "Predicate":
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
    # return sum(featureChanges[feat](p1.values[i], p2.values[i]) for i, feat in enumerate(p1.features))
    total = 0
    for i, f in enumerate(p1.features):
        val1 = p1.values[i]
        val2 = p2.values[i]
        costChange = featureChanges[f](val1, val2)
        total += costChange
    return total

def recIsValid(p1: Predicate, p2: Predicate) -> bool:
    n1 = len(p1.features)
    n2 = len(p2.features)
    if n1 != n2:
        return False
    featuresMatch = all(map(operator.eq, p1.features, p2.features))
    existsChange = any(map(operator.ne, p1.values, p2.values))
    return featuresMatch and existsChange
