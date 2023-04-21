from dataclasses import dataclass, field
from typing import List, Any, Dict
from enum import Enum
import operator
from pandas import DataFrame
from .parameters import ParameterProxy

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

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Predicate):
            return False
        
        d1 = self.to_dict()
        d2 = __o.to_dict()
        return d1 == d2
    
    def __hash__(self) -> int:
        return hash(repr(self))
    
    def __str__(self) -> str:
        ret = []
        first_iter = True
        for f, v in zip(self.features, self.values):
            if first_iter:
                first_iter = False
            else:
                ret.append(", ")
            
            ret.append(f"{f} = {v}")
        return "".join(ret)
    
    def __post_init__(self, operators=None):
        feats, vals = zip(*sorted(zip(self.features, self.values)))
        self.features = list(feats)
        self.values = list(vals)
        if operators is None:
            self.operators = [Operator.EQ for _ in range(len(self.features))]
        else:
            self.operators = operators
    
    @staticmethod
    def from_dict(d: Dict[str, str]) -> "Predicate":
        feats = list(d.keys())
        vals = list(d.values())
        ops = [Operator.EQ for _ in range(len(d))]
        return Predicate(features=feats, values=vals, operators=ops)
    
    def to_dict(self) -> Dict[str, str]:
        return dict(zip(self.features, self.values))
    
    def satisfies(self, x) -> bool:
        return all(op.eval(x[feat], val) for feat, val, op in zip(self.features, self.values, self.operators))
    
    def width(self):
        return len(self.features)
    
    def contains(self, other: object) -> bool:
        if not isinstance(other, Predicate):
            return False
        
        d1 = self.to_dict()
        d2 = other.to_dict()
        return all(feat in d1 and d1[feat] == val for feat, val in d2.items())

def featureCostPred(p1: Predicate, p2: Predicate, params: ParameterProxy = ParameterProxy()):
    ret = 0
    for i, f in enumerate(p1.features):
        if p1.values[i] != p2.values[i]:
            ret += params.featureCosts[f]
    return ret

def featureChangePred(p1: Predicate, p2: Predicate, params: ParameterProxy = ParameterProxy()):
    # return sum(featureChanges[feat](p1.values[i], p2.values[i]) for i, feat in enumerate(p1.features))
    total = 0
    for i, f in enumerate(p1.features):
        val1 = p1.values[i]
        val2 = p2.values[i]
        costChange = params.featureChanges[f](val1, val2)
        total += costChange
    return total

#def recIsValid(p1: Predicate, p2: Predicate) -> bool:
# def recIsValid(p1: Predicate, p2: Predicate) -> bool:
#     n1 = len(p1.features)
#     n2 = len(p2.features)
#     if n1 != n2:
#         return False
#     featuresMatch = all(map(operator.eq, p1.features, p2.features))
#     existsChange = any(map(operator.ne, p1.values, p2.values))
#     return featuresMatch and existsChange

def recIsValid(p1: Predicate, p2: Predicate,X: DataFrame ,drop_infeasible: bool) -> bool:
    feat_change = True
    n1 = len(p1.features)
    n2 = len(p2.features)
    if n1 != n2:
        return False
    
    featuresMatch = all(map(operator.eq, p1.features, p2.features))
    existsChange = any(map(operator.ne, p1.values, p2.values))
    
    if n1 == len(X.columns) and all(map(operator.ne, p1.values, p2.values)):
        return False

    if drop_infeasible == True:
        if all(map(operator.eq, p1.features, p2.features)) and any(map(operator.ne, p1.values, p2.values)):
            for count,feat in enumerate(p1.features):
                if feat == 'age':
                    age_change = p1.values[count].left <= p2.values[count].left
                    feat_change = feat_change and age_change
                elif feat == 'education-num':
                    edu_change = p1.values[count] <= p2.values[count]
                    feat_change = feat_change and edu_change
            return feat_change
        else: return False
    else:
        return featuresMatch and existsChange
    
def drop_two_above(p1: Predicate, p2: Predicate,l: list) -> bool:
    feat_change = True

    for count,feat in enumerate(p1.features):
        if feat=='education-num':
            edu_change =  p2.values[count] - p1.values[count] <=2
            feat_change = feat_change and edu_change
        elif feat == 'age':
            age_change = l.index(p2.values[count].left) - l.index(p1.values[count].left) <=2
            feat_change = feat_change and age_change

    return feat_change
