from dataclasses import dataclass, field
from typing import List, Dict, Mapping
import operator
import functools

import pandas as pd
from pandas import DataFrame
from .parameters import ParameterProxy

value_t = str | int | float | pd.Interval

@functools.total_ordering
@dataclass
class Predicate:
    """Represents a predicate with features and values."""

    features: List[str] = field(default_factory=list)
    values: List[value_t] = field(default_factory=list)
    _interval_feats: List[str] = field(default_factory=list)

    def __eq__(self, __o: object) -> bool:
        """
        Checks if the predicate is equal to another predicate.
        """
        if not isinstance(__o, Predicate):
            return False

        d1 = self.to_dict()
        d2 = __o.to_dict()
        return d1 == d2

    def __lt__(self, __o: object) -> bool:
        """
        Compares the predicate with another predicate based on their representations.
        Goal is to induce an arbitrary ordering on predicates.
        """
        return repr(self) < repr(__o)

    def __hash__(self) -> int:
        """
        Returns the hash value of the predicate.
        """
        return hash(repr(self))

    def __str__(self) -> str:
        """
        Returns the string representation of the predicate.
        """
        ret = []
        first_iter = True
        for f, v in zip(self.features, self.values):
            if first_iter:
                first_iter = False
            else:
                ret.append(", ")

            ret.append(f"{f} = {v}")
        return "".join(ret)

    def __post_init__(self):
        """
        Initializes the predicate after the data class is created.
        """
        pairs = sorted(zip(self.features, self.values))
        self.features = [f for f, _v in pairs]
        self.values = [v for _f, v in pairs]
        for f, v in pairs:
            if isinstance(v, pd.Interval):
                self._interval_feats.append(f)

    @staticmethod
    def from_dict(d: Dict[str, value_t]) -> "Predicate":
        """
        Creates a Predicate instance from a dictionary.

        Args:
            d: A dictionary representing the predicate.

        Returns:
            A Predicate instance.
        """
        feats = list(d.keys())
        vals = list(d.values())
        return Predicate(features=feats, values=vals)

    def to_dict(self) -> Dict[str, value_t]:
        """
        Converts the predicate to a dictionary representation.

        Returns:
            A dictionary representing the predicate.
        """
        return dict(zip(self.features, self.values))

    def satisfies(self, x: Mapping[str, str | int | float]) -> bool:
        """
        Checks if the predicate is satisfied by a given input.

        Args:
            x: The input to be checked against the predicate.

        Returns:
            True if the predicate is satisfied, False otherwise.
        """
        for feat, val in zip(self.features, self.values):
            x_val = x[feat]
            if isinstance(val, pd.Interval) and not isinstance(x_val, str):
                if x_val not in val:
                    return False
            else:
                if x_val != val:
                    return False
        return True
    
    def satisfies_v(self, X: DataFrame) -> pd.Series[bool]:
        simple_feats = [feat for feat in self.features if feat not in self._interval_feats]
        simple_vals = [val for feat, val in zip(self.features, self.values) if feat not in self._interval_feats]
        interval_feats = [feat for feat in self.features if feat in self._interval_feats]

        X_covered_bool_simple = (X[simple_feats] == simple_vals).all(axis=1)

        # X_c = X[interval_feats]
        # inters = [self.to_dict()[f] for f in interval_feats]
        # X_c.apply(lambda x: x, axis="index")
        X_covered_bool_interval = X_covered_bool_simple.map(lambda x: True)
        d = self.to_dict()
        for feat in interval_feats:
            interval = d[feat]
            assert isinstance(interval, pd.Interval)
            indicator = (X[feat] > interval.left) & (X[feat] <= interval.right)
            X_covered_bool_interval &= indicator
        return (X_covered_bool_simple & X_covered_bool_interval)

    def width(self):
        """
        Returns the number of features in the predicate.
        """
        return len(self.features)

    def contains(self, other: object) -> bool:
        """
        Checks if the predicate contains another predicate.

        Args:
            other: The predicate to check for containment.

        Returns:
            True if the predicate contains the other predicate, False otherwise.
        """
        if not isinstance(other, Predicate):
            return False

        d1 = self.to_dict()
        d2 = other.to_dict()
        return all(feat in d1 and d1[feat] == val for feat, val in d2.items())

def featureChangePred(
    p1: Predicate, p2: Predicate, params: ParameterProxy = ParameterProxy()
):
    """
    Calculates the feature change between two predicates.

    Args:
        p1: The first Predicate.
        p2: The second Predicate.
        params: The ParameterProxy object containing feature change functions.

    Returns:
        The feature change between the two predicates.
    """
    total = 0
    for i, f in enumerate(p1.features):
        val1 = p1.values[i]
        val2 = p2.values[i]
        costChange = params.featureChanges[f](val1, val2)
        # if f in params.num_features:
        #     i1, i2 = val1, val2
        #     assert isinstance(i1, pd.Interval) and isinstance(i2, pd.Interval)
        total += costChange
    return total

def recIsValid(
    p1: Predicate, p2: Predicate, X: DataFrame, drop_infeasible: bool
) -> bool:
    """
    Checks if the given pair of predicates is valid based on the provided conditions.

    Args:
        p1: The first Predicate.
        p2: The second Predicate.
        X: The DataFrame containing the data.
        drop_infeasible: Flag indicating whether to drop infeasible cases.

    Returns:
        True if the pair of predicates is valid, False otherwise.
    """
    n1 = len(p1.features)
    n2 = len(p2.features)
    if n1 != n2:
        return False

    featuresMatch = all(map(operator.eq, p1.features, p2.features))
    existsChange = any(map(operator.ne, p1.values, p2.values))

    # accept only those if-then pairs where if and then have the same features
    # and at least one of them is changed
    if not (featuresMatch and existsChange):
        return False

    # reject recourses that involve - and change - all features.
    if n1 == len(X.columns) and all(map(operator.ne, p1.values, p2.values)):
        return False

    if drop_infeasible == True:
        feat_change = True
        for count, feat in enumerate(p1.features):
            if p1.values[count] != "Unknown" and p2.values[count] == "Unknown":
                return False
            if isinstance(p1.values[count], pd.Interval):
                if isinstance(p2.values[count], pd.Interval):
                    if p1.values[count].overlaps(p2.values[count]):
                        return False
                else:
                    raise ValueError("Cannot have interval for an if and not interval for then")
            if feat == "parents":
                parents_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and parents_change
            if feat == "age":
                age_change = p1.values[count].left <= p2.values[count].left
                feat_change = feat_change and age_change
            elif feat == "ages":
                age_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and age_change
            elif feat == "education-num":
                edu_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and edu_change
            elif feat == "PREDICTOR RAT AGE AT LATEST ARREST":
                age_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and age_change
            elif feat == "age_cat":
                age_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and age_change
            elif feat == "sex":
                race_change = p1.values[count] == p2.values[count]
                feat_change = feat_change and race_change
        return feat_change
    
    return True


def drop_two_above(p1: Predicate, p2: Predicate, l: list) -> bool:
    """
    Checks if the values of the given predicates are within a difference of two based on the provided conditions.

    Args:
        p1: The first Predicate.
        p2: The second Predicate.
        l: The list of values for comparison.

    Returns:
        True if the values are within a difference of two, False otherwise.
    """
    feat_change = True

    for count, feat in enumerate(p1.features):
        if feat == "education-num":
            edu_change = p2.values[count] - p1.values[count] <= 2
            feat_change = feat_change and edu_change
        elif feat == "age":
            age_change = (
                l.index(p2.values[count].left) - l.index(p1.values[count].left) <= 2
            )
            feat_change = feat_change and age_change
        elif feat == "PREDICTOR RAT AGE AT LATEST ARREST":
            age_change = l.index(p2.values[count]) - l.index(p1.values[count]) <= 2
            feat_change = feat_change and age_change

    return feat_change
