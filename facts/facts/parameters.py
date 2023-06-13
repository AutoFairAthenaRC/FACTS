from typing import Dict, Callable, Any, List
import functools
from collections import defaultdict
from dataclasses import dataclass, field


def make_default_featureCosts():
    """Creates a defaultdict with a default value of 1 for feature costs.

    Returns:
        defaultdict: The default dictionary of feature costs.
    """
    return defaultdict(lambda: 1)


def default_change(v1, v2):
    """Compares two values and returns 0 if they are equal, and 1 if they are different.

    Args:
        v1: The first value to be compared.
        v2: The second value to be compared.

    Returns:
        int: 0 if the values are equal, 1 if the values are diff
    """

    return 0 if v1 == v2 else 1


def make_default_featureChanges():
    """Creates a defaultdict with a default value of the default_change function.

    Returns:
        defaultdict: A defaultdict with default_change as the default value.
    """
    return defaultdict(lambda: default_change)


def naive_feature_change_builder(
    num_cols: List[str],
    cate_cols: List[str],
    feature_weights: Dict[str, int],
) -> Dict[str, Callable[[Any, Any], int]]:
    """Builds a dictionary of feature change functions based on the provided lists of numerical
    and categorical columns, along with the weights for each feature.

    Args:
        num_cols (List[str]): List of numerical column names.
        cate_cols (List[str]): List of categorical column names.
        feature_weights (Dict[str, int]): Dictionary mapping feature names to their weights.

    Returns:
        Dict[str, Callable[[Any, Any], int]]: Dictionary of feature change functions.
    """

    def feature_change_cate(v1, v2, weight):
        """Calculates the change between two categorical values based on the provided weight.

        Args:
            v1: The first categorical value.
            v2: The second categorical value.
            weight: The weight assigned to the feature.

        Returns:
            int: The change between the categorical values multiplied by the weight.
                Returns 0 if the values are equal, otherwise returns 1 multiplied by the weight.
        """
        return (0 if v1 == v2 else 1) * weight

    def feature_change_num(v1, v2, weight):
        """Calculates the change between two numerical values based on the provided weight.

        Args:
            v1: The first numerical value.
            v2: The second numerical value.
            weight: The weight assigned to the feature.

        Returns:
            int: The absolute difference between the numerical values multiplied by the weight.
        """
        return abs(v1 - v2) * weight

    ret_cate = {
        col: functools.partial(feature_change_cate, weight=feature_weights.get(col, 1))
        for col in cate_cols
    }
    ret_num = {
        col: functools.partial(feature_change_num, weight=feature_weights.get(col, 1))
        for col in num_cols
    }
    return {**ret_cate, **ret_num}


@dataclass
class ParameterProxy:
    """Proxy class for managing recourse parameters."""

    featureCosts: Dict[str, int] = field(default_factory=make_default_featureCosts)
    featureChanges: Dict[str, Callable[[Any, Any], int]] = field(
        default_factory=make_default_featureChanges
    )
    lambda_cover: int = 1
    lambda_correctness: int = 1
    lambda_featureCost: int = 1
    lambda_featureChange: int = 1

    ##### Utility methods for setting the parameters
    def setFeatureCost(self, fc: Dict):
        """Sets the feature costs.

        Args:
            fc (Dict): A dictionary mapping feature names to their costs.
        """
        self.featureCosts.update(fc)

    def setFeatureChange(self, fc: Dict):
        """Set the feature changes.

        Args:
            fc (Dict): A dictionary mapping feature names to their change functions.
        """
        self.featureChanges.update(fc)

    def set_lambdas(self, l1=1, l2=1, l3=1, l4=1):
        """Set the lambda for different components.

        Args:
            l1 (int, optional): Lambda for the cover. Defaults to 1.
            l2 (int, optional): Lambda for the correctness. Defaults to 1.
            l3 (int, optional): Lambda for the feature cost. Defaults to 1.
            l4 (int, optional): Lambda for the feature change. Defaults to 1.
        """
        self.lambda_cover = l1
        self.lambda_correctness = l2
        self.lambda_featureCost = l3
        self.lambda_featureChange = l4

    ##### Utility methods for setting the parameters


##### Unused parameters
epsilon1 = 20
epsilon2 = 7
epsilon3 = 10
C_max = 1  # max(featureCosts.values())
M_max = 1
##### Unused parameters
