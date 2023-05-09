from pandas import DataFrame

from .recourse_sets import TwoLevelRecourseSet
from .models import ModelAPI
from .metrics import incorrectRecourses, cover, featureCost, featureChange

from .parameters import epsilon1, epsilon2, C_max, M_max

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