from typing import Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from predicate import Predicate, recIsValid, featureCostPred, featureChangePred
from models import ModelAPI
from recourse_sets import TwoLevelRecourseSet

def incorrectRecoursesSingle(sd: Predicate, h: Predicate, s: Predicate, X_aff: DataFrame, model: ModelAPI) -> int:
    assert recIsValid(h, s)
    # X_aff_subgroup = X_aff[[h.satisfies(x) for i, x in X_aff.iterrows()]]
    X_aff_covered = X_aff[X_aff.apply(lambda x: h.satisfies(x) and sd.satisfies(x), axis=1)].copy()
    if X_aff_covered.shape[0] == 0:
        return 0

    X_aff_covered.loc[:, s.features] = s.values # type: ignore
    preds = model.predict(X_aff_covered)
    return np.shape(preds)[0] - np.sum(preds)

def incorrectRecourses(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI) -> int:
    new_rows = []
    for _, x in X_aff.iterrows():
        for s in R.suggest(x):
            x_corrected = x.copy()
            x_corrected[s.features] = s.values
            new_rows.append(x_corrected.to_frame().T)
    X_changed = pd.concat(new_rows, ignore_index=True)
    preds = model.predict(X_changed)
    return np.shape(preds)[0] - np.sum(preds)

def incorrectRecoursesSubmodular(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI) -> int:
    triples = R.to_triples()
    covered = set()
    corrected = set()
    for sd, h, s in triples:
        X_aff_covered_indicator = X_aff.apply(lambda x: h.satisfies(x) and sd.satisfies(x), axis=1).to_numpy()
        X_copy = X_aff.copy()
        X_copy.loc[:, s.features] = s.values # type: ignore
        all_preds = model.predict(X_copy)
        covered_and_corrected = np.logical_and(all_preds, X_aff_covered_indicator).nonzero()[0]

        covered.update(X_aff_covered_indicator.nonzero()[0].tolist())
        corrected.update(covered_and_corrected.tolist())
    return len(covered - corrected)

def coverSingle(p: Predicate, X_aff: DataFrame) -> int:
    return sum(1 for _, x in X_aff.iterrows() if p.satisfies(x))

def cover(R: TwoLevelRecourseSet, X_aff: DataFrame, percentage=False):
    suggestions = [list(R.suggest(x)) for _, x in X_aff.iterrows()]
    ret = len([ss for ss in suggestions if len(ss) > 0])
    if percentage:
        return ret / X_aff.shape[0]
    else:
        return ret

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


