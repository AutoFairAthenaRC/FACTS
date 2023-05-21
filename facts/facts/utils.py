from typing import Dict, List, Tuple
import dill
from pathlib import Path
from os import PathLike
from pandas import DataFrame

from .predicate import Predicate
from .models import ModelAPI

def load_rules_by_if(file: PathLike) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    p = Path(file)
    with p.open("rb") as inf:
        rules_by_if = dill.load(inf)
    return rules_by_if

def save_rules_by_if(file: PathLike, rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]) -> None:
    p = Path(file)
    with p.open("wb") as outf:
        dill.dump(rules, outf)

def load_test_data_used(file: PathLike) -> DataFrame:
    p = Path(file)
    with p.open("rb") as inf:
        X_test = dill.load(inf)
    return X_test

def save_test_data_used(file: PathLike, X: DataFrame) -> None:
    p = Path(file)
    with p.open("wb") as outf:
        dill.dump(X, outf)

def load_model(file: PathLike) -> ModelAPI:
    p = Path(file)
    with p.open("rb") as inf:
        model = dill.load(inf)
    return model

def save_model(file: PathLike, model: ModelAPI) -> None:
    p = Path(file)
    with p.open("wb") as outf:
        dill.dump(model, outf)

def load_state(file: PathLike) -> Tuple[Dict, DataFrame, ModelAPI]:
    p = Path(file)
    with p.open("rb") as inf:
        (rules, X, model) = dill.load(inf)
    return (rules, X, model)

def save_state(file: PathLike, rules: Dict, X: DataFrame, model: ModelAPI) -> None:
    p = Path(file)
    with p.open("wb") as outf:
        dill.dump((rules, X, model), outf)
