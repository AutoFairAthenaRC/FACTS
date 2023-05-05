from typing import Dict, List, Tuple
import dill
from pathlib import Path
from os import PathLike

from gfacts import Predicate

def load_rules_by_if(file: PathLike) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    p = Path(file)
    with p.open("rb") as inf:
        rules_by_if = dill.load(inf)
    return rules_by_if

def save_rules_by_if(file: PathLike, rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]) -> None:
    p = Path(file)
    with p.open("wb") as outf:
        dill.dump(rules, outf)
