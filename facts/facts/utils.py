from typing import Dict, List, Tuple, Optional
import dill
from pathlib import Path
from os import PathLike
from pandas import DataFrame

from .predicate import Predicate
from .models import ModelAPI

def load_object(file: PathLike) -> object:
    p = Path(file)
    with p.open("rb") as inf:
        ret = dill.load(inf)
    return ret

def save_object(file: PathLike, o: object) -> None:
    p = Path(file)
    with p.open("wb") as outf:
        dill.dump(o, outf)

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

def rules_to_latex(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    bias_against: Dict[Predicate, Tuple[str, str, float]] = dict(),
    subgroup_names: Optional[List[Tuple[str]]] = None,
    indent_str: str = "    "
) -> str:
    ret = r"""
\begin{figure}[h]
\centering
\begin{minipage}{1\linewidth}
"""
    for i, (ifc, all_thens) in enumerate(rules.items()):
        ret += r"\begin{lstlisting}[style = base,escapechar=+]" + "\n"

        if subgroup_names is not None:
            ret += f"+\\textbf{{Subgroup {subgroup_names[i]}}}+" + "\n"
        
        ret += f"If {ifc}:" + "\n"
        for sg, (cov, thens) in all_thens.items():
            ret += f"{indent_str}Protected Subgroup = `{sg}', !{cov:.2%}! covered" + "\n"

            if thens == []:
                ret += f"{indent_str * 2}\t\t@No recourses for this subgroup.@" + "\n"
            for then_with_extras in thens:
                then = then_with_extras[0]
                corr = then_with_extras[1]
                ret += f"{indent_str * 2}Make @{then}@ with effectiveness &{corr:.2%}&" + "\n"
        
        biased_prot, metric, bias = bias_against[ifc]
        ret += f"{indent_str}_Bias against `{biased_prot}' due to {metric}. Unfairness score = {bias}._" + "\n"

        ret += r"\end{lstlisting}" + "\n"

    ret += r"""
\caption{}
\label{}
\end{minipage}
\end{figure}
"""
    return ret

def table_to_latex(
    comb_df: DataFrame,
    subgroups: List[Predicate],
    metric_names: List[Tuple[str, str]]
) -> str:
    ret = r"""
\begin{table}[ht]
\caption{}
  \label{}
  \centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
"""

    ret += r"\multicolumn{1}{r}{} "
    for i in range(len(subgroups)):
        ret += r"& \multicolumn{3}{c}{\textbf{Subgroup " + f"{i + 1}" + "}} "
    ret += r" \\ \cmidrule(r){2-" + f"{1 + 3 * len(subgroups)}" + "}"
    ret += "\n\n"

    ret += r"\multicolumn{1}{c}{} "
    for i in range(len(subgroups)):
        ret += r"& \multicolumn{1}{c}{rank} & \multicolumn{1}{c}{bias against} & \multicolumn{1}{c}{unfairness score} "
    ret += r"\\ \midrule" + "\n"

    for col_df_name, col_name in metric_names:
        ret += col_name + " "
        for i, sg in enumerate(subgroups):
            row = comb_df.loc[sg]
            rank = row[col_df_name]["rank"]
            if rank == 1:
                rank = r"\textbf{\textcolor{red}{1}}"
            score = row[col_df_name]["score"]
            bias_against = row[col_df_name]["bias against"]
            ret += f"& {rank} & {bias_against} & {round(score,3)} "
        ret += r"\\" + "\n"

    ret += r"""

\bottomrule
\end{tabular}%
}
\end{table}
"""
    return ret
