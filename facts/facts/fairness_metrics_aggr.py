from typing import List, Dict, Tuple
from functools import partial

import numpy as np
import pandas as pd
from pandas import DataFrame

from .parameters import ParameterProxy
from .predicate import Predicate, featureChangePred
from .metrics import (
    if_group_cost_mean_with_correctness,
    if_group_cost_min_change_correctness_threshold,
    if_group_cost_recoursescount_correctness_threshold,
    if_group_total_correctness,
    if_group_cost_change_cumulative_threshold,
    if_group_cost_min_change_correctness_cumulative_threshold,
    if_group_average_recourse_cost_cinf,
    if_group_average_recourse_cost_conditional,
    calculate_if_subgroup_costs,
    calculate_if_subgroup_costs_cumulative
)

def auto_budget_calculation(
    rules_with_cumulative: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    cor_thres: float,
    percentiles: List[float],
    ignore_inf: bool = True
) -> List[float]:
    all_minchanges_to_thres = []
    for ifc, all_thens in rules_with_cumulative.items():
        for sg, (cov, thens) in all_thens.items():
            all_minchanges_to_thres.append(if_group_cost_min_change_correctness_cumulative_threshold(ifc, thens, cor_thres))
    
    vals = np.array(all_minchanges_to_thres)
    if ignore_inf:
        vals = vals[vals != np.inf]
    return np.quantile(vals, percentiles).tolist()

def make_table(
    rules_with_both_corrs: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    sensitive_attribute_vals: List[str],
    effectiveness_thresholds: List[float],
    cost_budgets: List[float],
    c_infty_coeff: float = 2.,
    params: ParameterProxy = ParameterProxy()
) -> DataFrame:
    rows =  []
    for ifclause, all_thens in rules_with_both_corrs.items():
        thens_with_atomic = {
            sg: (cov, [(then, atomic_cor) for then, atomic_cor, _cum_cor in thens])
            for sg, (cov, thens) in all_thens.items()
        }
        thens_with_cumulative_and_costs = {
            sg: (cov, [(then, cum_cor, float(featureChangePred(ifclause, then, params))) for then, _atomic_cor, cum_cor in thens])
            for sg, (cov, thens) in all_thens.items()
        }

        weighted_averages = calculate_if_subgroup_costs(ifclause, thens_with_atomic, partial(if_group_cost_mean_with_correctness, params=params))
        mincostabovethreshold = tuple(
            calculate_if_subgroup_costs(ifclause, thens_with_atomic, partial(if_group_cost_min_change_correctness_threshold, cor_thres=th, params=params))
            for th in effectiveness_thresholds
        )
        numberabovethreshold = tuple(
            calculate_if_subgroup_costs(ifclause, thens_with_atomic, partial(if_group_cost_recoursescount_correctness_threshold, cor_thres=th, params=params))
            for th in effectiveness_thresholds
        )

        total_effs = calculate_if_subgroup_costs_cumulative(ifclause, thens_with_cumulative_and_costs, if_group_total_correctness)
        max_effs_within_budget = tuple(
            calculate_if_subgroup_costs_cumulative(ifclause, thens_with_cumulative_and_costs, partial(if_group_cost_change_cumulative_threshold, cost_thres=th))
            for th in cost_budgets
        )
        costs_of_effectiveness = tuple(
            calculate_if_subgroup_costs_cumulative(ifclause, thens_with_cumulative_and_costs, partial(if_group_cost_min_change_correctness_cumulative_threshold, cor_thres=th))
            for th in effectiveness_thresholds
        )

        correctness_cap = {ifclause: max(corr for _sg, (_cov, thens) in thens_with_cumulative_and_costs.items() for _then, corr, _cost in thens)}
        mean_recourse_costs_cinf = calculate_if_subgroup_costs_cumulative(
            ifclause,
            thens_with_cumulative_and_costs,
            partial(if_group_average_recourse_cost_cinf, correctness_caps=correctness_cap, c_infty_coeff=c_infty_coeff)
        )
        mean_recourse_costs_conditional = calculate_if_subgroup_costs_cumulative(
            ifclause,
            thens_with_cumulative_and_costs,
            if_group_average_recourse_cost_conditional
        )

        ecds = pd.DataFrame({
            sg: np.array([cor for _t, cor, _cost in thens])
            for sg, (cov, thens) in thens_with_cumulative_and_costs.items()
        })
        ecds_max = ecds.max(axis=1)
        ecds_min = ecds.min(axis=1)
        eff_cost_tradeoff_KS = (ecds_max - ecds_min).max()
        eff_cost_tradeoff_KS_idx = (ecds_max - ecds_min).argmax()
        unfair_row = ecds.iloc[eff_cost_tradeoff_KS_idx]
        eff_cost_tradeoff_biased = unfair_row.index[unfair_row.argmin()]

        row = (weighted_averages,) + \
            mincostabovethreshold + \
            numberabovethreshold + \
            (total_effs,) + \
            max_effs_within_budget + \
            costs_of_effectiveness + \
            (mean_recourse_costs_cinf, mean_recourse_costs_conditional)
        rows.append((ifclause,) + tuple(v for d in row for v in d.values()) + (eff_cost_tradeoff_KS, eff_cost_tradeoff_biased))
    
    cols = ["weighted-average"] \
        + [("mincost-above-th", th) for th in effectiveness_thresholds] \
        + [("number-above-th", th) for th in effectiveness_thresholds] \
        + ["total-effectiveness"] \
        + [("eff-within-budget", th) for th in cost_budgets] \
        + [("cost-of-effectiveness", th) for th in effectiveness_thresholds] \
        + ["mean-cost-cinf", "mean-cost-conditional"]
    cols = pd.MultiIndex.from_product([cols, sensitive_attribute_vals])
    cols = pd.MultiIndex.from_tuples([("subgroup", "subgroup")] + list(cols) + [("KStest", "value"), ("KStest", "bias")])

    return pd.DataFrame(rows, columns=cols)

def get_diff_table(
        df,
        sensitive_attribute_vals=["Male", "Female"]    
    ):
    z = df.copy()
    z = z.drop(columns=[('subgroup','subgroup')])
    diff = pd.DataFrame()

    for col in z.columns.get_level_values(0):
        diff[col] = abs(z[col,sensitive_attribute_vals[0]] - z[col,sensitive_attribute_vals[1]])

    diff['subgroup'] = df['subgroup','subgroup']
    first = diff.pop('subgroup')
    diff.insert(0,'subgroup',first)
    diff = diff.fillna(0)

    return diff
