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
    calculate_if_subgroup_costs_cumulative,
)


def auto_budget_calculation(
    rules_with_cumulative: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    cor_thres: float,
    percentiles: List[float],
    ignore_inf: bool = True,
) -> List[float]:
    all_minchanges_to_thres = []
    for ifc, all_thens in rules_with_cumulative.items():
        for sg, (cov, thens) in all_thens.items():
            all_minchanges_to_thres.append(
                if_group_cost_min_change_correctness_cumulative_threshold(
                    ifc, thens, cor_thres
                )
            )

    vals = np.array(all_minchanges_to_thres)
    if ignore_inf:
        vals = vals[vals != np.inf]
    return np.unique(np.quantile(vals, percentiles)).tolist()


def make_table(
    rules_with_both_corrs: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    sensitive_attribute_vals: List[str],
    effectiveness_thresholds: List[float],
    cost_budgets: List[float],
    c_infty_coeff: float = 2.0,
    params: ParameterProxy = ParameterProxy(),
) -> DataFrame:
    rows = []
    for ifclause, all_thens in rules_with_both_corrs.items():
        thens_with_atomic = {
            sg: (cov, [(then, atomic_cor) for then, atomic_cor, _cum_cor in thens])
            for sg, (cov, thens) in all_thens.items()
        }
        thens_with_cumulative_and_costs = {
            sg: (
                cov,
                [
                    (then, cum_cor, float(featureChangePred(ifclause, then, params)))
                    for then, _atomic_cor, cum_cor in thens
                ],
            )
            for sg, (cov, thens) in all_thens.items()
        }

        weighted_averages = calculate_if_subgroup_costs(
            ifclause,
            thens_with_atomic,
            partial(if_group_cost_mean_with_correctness, params=params),
        )
        mincostabovethreshold = tuple(
            calculate_if_subgroup_costs(
                ifclause,
                thens_with_atomic,
                partial(
                    if_group_cost_min_change_correctness_threshold,
                    cor_thres=th,
                    params=params,
                ),
            )
            for th in effectiveness_thresholds
        )
        numberabovethreshold = tuple(
            calculate_if_subgroup_costs(
                ifclause,
                thens_with_atomic,
                partial(
                    if_group_cost_recoursescount_correctness_threshold,
                    cor_thres=th,
                    params=params,
                ),
            )
            for th in effectiveness_thresholds
        )

        total_effs = calculate_if_subgroup_costs_cumulative(
            ifclause, thens_with_cumulative_and_costs, if_group_total_correctness
        )
        max_effs_within_budget = tuple(
            calculate_if_subgroup_costs_cumulative(
                ifclause,
                thens_with_cumulative_and_costs,
                partial(if_group_cost_change_cumulative_threshold, cost_thres=th),
            )
            for th in cost_budgets
        )
        costs_of_effectiveness = tuple(
            calculate_if_subgroup_costs_cumulative(
                ifclause,
                thens_with_cumulative_and_costs,
                partial(
                    if_group_cost_min_change_correctness_cumulative_threshold,
                    cor_thres=th,
                ),
            )
            for th in effectiveness_thresholds
        )

        correctness_cap = {
            ifclause: max(
                corr
                for _sg, (_cov, thens) in thens_with_cumulative_and_costs.items()
                for _then, corr, _cost in thens
            )
        }
        mean_recourse_costs_cinf = calculate_if_subgroup_costs_cumulative(
            ifclause,
            thens_with_cumulative_and_costs,
            partial(
                if_group_average_recourse_cost_cinf,
                correctness_caps=correctness_cap,
                c_infty_coeff=c_infty_coeff,
            ),
        )
        mean_recourse_costs_conditional = calculate_if_subgroup_costs_cumulative(
            ifclause,
            thens_with_cumulative_and_costs,
            if_group_average_recourse_cost_conditional,
        )

        ecds = pd.DataFrame(
            {
                sg: np.array([cor for _t, cor, _cost in thens])
                for sg, (cov, thens) in thens_with_cumulative_and_costs.items()
            }
        )
        ecds_max = ecds.max(axis=1)
        ecds_min = ecds.min(axis=1)
        eff_cost_tradeoff_KS = (ecds_max - ecds_min).max()
        eff_cost_tradeoff_KS_idx = (ecds_max - ecds_min).argmax()
        unfair_row = ecds.iloc[eff_cost_tradeoff_KS_idx]
        eff_cost_tradeoff_biased = unfair_row.index[unfair_row.argmin()]

        row = (
            (weighted_averages,)
            + mincostabovethreshold
            + numberabovethreshold
            + (total_effs,)
            + max_effs_within_budget
            + costs_of_effectiveness
            + (mean_recourse_costs_cinf, mean_recourse_costs_conditional)
        )
        rows.append(
            (ifclause,)
            + tuple([d[sens] for d in row for sens in sensitive_attribute_vals])
            + (eff_cost_tradeoff_KS, eff_cost_tradeoff_biased)
        )

    cols = (
        ["weighted-average"]
        + [
            ("Equal Cost of Effectiveness(Macro)", th)
            for th in effectiveness_thresholds
        ]
        + [("Equal Choice for Recourse", th) for th in effectiveness_thresholds]
        + ["Equal Effectiveness"]
        + [("Equal Effectiveness within Budget", th) for th in cost_budgets]
        + [
            ("Equal Cost of Effectiveness(Micro)", th)
            for th in effectiveness_thresholds
        ]
        + ["mean-cost-cinf", "Equal(Conditional Mean Recourse)"]
    )
    cols = pd.MultiIndex.from_product([cols, sensitive_attribute_vals])
    cols = pd.MultiIndex.from_tuples(
        [("subgroup", "subgroup")]
        + list(cols)
        + [
            ("Fair Effectiveness-Cost Trade-Off", "value"),
            ("Fair Effectiveness-Cost Trade-Off", "bias"),
        ]
    )

    return pd.DataFrame(rows, columns=cols)


def get_diff_table(
    df: DataFrame,
    sensitive_attribute_vals: List[str] = ["Male", "Female"],
    with_abs: bool = True
) -> DataFrame:
    z = df.copy()
    z = z.drop(columns=[("subgroup", "subgroup")])
    diff = pd.DataFrame()
    x = z["Fair Effectiveness-Cost Trade-Off"]
    z = z.drop(columns=["Fair Effectiveness-Cost Trade-Off"])
    for col in z.columns.get_level_values(0):
        if with_abs:
            diff[col] = abs(
                z[col, sensitive_attribute_vals[0]]
                - z[col, sensitive_attribute_vals[1]]
            )
        else:
            diff[col] = (
                z[col, sensitive_attribute_vals[0]]
                - z[col, sensitive_attribute_vals[1]]
            )

    diff[("Fair Effectiveness-Cost Trade-Off", "value")] = x["value"]
    diff[("Fair Effectiveness-Cost Trade-Off", "bias")] = x["bias"]
    diff["subgroup"] = df["subgroup", "subgroup"]
    first = diff.pop("subgroup")
    diff.insert(0, "subgroup", first)
    diff = diff.fillna(0)

    return diff


def get_comb_df(
    df: DataFrame,
    ranked: DataFrame,
    diff: DataFrame,
    rev_bias_metrics: List[str] = ["Equal Effectiveness", "Equal Effectiveness within Budget"],
    sensitive_attribute_vals: List[str] = ["Male", "Female"],
):
    diff_real_val = get_diff_table(df, sensitive_attribute_vals, with_abs=False)
    diff_real_val = diff_real_val.set_index("subgroup")
    diff_drop = diff.drop(columns=[("Fair Effectiveness-Cost Trade-Off", "bias")])
    first_lvl_subgroups = ranked.index.unique()
    second_lvl_metrics = ranked.columns
    third_level_vals = ["rank", "score"]
    map_ranks = {}
    map_scores = {}
    map_bias = {}

    for i, row in ranked.iterrows():
        for metric, c in row.items():
            map_ranks[(i, metric)] = c

    for i, row in diff_drop.iterrows():
        for metric, c in row.items():
            map_scores[(i, metric)] = c

    for i, row in diff_real_val.iterrows():
        for metric, c in row.items():
            id_tmp = metric
            val = None
            if metric[1] == "value":
                map_scores[(i, metric)] = c
                continue
            if metric[1] == "bias":
                val = c
                id_tmp = (metric[0], "value")
            else:
                val = (
                    sensitive_attribute_vals[0]
                    if c > 0
                    else sensitive_attribute_vals[1]
                )
                if c == 0:
                    val = "Fair"
                elif metric[0] in rev_bias_metrics:
                    val = (
                        sensitive_attribute_vals[0]
                        if val == sensitive_attribute_vals[1]
                        else sensitive_attribute_vals[1]
                    )
            map_bias[(i, id_tmp)] = val

    comb_columns = pd.MultiIndex.from_product(
        [second_lvl_metrics, ["rank", "score", "bias against"]]
    )
    comb_data = []

    for sg in first_lvl_subgroups:
        row_data = []
        for mtr in second_lvl_metrics:
            rank_value = map_ranks[(sg, mtr)]
            score_value = map_scores[(sg, mtr)]
            bias_value = map_bias[(sg, mtr)]
            row_data.extend([rank_value, score_value, bias_value])

        comb_data.append(row_data)

    comb_df = pd.DataFrame(comb_data, columns=comb_columns, index=first_lvl_subgroups)
    comb_df = comb_df.rename(
        columns={
            (
                "Fair Effectiveness-Cost Trade-Off",
                "value",
            ): "Fair Effectiveness-Cost Trade-Off"
        }
    )

    return comb_df


def get_analysis_df(comb_df, sensitive_attribute_vals=["Male", "Female"]):
    metric_rank_one = {}
    metric_male_cnt = {}
    metric_female_cnt = {}

    for sg, row in comb_df.iterrows():
        for metric_and_type, value in row.items():
            metric = metric_and_type[0]
            type_ = metric_and_type[1]

            if type_ == "rank":
                if value == 1:
                    current_value = 0
                    if metric in metric_rank_one:
                        current_value = metric_rank_one[metric]
                    metric_rank_one[metric] = current_value + 1
            elif type_ == "bias against":
                value_ = value.replace(" ", "")

                if value_ == sensitive_attribute_vals[0]:
                    current_value = 0
                    if metric in metric_male_cnt:
                        current_value = metric_male_cnt[metric]
                    metric_male_cnt[metric] = current_value + 1
                elif value_ == sensitive_attribute_vals[1]:
                    current_value = 0
                    if metric in metric_female_cnt:
                        current_value = metric_female_cnt[metric]
                    metric_female_cnt[metric] = current_value + 1
    data = []
    for metric in metric_rank_one:
        data.append(
            {
                "Metric": metric,
                "Rank = 1 Count": metric_rank_one[metric]
                if metric in metric_rank_one
                else 0,
                f"{sensitive_attribute_vals[0]} bias against Count": metric_male_cnt[
                    metric
                ]
                if metric in metric_male_cnt
                else 0,
                f"{sensitive_attribute_vals[1]} bias against Count": metric_female_cnt[
                    metric
                ]
                if metric in metric_female_cnt
                else 0,
            }
        )

    data_df = pd.DataFrame(data).set_index("Metric")
    total_counts = data_df.sum()

    total_row = pd.DataFrame(total_counts).T
    total_row.index = ["Total Count"]
    data_df = data_df.append(total_row)

    return data_df


def filter_on_fair_unfair(
    ranked: DataFrame,
    fair_lower_bound: int,
    unfair_lower_bound: int,
    fair_token: str,
    rank_upper: int,
) -> DataFrame:
    def elem_to_bool(x):
        if x == fair_token:
            return False
        if x < rank_upper:
            return True
        elif x >= rank_upper:
            return False
        else:
            raise NotImplementedError("This should be unreachable.", x)

    fair_unfair_indicator = ranked.applymap(elem_to_bool).apply(
        lambda row: row.sum() >= unfair_lower_bound
        and (~row).sum() >= fair_lower_bound,
        axis=1,
    )
    fair_unfair = ranked[fair_unfair_indicator]

    return fair_unfair
