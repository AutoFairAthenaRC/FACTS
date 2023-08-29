from typing import List, Dict, Tuple, Any, Optional, Protocol
from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from colorama import Fore, Style

from .models import ModelAPI
from .metrics import incorrectRecoursesIfThen
from .predicate import Predicate, recIsValid

ASSUME_ZERO = 10**(-7)

@dataclass
class ConcreteReport:
    printables: List[str | Dict[str, Tuple[List[float], List[float]]]]

    def append_string(self, s: str) -> None:
        self.printables.append(s)
    
    def append_ecd_plots(self, p: Dict[str, Tuple[List[float], List[float]]]) -> None:
        self.printables.append(p)
    
    def print_console(self) -> None:
        for printable in self.printables:
            if isinstance(printable, str):
                print(printable)
            elif isinstance(printable, dict):
                raise ValueError("Cannot print plots to console")
            else:
                raise ValueError(f"ConcreteReport cannot handle objects of type {type(printable)}")
    
    def print_notebook(self) -> None:
        for printable in self.printables:
            if isinstance(printable, str):
                print(printable)
            elif isinstance(printable, dict):
                cost_correctness_plot(printable)
                plt.plot()
            else:
                raise ValueError(f"ConcreteReport cannot handle objects of type {type(printable)}")


class Report(Protocol):
    @abstractmethod
    def append_string(self, s: str) -> None:
        pass

    @abstractmethod
    def append_ecd_plots(self, p: Dict[str, Tuple[List[float], List[float]]]) -> None:
        pass

def coverage_statistics_str(
    subgroup: str,
    coverage: float,
    pop_sizes: Optional[Dict[str, int]]
) -> str:
    ret = []
    ret.append(f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{coverage:.2%}{Fore.RESET} covered")
    if pop_sizes is not None:
        if subgroup in pop_sizes:
            ret.append(f" out of {pop_sizes[subgroup]}")
        else:
            ret.append(" (protected subgroup population size not given)")
    ret.append("\n")
    return "".join(ret)

def build_recourse_report(
    report_object: Report,
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    subgroup_costs: Optional[Dict[Predicate, Dict[str, float]]] = None,
    show_subgroup_costs: bool = False,
    show_then_costs: bool = False,
    show_cumulative_plots: bool = False,
    show_bias: Optional[str] = None,
    correctness_metric : bool = False,
    metric_name : str = 'Equal Effectiveness'
) -> None:
    if len(rules) == 0:
        report_object.append_string(f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}\n")
    
    for ifclause, all_thens in rules.items():
        if subgroup_costs is not None and show_bias is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            if biased_subgroup != show_bias:
                continue
        
        report_object.append_string(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:\n")
        for subgroup, (cov, thens) in all_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            cov_stats_str = coverage_statistics_str(subgroup, cov, population_sizes)
            report_object.append_string(cov_stats_str)

            # print each available recourse together with the respective correctness
            if thens == []:
                report_object.append_string(f"\t\t{Fore.RED}No recourses for this subgroup!{Fore.RESET}")
            for then, correctness, cost in thens:
                _, thenstr = ifthen2str(ifclause=ifclause, thenclause=then)

                # abs() used to get rid of -0.0
                assert correctness >= -ASSUME_ZERO
                cor_str = Fore.GREEN + f"{abs(correctness):.2%}" + Fore.RESET
                report_object.append_string(f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}")

                if show_then_costs:
                    report_object.append_string(f" and counterfactual cost = {round(cost,2)}")
                report_object.append_string(".\n")

            if subgroup_costs is not None and show_subgroup_costs:
                cost_of_current_subgroup = subgroup_costs[ifclause][subgroup]
                if f"{cost_of_current_subgroup:.2f}" == "-0.00":
                    cost_of_current_subgroup = 0
                report_object.append_string(f"\t\t{Style.BRIGHT}Aggregate cost{Style.RESET_ALL} of the above recourses = {Fore.MAGENTA}{cost_of_current_subgroup:.2f}{Fore.RESET}\n")
        
        # TODO: show bias message in (much) larger font size.
        if subgroup_costs is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            if correctness_metric == False:
                max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
                biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            else: 
                max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
                biased_subgroup, max_cost = min(curr_subgroup_costs.items(), key=lambda p: p[1])
            if max_intergroup_cost_diff > 0:
                report_object.append_string(f"\t{Fore.MAGENTA}Bias against {biased_subgroup} due to {metric_name}. Unfairness score = {round(max_intergroup_cost_diff,3)}.{Fore.RESET}\n")
            else:
                report_object.append_string(f"\t{Fore.MAGENTA}No bias!{Fore.RESET}\n")

        if show_cumulative_plots:
            report_object.append_string(f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}\n")
            cost_cors: Dict[str, Tuple[List[float], List[float]]] = {}
            for sg, (_cov, thens) in rules[ifclause].items():
                cost_cors[sg] = ([cost for _, _, cost in thens], [cor for _, cor, _ in thens])
            report_object.append_ecd_plots(cost_cors)


def print_recourse_report(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    subgroup_costs: Optional[Dict[Predicate, Dict[str, float]]] = None,
    show_subgroup_costs: bool = False,
    show_then_costs: bool = False,
    show_cumulative_plots: bool = False,
    show_bias: Optional[str] = None,
    correctness_metric : bool = False,
    metric_name : str = 'Equal Effectiveness'
    ) -> None:
    if len(rules) == 0:
        print(f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}")
    
    for ifclause, sg_thens in rules.items():
        if subgroup_costs is not None and show_bias is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            if biased_subgroup != show_bias:
                continue
        
        print(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            print(f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered", end="")
            if population_sizes is not None:
                if subgroup in population_sizes:
                    print(f" out of {population_sizes[subgroup]}", end="")
                else:
                    print(" (protected subgroup population size not given)", end="")
            print()

            # print each available recourse together with the respective correctness
            if thens == []:
                print(f"\t\t{Fore.RED}No recourses for this subgroup!{Fore.RESET}")
            for then, correctness, cost in thens:
                _, thenstr = ifthen2str(ifclause=ifclause, thenclause=then)

                # abs() used to get rid of -0.0
                assert correctness >= -ASSUME_ZERO
                cor_str = Fore.GREEN + f"{abs(correctness):.2%}" + Fore.RESET
                print(f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}", end="")

                if show_then_costs:
                    print(f" and counterfactual cost = {round(cost,2)}", end="")
                print(".")

            if subgroup_costs is not None and show_subgroup_costs:
                cost_of_current_subgroup = subgroup_costs[ifclause][subgroup]
                if f"{cost_of_current_subgroup:.2f}" == "-0.00":
                    cost_of_current_subgroup = 0
                print(f"\t\t{Style.BRIGHT}Aggregate cost{Style.RESET_ALL} of the above recourses = {Fore.MAGENTA}{cost_of_current_subgroup:.2f}{Fore.RESET}")
        
        # TODO: show bias message in (much) larger font size.
        if subgroup_costs is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            if correctness_metric == False:
                max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
                biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            else: 
                max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
                biased_subgroup, max_cost = min(curr_subgroup_costs.items(), key=lambda p: p[1])
            if max_intergroup_cost_diff > 0:
                print(f"\t{Fore.MAGENTA}Bias against {biased_subgroup} due to {metric_name}. Unfairness score = {round(max_intergroup_cost_diff,3)}.{Fore.RESET}")
            else:
                print(f"\t{Fore.MAGENTA}No bias!{Fore.RESET}")

        if show_cumulative_plots:
            print(f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}")
            cost_cors = {}
            for sg, (_cov, thens) in rules[ifclause].items():
                cost_cors[sg] = ([cost for _, _, cost in thens], [cor for _, cor, _ in thens])
            cost_correctness_plot(cost_cors)
            plt.show()

def print_recourse_report_KStest_cumulative(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    unfairness: Optional[Dict[Predicate, float]] = None,
    show_then_costs: bool = False,
    show_cumulative_plots: bool = False,
    metric_name = 'Fair Effectiveness-Cost Trade-Off'
) -> None:
    if len(rules) == 0:
        print(f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}")
    
    for ifclause, sg_thens in rules.items():
        print(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            print(f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered", end="")
            if population_sizes is not None:
                if subgroup in population_sizes:
                    print(f" out of {population_sizes[subgroup]}", end="")
                else:
                    print(" (protected subgroup population size not given)", end="")
            print()

            # print each available recourse together with the respective correctness
            if thens == []:
                print(f"\t\t{Fore.RED}No recourses for this subgroup!{Fore.RESET}")
            for then, correctness, cost in thens:
                _, thenstr = ifthen2str(ifclause=ifclause, thenclause=then)

                # abs() used to get rid of -0.0
                assert correctness >= -ASSUME_ZERO
                cor_str = Fore.GREEN + f"{abs(correctness):.2%}" + Fore.RESET
                print(f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}", end="")

                if show_then_costs:
                    print(f" and counterfactual cost = {round(cost,2)}", end="")
                print(".")

            
        if unfairness is not None:
                curr_subgroup_costs = unfairness[ifclause]
                print(f"\t{Fore.MAGENTA} Unfairness based on the {metric_name} = {round(curr_subgroup_costs,2)}.{Fore.RESET}")
    

        if show_cumulative_plots:
            print(f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}")
            cost_cors = {}
            for sg, (_cov, thens) in rules[ifclause].items():
                cost_cors[sg] = ([cost for _, _, cost in thens], [cor for _, cor, _ in thens])
            cost_correctness_plot(cost_cors)
            plt.show()

def ifthen2str(
    ifclause: Predicate,
    thenclause: Predicate,
    show_same_feats: bool = False,
    same_col: str = "default",
    different_col: str = Fore.RED
) -> Tuple[str, str]:
    # if not recIsValid(ifclause, thenclause,drop_infeasible):
    #     raise ValueError("If and then clauses should be compatible.")
    
    ifstr = []
    thenstr = []
    first_rep = True
    thendict = thenclause.to_dict()
    for f, v in ifclause.to_dict().items():
        if not show_same_feats and v == thendict[f]:
            continue

        if first_rep:
            first_rep = False
        else:
            ifstr.append(", ")
            thenstr.append(", ")
        
        if v == thendict[f]:
            if same_col != "default":
                ifstr.append(same_col + f"{f} = {v}" + Fore.RESET)
                thenstr.append(same_col + f"{f} = {v}" + Fore.RESET)
            else:
                ifstr.append(f"{f} = {v}")
                thenstr.append(f"{f} = {v}")
        else:
            ifstr.append(different_col + f"{f} = {v}" + Fore.RESET)
            thenstr.append(different_col + f"{f} = {thendict[f]}" + Fore.RESET)
    
    return "".join(ifstr), "".join(thenstr)




def cost_correctness_plot(
    costs_cors_per_subgroup: Dict[str, Tuple[List[float], List[float]]]
) -> Figure:
    subgroup_markers = {sg: (index, 0, 0) for index, sg in enumerate(costs_cors_per_subgroup.keys(), start=3)}
    fig, ax = plt.subplots()
    lines = []
    labels = []
    for sg, (costs, correctnesses) in costs_cors_per_subgroup.items():
        line, = ax.step(
            costs,
            correctnesses,
            where="post",
            marker=subgroup_markers[sg],
            label=sg,
            alpha=0.7
        )
        lines.append(line)
        labels.append(sg)
    ax.set_xlabel("Cost of change")
    ax.set_ylabel("Correctness percentage")
    ax.legend(lines, labels)
    return fig