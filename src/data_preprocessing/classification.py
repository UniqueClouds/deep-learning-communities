from itertools import combinations
import pandas as pd
import numpy as np
import random
import pymannkendall as mk
from sklearn.metrics import cohen_kappa_score
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns

coef = 0.8


def calculate_ratios(
    exisited_core_set,
    current_core_set,
    last_core_set,
    current_peripheral_set,
    current_active_set,
    current_dev_set,
    committer_after_set,
) -> (float, float, float, float, float, float):
    """
    A helper function. Calculates a number of ratios.

    Parameters:
        existed_core_set (set): The set of users who have ever held the core role.
        current_core_set (set): The current core set of users.
        last_core_set (set): The previous core set of users.
        current_peripheral_set (set): The current peripheral set of users.
        current_active_set (set): The current active set of users.
        current_dev_set (set): The current developer set of users.
        nolonger_commit_set (set): The set of users who have not committed since the current time window.

    Returns:
        tuple: A tuple containing the following ratios:
            - ratio_peri_core (float): The ratio of peripheral users to core users.
            - ratio_active_dev (float): The ratio of active users to developers.
            - retention_rate (float): The ratio of users retained from the previous core set.
            - dropout_rate (float): The ratio of users dropped from the previous core set.
            - core2peripheral_rate (float): The ratio of users flowing from the previous core set to the current peripheral set.
            - inflow_rate (float): The ratio of users flowing to the current core set without having held the core role before.
    """
    retention_rate = len(current_core_set & last_core_set) / len(last_core_set)
    dropout_rate = len(last_core_set - committer_after_set) / len(last_core_set)
    core2peripheral_rate = len(current_peripheral_set & last_core_set) / len(
        last_core_set
    )
    inflow_rate = len(current_core_set - exisited_core_set) / len(current_core_set)
    ratio_peri_core = len(current_peripheral_set) / len(current_core_set)
    ratio_active_dev = len(current_active_set) / len(current_dev_set)
    return [
        ratio_peri_core,
        ratio_active_dev,
        retention_rate,
        dropout_rate,
        core2peripheral_rate,
        inflow_rate,
    ]


def role_transition_capture(
    community_data: pd.DataFrame, classification_type: str
) -> pd.DataFrame:
    """
    Capture the role transition data during time windows
    This may take a few minutes to run.
    Parameters:
    community_data (pandas.DataFrame): A dataframe with column names [login, commit, LOC, issue_count, central_degree, version] or [login, commit, LOC, issue_count, central_degree, month].
    classification_type (str): A string indicating the type of classification (e.g., "version" or "month").

    Returns:
    pandas.DataFrame: A dataframe with column names [ratio_peri_core, ratio_active_dev, retention_rate, dropout_rate, core2peripheral_rate, inflow_rate, version] or [ratio_peri_core, ratio_active_dev, retention_rate, dropout_rate, core2peripheral_rate, inflow_rate, month].
    """
    assert classification_type in ["version", "month"]
    # init columns
    columns = [
        "ratio_peri_core",
        "ratio_active_dev",
        "retention_rate",
        "dropout_rate",
        "core2peripheral_rate",
        "inflow_rate",
    ]
    # add version or month column
    if classification_type == "version":
        columns.append("version")
        loop_list = sorted(
            (community_data["version"].unique()),
            key=lambda s: list(map(int, s.split("."))),
            reverse=False,
        )
        # loop_list = sorted(community_data["version"].unique())
        loop = "version"
    else:
        columns.append("month")
        loop_list = sorted(community_data["month"].unique())
        loop = "month"
    role_transition_result = pd.DataFrame(columns=columns)
    # init core set
    exisited_core_set = set()
    last_core_set = set()
    # loop to calculate ratio
    for i, loop_item in enumerate(loop_list):
        (
            _,
            cur_core_set,
            cur_peripheral_set,
            cur_dev_set,
            cur_active_set,
        ) = role_classification(community_data[community_data[loop] == loop_item])
        if i != 0:
            # calculate committer_after_set
            committer_after_set = set()
            for after_loop_item in loop_list[i:]:
                committer_set = set()
                contribution_table = community_data[
                    community_data[loop] == after_loop_item
                ]
                for _, row in contribution_table.iterrows():
                    if row["commit_count"] > 0:
                        committer_set.add(row["login"])
                committer_after_set |= committer_set
            ratios = calculate_ratios(
                exisited_core_set,
                cur_core_set,
                last_core_set,
                cur_peripheral_set,
                cur_active_set,
                cur_dev_set,
                committer_after_set,
            )
            role_transition_result = pd.concat(
                [
                    role_transition_result,
                    pd.DataFrame(
                        {
                            "ratio_peri_core": ratios[0],
                            "ratio_active_dev": ratios[1],
                            "retention_rate": ratios[2],
                            "dropout_rate": ratios[3],
                            "core2peripheral_rate": ratios[4],
                            "inflow_rate": ratios[5],
                            loop: loop_item,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
        else:
            role_transition_result = pd.concat(
                [
                    role_transition_result,
                    pd.DataFrame(
                        {
                            "ratio_peri_core": len(cur_peripheral_set) / len(cur_core_set),
                            "ratio_active_dev": len(cur_active_set) / len(cur_dev_set),
                            "retention_rate": 0,
                            "dropout_rate": 0,
                            "core2peripheral_rate": 0,
                            "inflow_rate": 0,
                            loop: loop_item,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
        exisited_core_set |= cur_core_set
        last_core_set = cur_core_set
    print(role_transition_result)
    return role_transition_result


def role_classification(
    contribution_table: pd.DataFrame,
) -> (pd.DataFrame, set(), set(), set(), set()):
    """
    Classify users into core, peripheral, developer, and active user from the contribution table.

    Parameters:
    contribution_table (pd.DataFrame): Dataframe with column names [login, commit_count, LOC, issue_count, central_degree].

    Returns:
    pd.DataFrame: DataFrame with columns [metric, core_set, peripheral_set, dev_set, active_set],
    set: Set of core users.
    set: Set of peripheral users.
    set: Set of developer users.
    set: Set of active users.

    """
    dev_set, core_set, peripheral_set, active_set = set(), set(), set(), set()
    metrics = ["commit_count", "LOC", "issue_count", "central_degree"]
    result_each_metric = []
    # identify dev_set and active_set
    for _, row in contribution_table.iterrows():
        if row["commit_count"] > 0:
            dev_set.add(row["login"])
        elif row["issue_count"] or row["central_degree"] > 0:
            active_set.add(row["login"])
    contribution_table = contribution_table[contribution_table["commit_count"] > 0]
    for metric in metrics:
        contribution_table_sorted = contribution_table.sort_values(
            by=metric, ascending=False
        )
        cnt = 0
        core_set_single_metric = set()
        peripheral_set_single_metric = set()
        total_val = sum(contribution_table_sorted[metric])
        for _, row in contribution_table_sorted.iterrows():
            login = row["login"]
            val = row[metric]
            if cnt < total_val * coef:
                core_set_single_metric.add(login)
            cnt += val
        peripheral_set_single_metric = dev_set - core_set_single_metric
        result_each_metric.append(
            (metric, core_set_single_metric, peripheral_set_single_metric, active_set)
        )
        core_set |= core_set_single_metric
    peripheral_set = dev_set - core_set
    result_each_metric_df = pd.DataFrame(
        result_each_metric,
        columns=["metric", "core_set", "peripheral_set", "active_set"],
    )
    return result_each_metric_df, core_set, peripheral_set, dev_set, active_set


def statistical_analysis_and_mann_kendall_test(
    role_transition_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform statistical analysis and Mann-Kendall test on the role transition data.

    Parameters:
    role_transition_data (pandas.DataFrame): A dataframe with column names [ratio_peri_core, ratio_active_dev, retention_rate, dropout_rate, core2peripheral_rate, inflow_rate, version/month].

    Returns:
    trend_data (pandas.DataFrame): a dataframe with column names [ratio_peri_core, ratio_active_dev, retention_rate, dropout_rate, core2peripheral_rate, inflow_rate, version, trend]
    """
    analysis_result = pd.DataFrame(
        columns=[
            "metric-rate",
            "min_value",
            "max_value",
            "mean_value",
            "std",
            "trend",
            "p-value",
        ]
    )
    for metric_rate in [
        "retention_rate",
        "core2peripheral_rate",
        "dropout_rate",
        "inflow_rate",
        "ratio_peri_core",
        "ratio_active_dev"
    ]:
        min_value = role_transition_data[metric_rate].min()
        max_value = role_transition_data[metric_rate].max()
        mean_value = role_transition_data[metric_rate].mean()
        std = role_transition_data[metric_rate].std()
        result = mk.original_test(role_transition_data[metric_rate])
        analysis_result = pd.concat(
            [
                analysis_result,
                pd.DataFrame(
                    {
                        "metric-rate": metric_rate,
                        "min_value": min_value,
                        "max_value": max_value,
                        "mean_value": mean_value,
                        "std": std,
                        "trend": result.trend,
                        "p-value": result.p,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )
    print(analysis_result)
    return analysis_result

def draw_kappa_matrix(df, save_path, show=False):
    mask = [[0 for i in range(4)] for j in range(4)]
    for i in range(4):
        for j in range(4):
            if i > j:
                mask[i][j] = True
            else:
                mask[i][j] = False
    mask = np.array(mask)
    # print(df)
    with sns.axes_style("white"):
        fig = sns.heatmap(df, mask=mask, annot=True, square=True, cmap='Greens', cbar=False,
                          annot_kws={"fontsize": 12, "fontweight": 'bold'}, fmt='.2f', vmin=0.5, vmax=1.0)
        fig.xaxis.tick_top()
        fig.yaxis.tick_left()
        plt.xticks(fontsize=8, fontweight='bold', rotation=15)
        plt.yticks(fontsize=8, fontweight='bold', rotation=0)
        for _, spine in fig.spines.items():
            spine.set_visible(True)
        fig = fig.get_figure()
        if show:
            plt.show()
        if save_path:
            fig.savefig(save_path)

def kappa_calculation(result_each_metric_df, dev_set):
    """
    Calculate the kappa matrix for each pair of methods.

    Parameters:
    result_each_metric_df (pandas.DataFrame): A dataframe with column names = [core_set, peripheral_set, dev_set, active_set] and index = [core_set, peripheral_set, dev_set, active_set].
    dev_set (set): A set of developers.

    Returns:
    pandas.DataFrame: A dataframe with the kappa matrix.
    """
    methods = result_each_metric_df["metric"].unique()
    num_methods = len(methods)
    kappa_matrix = np.zeros((num_methods, num_methods))

    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            method1 = methods[i]
            method2 = methods[j]
            method1_core_set = result_each_metric_df[
                result_each_metric_df["metric"] == method1
            ]["core_set"].iloc[0]
            method2_core_set = result_each_metric_df[
                result_each_metric_df["metric"] == method2
            ]["core_set"].iloc[0]
            # dev_s
            # dev_set = method1_core_set.union(method2_core_set)
            method1_core_rater, method2_core_rater = [
                0 for i in range(len(list(dev_set)))
            ], [0 for i in range(len(list(dev_set)))]
            for k, dev in enumerate(list(dev_set)):
                if dev in method1_core_set:
                    method1_core_rater[k] = 1
                if dev in method2_core_set:
                    method2_core_rater[k] = 1
            # print(
            #     f"{method1}, {method2}, hamming: {distance.hamming(method1_core_rater, method2_core_rater)}"
            # )

            kappa_matrix[i, j] = cohen_kappa_score(
                method1_core_rater, method2_core_rater
            )
    # set diagonal value to 1
    for i in range(num_methods):
        kappa_matrix[i, i] = 1

    return pd.DataFrame(kappa_matrix, index=methods, columns=methods)


def caculate_kappas_along_time_window(community_data: pd.DataFrame, time_window: str):
    """
    Calculate kappa matrix along time window
    Parameters:
    community_data (pandas.DataFrame): A dataframe with column names [login, commit, LOC, issue_count, central_degree, version] or [login, commit, LOC, issue_count, central_degree, month].
    time_window (str): A string indicating the type of classification (e.g., "version" or "month").

    Returns:
    list: A list of dataframes with column names   [metric, core_set, peripheral_set, dev_set, active_set] and index = [core_set, peripheral_set, dev_set, active_set].
    """
    df_list = []
    assert time_window in ["version", "month"]
    if time_window == "version":
        # loop_list = sorted(community_data["version"].unique())
        loop_list = sorted(
            (community_data["version"].unique()),
            key=lambda s: list(map(int, s.split("."))),
            reverse=False,
        )
        loop = "version"
    else:
        loop_list = sorted(community_data["month"].unique())
        loop = "month"

    for i, loop_item in enumerate(loop_list):
        if i in [21, 22]:
            print(i)
        df_item = community_data[community_data[loop] == loop_item]
        (
            result_each_metric_df,
            core_set,
            peripheral_set,
            dev_set,
            active_set,
        ) = role_classification(df_item)
        kappa_matrix_item = kappa_calculation(result_each_metric_df, dev_set)
        df_list.append(kappa_matrix_item)
    return df_list


def statistical_analysis_of_kappa_value_along_time_window(df_list):
    """
    Calculate statistical analysis of kappa value along time window
    Parameters:
    df_list (list): A list of dataframes with column names   [metric, core_set, peripheral_set, dev_set, active_set] and index = [core_set, peripheral_set, dev_set, active_set].

    Returns:
    pandas.DataFrame: A dataframe with column names [method1, method2, min_value, max_value, mean_value, midian_value, std]

    """
    methods = df_list[0].index
    num_methods = len(methods)

    analysis_result = pd.DataFrame(
        columns=[
            "method1",
            "method2",
            "min_value",
            "max_value",
            "mean_value",
            "midian_value",
            "std",
        ]
    )
    # make lists of kappa values for each pair of methods
    kappa_values = []
    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            kappa_values.append([])
    # loop through each time window

    for df in df_list:
        cnt = 0
        for i in range(num_methods):
            for j in range(i + 1, num_methods):
                kappa_values[cnt].append(df.iloc[i, j])
                cnt += 1
    # calculate statistics
    cnt = 0
    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            min_value = min(kappa_values[cnt])
            max_value = max(kappa_values[cnt])
            mean_value = np.mean(kappa_values[cnt])
            midian_value = np.median(kappa_values[cnt])
            std = np.std(kappa_values[cnt])
            new_row = pd.DataFrame(
                {
                    "method1": methods[i],
                    "method2": methods[j],
                    "min_value": min_value,
                    "max_value": max_value,
                    "mean_value": mean_value,
                    "midian_value": midian_value,
                    "std": std,
                },
                index=[0],
            )
            analysis_result = pd.concat([analysis_result, new_row], ignore_index=True)
            cnt += 1
    print(analysis_result)
    return analysis_result, kappa_values


def get_pcmci_analysis_data(
    community_data: pd.DataFrame, community_star_data: pd.DataFrame
):
    """
    Parameters:
    community_data: pd.DataFrame with columna names [login,commit_count,LOC,issue_count,central_degree,month,date]
    community_star_data: pd.DataFrame with column names [timestamp, stars]

    Returns:
    pd.DataFrame: A dataframe with column names [month,active_users,core_developers,peripheral_developers,retention rate,inflow rate,dropout rate,core-to-peri rate,ratio p-c,ratio a-d,stars].
    """
    # init columns
    columns = [
        "version",
        "active_users",
        "core_developers",
        "peripheral_developers",
        "retention rate",
        "inflow rate",
        "dropout rate",
        "core-to-peri rate",
        "ratio p-c",
        "ratio a-d",
        "stars",
    ]
    def translate_date(date_string):
        return '-'.join(date_string.split('.')[:2])
    
    community_star_data["timestamp"] = community_star_data["timestamp"].apply(lambda x: translate_date(x))
    community_star_data = community_star_data.rename(columns={"timestamp": "month"})
    pcmci_data = pd.DataFrame(columns=columns)
    # init core set
    exisited_core_set = set()
    last_core_set = set()
    loop_list = sorted(community_data["month"].unique())
    # loop to calculate ratio
    for i, loop_item in enumerate(loop_list):
        (
            _,
            cur_core_set,
            cur_peripheral_set,
            cur_dev_set,
            cur_active_set,
        ) = role_classification(community_data[community_data["month"] == loop_item])
        if i != 0:
            # calculate committer_after_set
            committer_after_set = set()
            for after_loop_item in loop_list[i:]:
                committer_set = set()
                contribution_table = community_data[
                    community_data["month"] == after_loop_item
                ]
                for _, row in contribution_table.iterrows():
                    if row["commit_count"] > 0:
                        committer_set.add(row["login"])
                committer_after_set |= committer_set
            ratios = calculate_ratios(
                exisited_core_set,
                cur_core_set,
                last_core_set,
                cur_peripheral_set,
                cur_active_set,
                cur_dev_set,
                committer_after_set,
            )
            new_row = {
                "active_users": len(cur_active_set),
                "core_developers": len(cur_core_set),
                "peripheral_developers": len(cur_peripheral_set),
                "ratio p-c": ratios[0],
                "ratio a-d": ratios[1],
                "retention rate": ratios[2],
                "dropout rate": ratios[3],
                "core-to-peri rate": ratios[4],
                "inflow rate": ratios[5],
                "version": loop_item,
                "stars": community_star_data[
                    community_star_data["month"] == loop_item
                ]["stars"].iloc[0],
            }
            pcmci_data = pd.concat([pcmci_data, pd.DataFrame([new_row])], ignore_index=True)
        else:
            new_row = {
                "active_users": len(cur_active_set),
                "core_developers": len(cur_core_set),
                "peripheral_developers": len(cur_peripheral_set),
                "ratio p-c": len(cur_peripheral_set) / len(cur_core_set),
                "ratio a-d": len(cur_active_set) / len(cur_dev_set),
                "retention rate": 0,
                "dropout rate": 0,
                "core-to-peri rate": 0,
                "inflow rate": 0,
                "version": loop_item,
                "stars": community_star_data[
                    community_star_data["month"] == loop_item
                ]["stars"].iloc[0],
            }
            pcmci_data = pd.concat([pcmci_data, pd.DataFrame([new_row])], ignore_index=True)
        exisited_core_set |= cur_core_set
        last_core_set = cur_core_set
    print(pcmci_data)
    return pcmci_data


if __name__ == "__main__":

    community_data = pd.read_csv('../../data/contributions/months/pytorch.csv')
    community_star_data = pd.read_csv('../../data/raw/stars/pytorch_monthly.csv')
    
    pcmci_data = get_pcmci_analysis_data(community_data, community_star_data)
    print(pcmci_data)
    pcmci_data.to_csv('../../outputs/results/pcmci_data.csv', index=False)