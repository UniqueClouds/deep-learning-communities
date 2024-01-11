import pandas as pd
from data_preprocessing.classification import role_transition_capture, role_classification


def generate_longitudinal_data(community_data: pd.DataFrame, star_data: pd.DataFrame):
    """
    Genereate dataframes for longtitudinal analysis and plot.

    Parameters:
    community_data (pandas.DataFrame): A dataframe with column names [login, commit, LOC, issue_count, central_degree, version]
    star_data (pandas.DataFrame): A dataframe with column names [timestamp, stars]
    Returns:
    pandas.DataFrames: A dataframe with column names ['version', 'active_users', 'core_developers', 'peripheral_developers', 'retention_rate', 'inflow_rate', 'dropout_rate', 'ratio_peri_core', 'ratio_active_dev', 'dropout_rate', 'core2peripheral_rate', 'inflow_rate', 'release_date']
    """
    # 1. get role classification of each version
    role_classification_version = pd.DataFrame(
        columns=["version", "active_users", "core_developers", "peripheral_developers"]
    )
    version_list = sorted(
        community_data["version"].unique(),
        key=lambda s: list(map(int, s.split("."))),
        reverse=False,
    )
    print(version_list)
    for version in version_list:
        (_, core_set, peripheral_set, _, active_set) = role_classification(
            community_data[community_data["version"] == version]
        )
        # get numbers of each set
        num_core_dev = len(core_set)
        num_peri_dev = len(peripheral_set)
        num_active_usr = len(active_set)
        # add a line to role_classfication_version
        role_classification_version = pd.concat(
            [
                role_classification_version,
                pd.DataFrame(
                    {
                        "version": [version],
                        "active_users": [num_active_usr],
                        "core_developers": [num_core_dev],
                        "peripheral_developers": [num_peri_dev],
                    }
                ),
            ],
            ignore_index=True,
        )
    # 2. returns [ratio_peri_core, ratio_active_dev, retention_rate, dropout_rate, core2peripheral_rate, inflow_rate, version]
    role_transition_result = role_transition_capture(community_data, "version")

    # 3. get stars of each version and add timestamp
    # add a column and value to star_data
    # add version_list to star_data['version'] but star_data has no column 'version'
    star_data["version"] = version_list
    # change column name 'timestamp' to 'release_date'
    star_data.rename(columns={"timestamp": "release_date"}, inplace=True)

    # merge role_classification_version and role_transition_result according the 'version'
    longitudinal_data = pd.merge(
        pd.merge(
            role_classification_version,
            role_transition_result,
            on="version",
            how="outer",
        ),
        star_data,
        on="version",
        how="outer",
    )

    # change the result column order   ['version', 'active_users', 'core_developers', 'peripheral_developers', 'retention_rate', 'inflow_rate', 'dropout_rate', 'ratio_peri_core', 'ratio_active_dev', 'dropout_rate', 'core2peripheral_rate', 'inflow_rate','stars', 'release_date']
    longitudinal_data = longitudinal_data[
        [
            "version",
            "active_users",
            "core_developers",
            "peripheral_developers",
            "retention_rate",
            "inflow_rate",
            "dropout_rate",
            "ratio_peri_core",
            "ratio_active_dev",
            "core2peripheral_rate",
            "inflow_rate",
            "stars",
            "release_date",
        ]
    ]
    return longitudinal_data

def generate_longitudinal_data_month(community_data: pd.DataFrame, star_data: pd.DataFrame):
    """
    Genereate dataframes for longtitudinal analysis and plot.

    Parameters:
    community_data (pandas.DataFrame): A dataframe with column names [login, commit, LOC, issue_count, central_degree, version]
    star_data (pandas.DataFrame): A dataframe with column names [timestamp, stars]
    Returns:
    pandas.DataFrames: A dataframe with column names ['version', 'active_users', 'core_developers', 'peripheral_developers', 'retention_rate', 'inflow_rate', 'dropout_rate', 'ratio_peri_core', 'ratio_active_dev', 'dropout_rate', 'core2peripheral_rate', 'inflow_rate', 'release_date']
    """
    # 1. get role classification of each version
    role_classification_version = pd.DataFrame(
        columns=["version", "active_users", "core_developers", "peripheral_developers"]
    )
    version_list = sorted(
        community_data["version"].unique(),
        key=lambda s: list(map(int, s.split("."))),
        reverse=False,
    )
    print(version_list)
    for version in version_list:
        (_, core_set, peripheral_set, _, active_set) = role_classification(
            community_data[community_data["month"] == month]
        )
        # get numbers of each set
        num_core_dev = len(core_set)
        num_peri_dev = len(peripheral_set)
        num_active_usr = len(active_set)
        # add a line to role_classfication_version
        role_classification_version = role_classification_version.append(
            {
                "version": version,
                "active_users": num_active_usr,
                "core_developers": num_core_dev,
                "peripheral_developers": num_peri_dev,
            },
            ignore_index=True,
        )

    # 2. returns [ratio_peri_core, ratio_active_dev, retention_rate, dropout_rate, core2peripheral_rate, inflow_rate, version]
    role_transition_result = role_transition_capture(community_data, "version")

    # 3. get stars of each version and add timestamp
    # add a column and value to star_data
    star_data["version"] = version_list
    # change column name 'timestamp' to 'release_date'
    star_data.rename(columns={"timestamp": "release_date"}, inplace=True)

    # merge role_classification_version and role_transition_result according the 'version'
    longitudinal_data = pd.merge(
        pd.merge(
            role_classification_version,
            role_transition_result,
            on="version",
            how="outer",
        ),
        star_data,
        on="version",
        how="outer",
    )

    # change the result column order   ['version', 'active_users', 'core_developers', 'peripheral_developers', 'retention_rate', 'inflow_rate', 'dropout_rate', 'ratio_peri_core', 'ratio_active_dev', 'dropout_rate', 'core2peripheral_rate', 'inflow_rate','stars', 'release_date']
    longitudinal_data = longitudinal_data[
        [
            "version",
            "active_users",
            "core_developers",
            "peripheral_developers",
            "retention_rate",
            "inflow_rate",
            "dropout_rate",
            "ratio_peri_core",
            "ratio_active_dev",
            "core2peripheral_rate",
            "inflow_rate",
            "stars",
            "release_date",
        ]
    ]
    return longitudinal_data

