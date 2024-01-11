import sys
import time
import json
import pandas as pd
sys.path.append("..")
from utils.tool import (
    # load_config,
    load_csv,
    save_csv,
    parse_timestamp,
    build_releases_windows,
    build_months_windows,
    load_list_via_txt,
)


# load config
with open('../config.json') as file:
    config = json.load(file)


def filter_dataframe_by_time(dataframe, start_date, end_date):
    """
    Filters the dataframe based on a time window.

    :param dataframe: Dataframe to be filtered.
    :param start_date: Start of the time window.
    :param end_date: End of the time window.
    :return: Filtered dataframe.
    """
    start_timestamp = parse_timestamp(start_date)
    end_timestamp = parse_timestamp(end_date)

    return dataframe[(dataframe["timestamp"] <= end_timestamp) & (dataframe["timestamp"] >= start_timestamp)]

# def raw_data_sampling(community, data, time_window):
#     if time_window == [config[community]["start_date"], config[community]["end_date"]]:
#         return data
#     t1, t2 = parse_timestamp(time_window[0]), parse_timestamp(time_window[1])

#     return data[(data["timestamp"] <= t2) & (data["timestamp"] >= t1)]

def raw_data_sampling(community, data, time_window):
    """
    Filter the data based on the specified time window.

    Args:
        community (str): The community name.
        data (DataFrame): The input data.
        time_window (list): A list containing the start and end dates of the time window.

    Returns:
        DataFrame: The filtered data within the specified time window.
    """
    if time_window == [config[community]["start_date"], config[community]["end_date"]]:
        return data
    
    start_date, end_date = parse_timestamp(time_window[0]), parse_timestamp(time_window[1])
    filtered_data = data[(data["timestamp"] <= end_date) & (data["timestamp"] >= start_date)]
    
    return filtered_data

def sample_and_aggregate_commits(community, commits, time_window, developers):
    """
    Samples commits based on the specified time window, then aggregates commit counts and lines of code (LOC) per developer.

    :param community: Name of the community to filter data for.
    :param commits: DataFrame containing commit data.
    :param time_window: List containing start and end timestamps defining the time window.
    :param developers: DataFrame containing developer data.
    :return: DataFrame containing aggregated commit counts and LOC per developer.
    """
    # Sample commits based on the time window
    sampled_commits = raw_data_sampling(community, commits, time_window)
    
    # Check if there are any commits in the sampled data
    if sampled_commits.empty:
        raise ValueError("No commits found in the specified time window.")
    
    # Preprocess the data: set empty LOC to 0, convert LOC to integers
    sampled_commits['LOC'] = sampled_commits['LOC'].apply(lambda x: int(float(x)) if x != "" else 0)

    # Aggregate commit counts and LOC by developer login
    aggregation_functions = {'LOC': 'sum', 'name': 'count'}
    aggregated_data = sampled_commits.groupby('name').agg(aggregation_functions)
    aggregated_data.rename(columns={'name': 'commit_count'}, inplace=True)

    # Ensure the developer logins in the aggregated data are consistent with the developers dataframe
    valid_developer_logins = developers['login'].unique()
    aggregated_data = aggregated_data[aggregated_data.index.isin(valid_developer_logins)]

    return aggregated_data

# def commit_sampling(community, commits, time_window, developers):
#     """
#     Perform commit sampling and calculate commit count and LOC (Lines of Code) for each developer.

#     Args:
#         community (str): Name of the community.
#         commits (DataFrame): DataFrame containing commit data.
#         time_window (str): Time window for sampling commits.
#         developers (DataFrame): DataFrame containing developer data.

#     Returns:
#         DataFrame: DataFrame containing commit count and LOC for each developer.

#     """
#     sampled_commits = raw_data_sampling(community, commits, time_window)

#     commit_count_dic = {}
#     LOC_dic = {}

#     # Create a set of unique logins from the developers DataFrame
#     developer_logins = set(developers["login"])

#     for idx, commit in sampled_commits.iterrows():
#         login = commit["name"]

#         # Check if the login exists in the set of developer logins
#         if login in developer_logins:
#             LOC = int(float(commit["LOC"])) if commit["LOC"] != "" else 0

#             # Use defaultdict to simplify the logic for updating commit_count_dic and LOC_dic
#             commit_count_dic[login] = commit_count_dic.get(login, 0) + 1
#             LOC_dic[login] = LOC_dic.get(login, 0) + LOC

#     commit_count_df = pd.DataFrame.from_dict(commit_count_dic, orient="index", columns=["commit_count"])
#     LOC_df = pd.DataFrame.from_dict(LOC_dic, orient="index", columns=["LOC"])

#     return commit_count_df.join(LOC_df)

def commit_sampling(community, commits, time_window, developers):
    sampled_commits = raw_data_sampling(community, commits, time_window)

    commit_count_dic, LOC_dic = {}, {}
    for idx, commit in sampled_commits.iterrows():
        name, email = commit["name"], commit["email"]
        query = developers[
            (developers["name"] == name) & (developers["email"] == email)
        ]
        ans = query["login"].values
        if len(ans) != 0:
            login = ans[0]
            LOC = int(float(commit["LOC"])) if commit["LOC"] != "" else 0
            if commit_count_dic.get(login):
                commit_count_dic[login] += 1
                LOC_dic[login] += LOC
            else:
                commit_count_dic[login] = 1
                LOC_dic[login] = LOC
    commit_count_df = pd.DataFrame.from_dict(
        commit_count_dic, orient="index", columns=["commit_count"]
    )
    LOC_df = pd.DataFrame.from_dict(LOC_dic, orient="index", columns=["LOC"])

    return commit_count_df.join(LOC_df)


def issue_sampling(community, issues, time_window):
    sampled_issues = raw_data_sampling(community, issues, time_window)

    issue_count_dic = {}
    for _, issue in sampled_issues.iterrows():
        login = issue["author"]
        if issue_count_dic.get(login):
            issue_count_dic[login] += 1
        else:
            issue_count_dic[login] = 1
    issue_count_df = pd.DataFrame.from_dict(
        issue_count_dic, orient="index", columns=["issue_count"]
    )

    return issue_count_df

# def build_network(events, network_args):
#     """
#     Build a network based on events data and calculate the central degree for each actor.

#     Args:
#         events (DataFrame): DataFrame containing events data.
#         network_args (dict): Dictionary containing network parameters.

#     Returns:
#         DataFrame: DataFrame containing the central degree for each actor.

#     """

#     # Parameters setting
#     directed = network_args["directed"]
#     respect_temporal_order = network_args["respect_temporal_order"]

#     central_degree_count = {}

#     def dict_value_plus_one(dic, key):
#         if key in dic:
#             dic[key] += 1
#         else:
#             dic[key] = 1
#     # Group by 'issue_id'
#     grouped = events.groupby('issue_id')
#     for _, group in grouped:
#         if respect_temporal_order:
#             former_authors = []
#             for _, cur_event in group.iterrows():
#                 for author in former_authors:
#                     dict_value_plus_one(central_degree_count, cur_event["actor"])
#                     if not directed:
#                         dict_value_plus_one(central_degree_count, author)
#                 former_authors.append(cur_event["actor"])
#         else:
#             authors = set(group["actor"])
#             for _, cur_event in group.iterrows():
#                 for author in authors:
#                     dict_value_plus_one(central_degree_count, cur_event["actor"])
#                     if not directed:
#                         dict_value_plus_one(central_degree_count, author)

#     central_degree_count_df = pd.DataFrame.from_dict(central_degree_count, orient="index", columns=["central_degree"])
#     return central_degree_count_df

def build_network(events, network_args):
    """
    :param issues:
    :param events:
    :param developers:
    :return network: (source, target, timestamp)
    """

    # parameters setting
    directed = network_args["directed"]
    respect_temporal_order = network_args["respect_temporal_order"]

    central_degree_count = {}

    def dict_value_plus_one(dic, key):
        if dic.get(key):
            dic[key] += 1
        else:
            dic[key] = 1

    cursor_x, cursor_y = 0, 0
    events_num = events.shape[0]
    while cursor_y < events_num:
        while (
            cursor_y < events_num
            and events.iloc[cursor_x]["issue_id"] == events.iloc[cursor_y]["issue_id"]
        ):
            cursor_y += 1

        events_group_by_id = events[cursor_x:cursor_y]
        # print(events_group_by_id.iloc[0]['issue_id'], events_group_by_id.shape[0])

        if respect_temporal_order:
            former_authors = []
            for i, cur_event in events_group_by_id.iterrows():
                for author in former_authors:
                    dict_value_plus_one(central_degree_count, cur_event["actor"])
                    if not directed:
                        dict_value_plus_one(central_degree_count, author)
                former_authors.append(cur_event["actor"])
        else:
            authors = set(events_group_by_id["actor"])
            for i, cur_event in events_group_by_id.iterrows():
                for author in authors:
                    dict_value_plus_one(central_degree_count, cur_event["actor"])
                    if not directed:
                        dict_value_plus_one(central_degree_count, author)
        cursor_x = cursor_y

    central_degree_count_df = pd.DataFrame.from_dict(
        central_degree_count, orient="index", columns=["central_degree"]
    )
    return central_degree_count_df


def events_sampling(network, allow_self_loop=False):
    central_degree_count = {}
    for idx, edge in network:
        a, b = edge["source"], edge["target"]
        if not allow_self_loop and a == b:
            continue
        if central_degree_count.get(a):
            central_degree_count[a] += 1
        else:
            central_degree_count[a] = 1

    central_degree_count_df = pd.DataFrame.from_dict(
        central_degree_count, orient="index", columns=["central_degree"]
    )
    return central_degree_count_df


def contribution_sampling(
    community, time_window, developers, commits, issues, events, network_args
):
    """
    Obtain the contribution of the give time window.
    """
    # compute commit_count and LOC
    # (login, commit_count, LOC)
    commit_df = commit_sampling(community, commits, time_window, developers)

    # compute issues_count
    # (login, issue_count)
    issue_df = issue_sampling(community, issues, time_window)

    # compute comments_count
    # (login, central_degree)
    sampled_events = raw_data_sampling(community, events, time_window)
    event_df = build_network(sampled_events, network_args)

    # aggregate the contributions
    contributions_df = pd.concat([commit_df, issue_df, event_df], axis=1).fillna(0)
    contributions_df = contributions_df.astype(int)
    contributions_df = contributions_df.reset_index().rename(columns={"index": "login"})
    return contributions_df


def remove_bots(
    commits,
    issues,
    events,
    removed_authors,
    removed_logins,
    developers_only,
    developers,
):
    # print("original commits shape: ", commits.shape[0])
    # print the commits name in removed_logins
    commits = commits[~commits["name"].isin(removed_authors)]
    commits = commits[~commits["name"].isin(removed_logins)]
    commits_copy = commits.copy()
    commits_copy["timestamp"] = commits_copy["timestamp"].apply(parse_timestamp)
    # print("commits shape after removing bots: ", commits.shape[0])
    print("original issues shape: ", issues.shape[0])
    issues = issues[~issues["author"].isin(removed_authors)]
    issues = issues[~issues["author"].isin(removed_logins)]
    print("issues shape after removing bots: ", issues.shape[0])
    # issues["timestamp"] = issues["timestamp"].apply(parse_timestamp)
    issues_copy = issues.copy()
    issues_copy["timestamp"] = issues_copy["timestamp"].apply(parse_timestamp)
    print("original events shape: ", events.shape[0])
    events = events[~events["actor"].isin(removed_authors)]
    events = events[~events["actor"].isin(removed_logins)]
    events_copy = events.copy()
    events_copy["timestamp"] = events_copy["timestamp"].apply(parse_timestamp)
    print("events shape after removing bots: ", events.shape[0])
    # events["timestamp"] = events["timestamp"].apply(parse_timestamp)

    if developers_only:
        dev_set = set()
        for idx, developer in developers.iterrows():
            dev_set.add(developer["login"])
        drop_index = []
        for idx, event in events_copy.iterrows():
            if event["actor"] not in dev_set:
                drop_index.append(idx)
        events_copy = events_copy.drop(drop_index, axis=0)
    return commits_copy, issues_copy, events_copy


def get_contribution_with_different_time_windows(
    community, time_window_unit, logins, developers, commits, issues, events, network_args
):
    print(f"get {time_window_unit} windows ...")
    if time_window_unit == "releases":
        names, windows, dates = build_releases_windows(community,
            f"../../data/raw/releases_time/{community}.csv", False
        )
    elif time_window_unit == "months":
        names, windows, dates = build_months_windows(community)
    else:
        return

    contributions = pd.DataFrame()
    for i, (save_name, window, date) in enumerate(zip(names, windows, dates)):
        print(save_name, window)
        df = contribution_sampling(
            community, window, developers, commits, issues, events, network_args
        )
        df = df.merge(logins, how="outer").fillna(0)
        df[["commit_count", "LOC", "issue_count", "central_degree"]] = df[
            ["commit_count", "LOC", "issue_count", "central_degree"]
        ].astype(int)

        if time_window_unit == "releases":
            df["version"] = save_name
        elif time_window_unit == "months":
            df["month"] = save_name

        df["date"] = date
        if i == 0:
            contributions = df
        else:
            contributions = pd.concat([contributions, df], axis=0)
        # save_csv(df, f'data/contributions/releases/{community}_{save_name}.csv')
    save_csv(contributions, f"../../data/contributions/{time_window_unit}/{community}.csv")
    print(f'The contributions of {time_window_unit} are in "../../data/contributions/{time_window_unit}"')


def get_contribution_table(
    community,
    developers_file,
    commits_file,
    issues_file,
    comments_file,
    removed_authors,
    removed_logins,
    network_args,
):
    """
    Obtain the contribution table identified by login based on commit and issue data and the list of developers.
    The statistics granularity including [release] and [month]

    :param developers_file      : (login, name, email, have login)
    :param commits_file         : (hash, timestamp, name, email, LOC)
    :param issue_author_file          : (issue_id, type, open_timestamp, author)
    :param issue_events_file        : (issue_id, actor, type, timestamp, body)

    :return contribution_table  : (login, commit count, LOC, issue count, central degree, release/month)
    """

    # load data
    print("load data ...")
    developers = load_csv(developers_file)
    commits = load_csv(commits_file)
    issues = load_csv(issues_file)
    # events = load_csv(comments_file)
    events = pd.read_csv(comments_file )
    # events = pd.read_csv(comments_file, dtype={"issue_id": str})

    # filter bots
    print("filter bots ...")
    commits, issues, events = remove_bots(
        commits,
        issues,
        events,
        removed_authors,
        removed_logins,
        network_args["developers_only"],
        developers,
    )

    # global sampling results
    print("get global contribution table results ...")
    global_time_window = [
        config[community]["start_date"],
        config[community]["end_date"],
    ]
    global_contribution = contribution_sampling(
        community, global_time_window, developers, commits, issues, events, network_args
    )
    # global_contribution = load_csv(f"../../data/contributions/global/{community}.csv")
    # logins = global_contribution["login"]
    logins = global_contribution["login"]
    save_csv(global_contribution, f"../../data/contributions/global/{community}.csv")

    # releases time window sampling
    get_contribution_with_different_time_windows(
        community, "releases", logins, developers, commits, issues, events, network_args
    )

    # months time window sampling
    get_contribution_with_different_time_windows(
        community, "months", logins, developers, commits, issues, events, network_args
    )


if __name__ == "__main__":
    network_args = {
        "developers_only": True,
        "directed": True,  # True or False
        "respect_temporal_order": True,  # True or False
    }
    
    for community in ["pytorch", "tensorflow"]:
    # for community in ["tensorflow"]:
        # for community in ['tensorflow']:
        print(f"--------Contribution Table Generation of {community}--------")
        print("Please wait. This procedure may take a few minutes to complete.")
        start_time = time.time()
        get_contribution_table(
            community=community,
            developers_file=f"../../data/preprocessing/{community}_developers.csv",
            commits_file=f"../../data/raw/commits/{community}.csv",
            issues_file=f"../../data/raw/issues/{community}.csv",
            comments_file=f"../../data/raw/events/{community}.csv",
            removed_authors=load_list_via_txt(f"../../data/raw/bots/{community}_authors.txt"),
            removed_logins=load_csv(f"../../data/raw/bots/{community}_logins.csv")["login"],
            network_args=network_args,
        )
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration of {community} : {duration:.2f} seconds")
        print(f"Finishing Data Sampling of {community}")
    print("Finishing Data Sampling, the results are currently stored in \"../../data/contributions\"")