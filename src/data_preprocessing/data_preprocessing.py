import pandas as pd
import hashlib
import sys

sys.path.append("../")
from utils.compute import compute_similarity
from utils.tool import load_list_via_txt, load_csv


def remove_bots(
    commits_file, community_name, hash2login_file, removed_authors, removed_logins
):
    """
    To remove bot accounts from the commits_file and hash2login_file.

    :param commits_file     : (hash, timestamp, name, email, LOC)
    :param hash2login_file  : (hash, login)
    :param removed_authors  : list of authors' name
    :param removed_logins   : list of participants' login
    """

    # load data
    commits = pd.read_csv(commits_file, encoding="utf_8_sig", keep_default_na=False)
    hash2login = pd.read_csv(
        hash2login_file, encoding="utf_8_sig", keep_default_na=False
    )
    original_length_commits = commits.shape[0]
    # remove bots according to 'name' column
    remove_hash = []
    remove_idx = []
    for name in removed_authors:
        indexes = commits[commits["name"] == name].index
        for i in indexes:
            remove_hash.append(commits["hash"].iloc[i])
            remove_idx.append(i)
    for name in removed_logins:
        indexes = commits[commits["name"] == name].index
        for i in indexes:
            remove_hash.append(commits["hash"].iloc[i])
            remove_idx.append(i)
    commits = commits.drop(remove_idx, axis=0)
    remove_idx = []
    for i in range(hash2login.shape[0]):
        sha = hash2login["hash"].iloc[i]
        if sha in remove_hash:
            remove_idx.append(i)
    hash2login = hash2login.drop(remove_idx, axis=0)
    for login in removed_logins:
        hash2login = hash2login.drop(
            hash2login[hash2login["login"] == login].index, axis=0
        )
    
    print(f"{community_name} community has {original_length_commits} commits originally.")
    print(f"After removing bot accounts according to 'name' column, {community_name} community has {commits.shape[0]} commits.")
    print(
        f"Among them, {hash2login.shape[0]}/{commits.shape[0]} ({round(hash2login.shape[0]/commits.shape[0] * 100, 2)}%) matched to their logins."
    )

    return commits, hash2login


def developers_matching_by_common_login(community_name, commits_file, hash2login_file):
    """
    Using the commits and hash2login tables, code authors and issue participants are matched by a common hash,
    resulting in tuples of (login, name, email), we then matched the developers by their common login,
    and store authors that failed to match (name, email) in the no_login_file for subsequent processing.

    :param commits_file     : (hash, timestamp, name, email, LOC)
    :param hash2login_file  : (hash, login)

    :return match_by_common_login_file  : (login, name, email)
    :return no_login_file               : (name, email)
    """
    # load commit and hash2login file
    commits, hash2login = remove_bots(
        commits_file=commits_file,
        community_name=community_name,
        hash2login_file=hash2login_file,
        removed_authors=load_list_via_txt(
            f"../../data/raw/bots/{community_name}_authors.txt"
        ),
        removed_logins=list(
            load_csv(f"../../data/raw/bots/{community_name}_logins.csv")["login"]
        ),
    )

    # match by the common login
    login_name_email = pd.merge(commits, hash2login)

    login_name_email = login_name_email[["login", "name", "email"]]
    matches = login_name_email[login_name_email["login"] != ""].drop_duplicates()
    matches = matches.sort_values(by="login")

    # find unmatched (name, email) pairs
    all_name_email = commits[["name", "email"]].drop_duplicates()
    matched_name_email = matches[["name", "email"]].drop_duplicates()
    no_login = set(list(zip(all_name_email["name"], all_name_email["email"]))) - set(
        list(zip(matched_name_email["name"], matched_name_email["email"]))
    )
    names, emails = [], []
    for name, email in no_login:
        names.append(name)
        emails.append(email)
    no_login_df = pd.DataFrame(data={"name": names, "email": emails})

    # store the pandas.DataFrame format results into csv file
    matches_file, no_login_file = (
        f"../../data/preprocessing/{community_name}_match_by_common_login.csv",
        f"../../data/preprocessing/{community_name}_no_login.csv",
    )
    matches.drop_duplicates().to_csv(matches_file, encoding="utf_8_sig", index=False)
    no_login_df.drop_duplicates().to_csv(
        no_login_file, encoding="utf_8_sig", index=False
    )

    print(
        f'After matching by common login, there are {len(set(matches["login"]))} unqiue developers.'
    )

    return matches_file, no_login_file


def heuristic_1st_round(no_login, matches):
    """
    heuristic attach to current authors who are matched to their logins
    """
    add_merge = pd.DataFrame(
        columns=matches.columns
    )  # stores the aliases that are matched by heuristic
    no_login_delete_index = []

    for i in range(no_login.shape[0]):
        # (name, email) that has not matched to their login
        name_s, email_s = no_login.iloc[i]["name"], no_login.iloc[i]["email"]

        sim_cnt = 0  # number of logins the (name_s, email_s) has matched by heuristic
        logins = (
            set()
        )  # set of login that are matched to (name_s, email_s) by heuristic
        for j in range(matches.shape[0]):
            name_t, email_t = matches.iloc[j]["name"], matches.iloc[j]["email"]

            # compute the similarity between (name_s, email_s) and (name_t, email_t)
            sim_score = compute_similarity(
                x={"name": name_s, "email": email_s},
                y={"name": name_t, "email": email_t},
            )
            if sim_score >= 95:
                login_t = matches.iloc[j]["login"]  # matched login of (name_s, email_s)
                if login_t not in logins:
                    logins.add(login_t)
                    sim_cnt += 1
        if sim_cnt == 0 or sim_cnt >= 2:
            # There is no (name, email) or multiple pair matching to (name_s, email_s), it needs to be checked manually
            continue
        else:
            # There is one login matching to (name_s, email_s), it needs to be merged
            login = list(logins)[0]
            add_merge.loc[add_merge.shape[0]] = [login, name_s, email_s]
            no_login_delete_index.append(i)
    no_login = no_login.drop(index=no_login_delete_index, axis=0)
    # no_login.to_csv(no_login_file, encoding='utf_8_sig', index=False)

    # save matches
    add_merge_df = pd.DataFrame(add_merge, columns=matches.columns)
    matches = pd.concat([matches, add_merge_df], ignore_index=True)

    matches = matches.drop_duplicates().sort_values(by="login", ascending=True)

    print(
        f'After 1st round heuristic matching, there are {len(set(matches["login"]))} unqiue developers.'
    )

    return no_login, matches


def heuristic_2nd_round(no_login):
    """
    heuristic merge for (name, email) that haves no login so far.
    """
    n = no_login.shape[0]

    threshold = 95
    similar_authors = []
    for i in range(n):
        for j in range(i + 1, n):
            name1, email1 = no_login["name"].iloc[i], no_login["email"].iloc[i]
            name2, email2 = no_login["name"].iloc[j], no_login["email"].iloc[j]
            sim_score = compute_similarity(
                x={"name": name1, "email": email1}, y={"name": name2, "email": email2}
            )
            if sim_score >= threshold:
                similar_authors.append((i, j))

    visit = [0 for _ in range(n)]
    adjacency_table = [[] for _ in range(n)]
    for e in similar_authors:
        a, b = e
        adjacency_table[a].append(b)
        adjacency_table[b].append(a)

    def merge_authors(idx, cur):
        if visit[idx]:
            return
        visit[idx] = 1
        cur.append((no_login.iloc[idx]["name"], no_login.iloc[idx]["email"]))
        for a in adjacency_table[idx]:
            if not visit[a]:
                merge_authors(a, cur)

    authors_list = []
    for i in range(len(adjacency_table)):
        if visit[i]:
            continue
        cur = []
        merge_authors(i, cur)
        authors_list.append(cur)

    manual_list = pd.DataFrame(columns=["id", "name", "email"])
    for i in range(len(authors_list)):
        aliases = authors_list[i]
        for alias in aliases:
            manual_list.loc[manual_list.shape[0]] = [i, alias[0], alias[1]]

    print(
        f'After 2nd round heuristic matching, there are {max(list(manual_list["id"])) + 1} developers to be matched manually.'
    )

    return manual_list


def developers_matching_by_heuristic(
    community_name, match_by_common_login_file, no_login_file
):
    """
    To attempt matching name and email pairs that didn't succeed through common login using a heuristic approach.

    We use Levenshtein Distance as the similarity metric. To calculate the similarity between the two pairs (name1,
    email1) and (name2, email2), we calculate the similarity between name1 and name2, as well as the similarity
    between the prefixes of email1 and email2 (where the prefix refers to the string before the "@" symbol),
    and then take the maximum value to obtain the similarity between the two pairs (name1, email1) and (name2,
    email2). We set the threshold to 95.

    :param match_by_common_login_file   : (login, name, email)
    :param no_login_file                : (name, email)

    :return matches_file        : (login, name, email)
    :return manual_list_file    : (name, email)
    """

    # load data
    no_login = pd.read_csv(no_login_file, encoding="utf_8_sig")
    matches = pd.read_csv(match_by_common_login_file, encoding="utf_8_sig")

    # 1st round: heuristic attach to current authors who are matched to their logins
    no_login, matches = heuristic_1st_round(no_login, matches)

    # 2nd round: heuristic merge for (name, email) that haves no login so far.
    manual_list = heuristic_2nd_round(no_login)

    # save results
    matches_file = f"../../data/preprocessing/{community_name}_matches.csv"
    matches.to_csv(matches_file, encoding="utf_8_sig", index=False)
    manual_list_file = f"../../data/preprocessing/{community_name}_manual_list.csv"
    manual_list.to_csv(manual_list_file, encoding="utf_8_sig", index=False)

    return matches_file, manual_list_file


def get_manual_matches(community_name):
    manual_list = pd.read_csv(
        f"../../data/preprocessing/{community_name}_manual_list.csv", encoding="utf_8_sig"
    )
    manuals = pd.read_csv(
        f"../../data/matching/{community_name}_manual.csv", encoding="utf_8_sig"
    )
    id2login = {}
    for i in range(manuals.shape[0]):
        login, name, email = (
            manuals["login"].iloc[i],
            manuals["name"].iloc[i],
            manuals["email"].iloc[i],
        )
        answers = manual_list[
            (manual_list["name"] == name) & (manual_list["email"] == email)
        ]["id"].values
        if len(answers) != 0:
            author_id = answers[0]
            id2login[author_id] = login

    login_answers = []
    for i in range(manual_list.shape[0]):
        id, name, email = (
            manual_list["id"].iloc[i],
            manual_list["name"].iloc[i],
            manual_list["email"].iloc[i],
        )
        if id2login.get(id):
            login_answers.append(id2login[id])
        else:
            login_answers.append("")
    manual_list["login"] = login_answers
    manual_list = manual_list.sort_values(by="login", ascending=True)
    manual_matches_file = f"../../data/preprocessing/{community_name}_manual_matches.csv"
    manual_list.to_csv(manual_matches_file, encoding="utf_8_sig", index=False)

    print(
        f"After manual matching, {len(list(id2login.keys()))} developers have matched to their logins."
    )


def get_developers(community_name, matches_file, manual_matches_file):
    matches = pd.read_csv(matches_file, encoding="utf_8_sig", keep_default_na=False)
    manual_matches = pd.read_csv(
        manual_matches_file, encoding="utf_8_sig", keep_default_na=False
    )

    grouped_authors = manual_matches.groupby("id")
    have_login = [
        1 for _ in range(matches.shape[0])
    ]  # have_login[i] denotes whether author i has login
    md5 = hashlib.md5()
    for i in set(manual_matches["id"]):
        author = grouped_authors.get_group(i)
        do_have_login = 1
        if author["login"].iloc[0] == "":
            # no login
            s = str(author["name"].iloc[0] + " <" + author["email"].iloc[0] + ">")
            md5.update(s.encode("utf-8"))
            login = md5.hexdigest()
            do_have_login = 0
        else:
            login = author["login"].iloc[0]

        for j in range(author.shape[0]):
            name, email = author["name"].iloc[j], author["email"].iloc[j]
            matches.loc[matches.shape[0]] = [login, name, email]
            have_login.append(do_have_login)

    matches["have login"] = have_login
    matches = matches.sort_values(by="login")

    developers_file = f"../../data/preprocessing/{community_name}_developers.csv"
    matches.to_csv(developers_file, encoding="utf_8_sig", index=False)
    print(f'After preprocessing, there are {len(set(matches["login"]))} developers.')
    print(
        f'{len(set(matches[matches["have login"] == 1]["login"]))} of them matched to their logins successfully.'
    )
    print(
        f'{len(set(matches[matches["have login"] == 0]["login"]))} of them failed to matched to their logins.'
    )

    return developers_file


def get_participants(community_name, comments_file):
    participants_file = f"../../data/preprocessing/{community_name}_participants.csv"
    removed_logins = list(
        load_csv(f"../../data/raw/bots/{community_name}_logins.csv")["login"]
    )
    participants = pd.read_csv(comments_file, encoding="utf_8_sig")
    logins = list(set(list(participants["participant"])) - set(removed_logins))
    pd.DataFrame(data={"login": logins}).to_csv(
        participants_file, encoding="utf_8_sig", index=False
    )

    return participants_file


if __name__ == "__main__":
    # for community in ['test']:
    for community in ["tensorflow", "pytorch"]:
        print(f"------------{community}---------------")
        match_by_common_login_file, no_login_file = developers_matching_by_common_login(
            community_name=community,
            commits_file=f"../../data/raw/commits/{community}.csv",
            hash2login_file=f"../../data/raw/sha/{community}_sha2login.csv",
        )
        matches_file, manual_list_file = developers_matching_by_heuristic(
            community, match_by_common_login_file, no_login_file
        )
        get_manual_matches(community)
        get_developers(
            community_name=community,
            matches_file=f"../../data/preprocessing/{community}_matches.csv",
            manual_matches_file=f"../../data/preprocessing/{community}_manual_matches.csv",
        )
        # get_participants(comments_file=f'data/raw/events/{community}.csv')
        print("\n")
