import csv
from utils.tool import datetime_to_timestamp
import pandas as pd

def write_to_cache(url, filename):
    with open(filename, 'a+', encoding='UTF8') as f:
        f.write(str(url) + '\n')

def write_issues(source, type, infos):
    save_pos = f'../data/raw/{type}/{source}.csv'
    try:
        with open(save_pos, 'a+', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for item in infos:
                row = [
                    item['No'],
                    item['type'],
                    item['timestamp'],
                    item['author'],
                ]
                if type != 'issues':
                    row.append(item['publisher'])
                writer.writerow(row)
    except:
        return False

    return True

def write_bots(author, filename):
    with open(filename, 'a+', encoding='UTF8') as f:
        f.write(author + '\n')

def write_sha2login(sha, login, filename):
    with open(filename, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([sha, login])

def write_events(issue_id, login, timestamp, type, body, filename):
    with open(filename, 'a+', encoding='utf_8_sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([issue_id, login, timestamp, type, body])

def write_page_num(issue_id, page_num, filename):
    with open(filename, 'a+', encoding='utf_8_sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([issue_id, page_num])

def write_commits(filepath, repo):
    '''
    The function write all commits' information of the community to filepath
    '''
    # for commit in repo.traverse_commits():
    #     if commit.in_main_branch:
    #         print([
    #                 commit.hash,
    #                 datetime_to_timestamp(commit.author_date),
    #                 commit.author.name,
    #                 commit.author.email,
    #                 commit.lines
    #             ])
    with open(filepath, "w+", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['hash', 'timestamp', 'name', 'email', 'LOC'])
        for commit in repo.traverse_commits():
            if commit.in_main_branch:
                writer.writerow([
                    commit.hash,
                    datetime_to_timestamp(commit.author_date),
                    commit.author.name,
                    commit.author.email,
                    commit.lines
                ])

def write_developers(community, authors_list):
    filepath = f"data/profiles/{community}_authors.csv"
    with open(filepath, "w+", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'name', 'email'])
        for i in range(len(authors_list)):
            aliases = authors_list[i]
            for alias in aliases:
                writer.writerow([i, alias[0], alias[1]])