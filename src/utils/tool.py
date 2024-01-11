import datetime
import json
import pandas as pd
import re
import csv

def load_config():
    with open('../config.json', 'r') as file:
        config = json.load(file)
    return config


def load_csv(file):
    return pd.read_csv(file, keep_default_na=False, encoding='utf_8_sig', quotechar='"')


def save_csv(df, file):
    df.to_csv(file, encoding='utf_8_sig', index=False)


def parse_timestamp(timestamp):
    # YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD or YYYY.MM.DD
    if not isinstance(timestamp, str):
        return datetime.datetime(1970, 1, 1)

    if not re.match('^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', timestamp) \
            and not re.match('\d{4}.\d{2}.\d{2}$', timestamp) \
            and not re.match('\d{4}-\d{2}-\d{2}$', timestamp):
        return datetime.datetime(1970, 1, 1)

    year, month, day = int(timestamp[:4]), int(timestamp[5:7]), int(timestamp[8:10])
    if len(timestamp) == 20:
        hour, minute, second = int(timestamp[11:13]), int(timestamp[14:16]), int(timestamp[17:19])
        return datetime.datetime(year, month, day, hour, minute, second)
    else:
        return datetime.datetime(year, month, day)


def datetime_to_timestamp(dtime):
    utc_time = datetime.datetime.utcfromtimestamp(dtime.timestamp())
    return utc_time.strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_issues(type, doc, id):
    # parse relative html to formatted list, each item is a dict, represents a comment
    parsed_comments = []

    # get publish author
    publish_date = doc('.gh-header-meta relative-time').attr('datetime')

    # get author, comments and reviews
    issue_comments = doc('.timeline-comment').items()
    pr_comments = doc('[id^="pullrequestreview-"]').items()
    publishers, timestamps = [], []
    for comment in issue_comments:
        publishers.append(comment('a').filter('.author').text())
        timestamps.append(comment('relative-time').attr('datetime'))
    comments = pd.DataFrame(data={'publisher': publishers, 'timestamp': timestamps})
    comments = comments.sort_values(by='timestamp', kind='mergesort')
    author = comments.iloc[0]['publisher']


    for idx, comment in comments.iterrows():
        publisher = comment['publisher']
        timestamp = comment['timestamp']
        if not timestamp or not publisher:
            continue

        if not publish_date:
            publish_date = timestamp

        comment_item = {
            'author': author,
            'publisher': publisher,
            'timestamp': timestamp,
            'No': id,
            'type': type,
        }
        parsed_comments.append(comment_item)

    for review in pr_comments:
        publisher_ = review('a').filter('.author').items()
        publisher = ""
        for p in publisher_:
            publisher = p.text()
            break  # in case publisher duplication

        timestamp = review('relative-time').attr('datetime')
        if not timestamp or not publisher:
            continue

        if not publish_date:
            publish_date = timestamp

        review_item = {
            'No': id,
            'type': type,
            'timestamp': timestamp,
            'author': author,
            'publisher': publisher,
        }
        parsed_comments.append(review_item)

    global_info = {
        'No': id,
        'type': type,
        'timestamp': publish_date,
        'author': author,
    }

    return parsed_comments, [global_info]


def parse_sha(doc, sha):
    a_content = doc('a').filter('.commit-author')
    span_content = doc('span').filter('.commit-author')
    login = ''
    if len(a_content) == 2 or (len(a_content) == 1 and len(span_content) == 0):
        login = a_content[0].text
    return sha, login


def load_list_via_txt(file):
    lis = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lis.append(line.strip())
    return lis


def build_releases_windows(community, release2time_file, three_month_restriction=False):
    from datetime import datetime, timedelta
    release2time = load_csv(release2time_file)
    release_times = release2time['timestamp']
    releases = release2time['releases']
    config = load_config()
    start_date = parse_timestamp(config[community]['start_date'])
    start_date = start_date.strftime('%Y.%m.%d')
    # release_times = release_times.apply(lambda x: parse_timestamp(x))
    
    time_windows = []
    for timestamp in release_times:
        end_date = timestamp
        time_window = [start_date, end_date]
        time_windows.append(time_window)
        date = datetime.strptime(end_date, '%Y.%m.%d')
        new_date = date + timedelta(days=1)
        start_date = new_date.strftime('%Y.%m.%d')
        
    # i = 1
    # while i < len(release_times):
    #     last_time = release_times[i - 1]
    #     last_release = releases[i - 1]
    #     if three_month_restriction:
    #         while parse_timestamp(release_times[i]) - parse_timestamp(last_time) <= datetime.timedelta(days=90):
    #             i += 1
    #     time_windows.append([last_time, release_times[i]])
    #     labels.append(releases[i])
    #     # dates.append(release_times[i])
    #     i += 1

    return releases, time_windows, release_times

def build_months_windows(community):
    '''
    :param community:
    :return: [(label, time_window)]
    '''
    config = load_config()
    start_date = parse_timestamp(config[community]['start_date'])
    end_date = parse_timestamp(config[community]['end_date'])

    st_month = max(1, (start_date.month + 1) % 13)
    st_year = start_date.year if start_date.month != 12 else start_date.year + 1

    ed_month = end_date.month
    ed_year = end_date.year

    y, m = st_year, st_month
    labels, time_windows, dates = [], [], []
    while not (y == ed_year and m == ed_month):
        nm = max(1, (m + 1) % 13)
        ny = y + 1 if nm == 1 else y
        labels.append('{:0>4d}-{:0>2d}'.format(y, m))
        time_windows.append(['{:0>4d}-{:0>2d}-01'.format(y, m), '{:0>4d}-{:0>2d}-01'.format(ny, nm)])
        dates.append('{:0>4d}-{:0>2d}-01'.format(y, m))
        y, m = ny, nm

    return labels, time_windows, dates

def name_preprocess(name):
    ret = " ".join(name.split())
    return ret.lower()


def serialize_author_id(df):
    author_id_set = list(set(list(df['id'])))
    id_mapping = dict(zip(author_id_set, [i for i in range(len(author_id_set))]))
    df['id'] = df['id'].map(id_mapping)
    return df


if __name__ == '__main__':
    names, windows = build_months_windows('pytorch')