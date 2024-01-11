import csv
import sys
sys.path.append("..")
from utils.scraper import async_crawl_bots
import asyncio
from utils.tool import load_csv, save_csv
import pandas as pd
from Levenshtein import distance as lev
from sklearn.cluster import DBSCAN
import numpy as np
import itertools
import pickle
import pkg_resources
import warnings


def async_collect_bots(community, cache_pos, ids):
    issue_path = f'https://github.com/{community}/{community}/issues/'
    pr_path = f'https://github.com/{community}/{community}/pull/'

    urls = []
    for i in ids:
        url = issue_path + str(i)
        urls.append(url)

    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(async_crawl_bots(url, cache_pos, community)) for url in urls]
    tasks = asyncio.gather(*tasks)
    loop.run_until_complete(tasks)
    loop.close()

# --- Text process and feature production ---
def tokenizer(text):
    return text.split(' ')

def jaccard(x, y):
    """
    To tokenize text and compute jaccard disatnce
    """
    x_w = set(tokenizer(x))
    y_w = set(tokenizer(y))
    return (
        len(x_w.symmetric_difference(y_w)) / (len(x_w.union(y_w)) if len(x_w.union(y_w)) > 0 else 1)
    )

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    if len(array) == 0:
        return 0
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def levenshtein(x, y, n=None):
    if n is not None:
        x = x[:n]
        y = y[:n]
    return lev(x, y) / (max(len(x), len(y)) if max(len(x), len(y)) > 0 else 1)

def average_jac_lev(x, y):
    """
    Computes average of jacard and levenshtein for 2 given strings
    """
    return (jaccard(x, y) + levenshtein(x, y)) / 2

def compute_distance(items, distance):
    """
    Computes a distance matrix for given items, using given distance function.
    """
    m = np.zeros((len(items), len(items)))
    enumitems = list(enumerate(items))
    for xe, ye in itertools.combinations(enumitems, 2):
        i, x = xe
        j, y = ye
        d = distance(x, y)
        m[i, j] = m[j, i] = d
    return m

# --- Load model and prediction ---
def get_model():
    warnings.filterwarnings("ignore")
    path = 'model.json'
    filename = pkg_resources.resource_filename(__name__, path)
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    return model

def predict_bots():
    community = 'tensorflow'
    events = load_csv(f'../data/raw/events/{community}.csv')
    events = events[events['type'] == 'commented']
    params = {'func': average_jac_lev, 'source': 'body', 'eps': 0.5}
    model_inputs = pd.DataFrame(columns=['login', 'comments', 'empty comments', 'patterns', 'inequality'])
    model = get_model()
    empties = []
    for i, row in events.iterrows():
        if row['body'] == '' or str(row['body']) == 'nan':
            empties.append(1)
        else:
            empties.append(0)
    events['empty'] = empties
    f = open(f'../data/raw/bots/{community}_predict_bots.csv', 'a+', encoding='utf_8_sig', newline='')
    writer = csv.writer(f)
    group_iterator = events.groupby('actor')

    for author, group in group_iterator:
        group = group.copy()
        clustering = DBSCAN(eps=params['eps'], min_samples=1, metric='precomputed')
        items = compute_distance(getattr(group, params['source']), params['func'])
        clusters = clustering.fit_predict(items)

        comments_num = len(group)
        empty_comments = np.count_nonzero(group['empty'])
        patterns = len(np.unique(clusters))
        ineq = gini(items[np.tril(items).astype(bool)])

        print(author)

        model_inputs.loc[model_inputs.shape[0]] = [author, comments_num, empty_comments, patterns, ineq]
        writer.writerow([author, comments_num, empty_comments, patterns, ineq])

    model_inputs.assign(
        prediction=lambda x: np.where(model.predict(
            x[['comments', 'empty comments', 'patterns', 'inequality']]) == 1, 'Bot', 'Human')
    )

    bot_logins = model_inputs[model_inputs['prediction'] == 'Bot']['login']
    for login in bot_logins:
        print(login)


if __name__ == '__main__':
    # community = 'tensorflow'
    # option = 'cache'
    # root_dir = f'https://github.com/{community}/{community}/commit/'
    # cache_pos = f'log/{community}_bots.txt'
    # async_collect_bots(community, cache_pos, ids=[i for i in range(1, 56964)])
    predict_bots()
