import sys
sys.path.append("..")
from utils.scraper import async_crawl_events, async_crawl_events_page_num
from utils.tool import load_list_via_txt
import asyncio
import time
import random
import pandas as pd
import csv

tokens = [
    [
        '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',
        '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',
        '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',
        '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',
        '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',
    ],
    [
        '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',
        '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',
        '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',
    ],
    [
        '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',
        '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',
        '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',
        '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',
        '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'
    ],
    [
        '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'
    ]
]


def async_collect_events_page_num(community, cache_pos, ids, writer):
    loop = asyncio.get_event_loop()
    tasks = []
    j = 0
    for issue_id in ids:
        access_token = tokens[j][random.randint(0, len(tokens[j]) - 1)]
        j = (j + 1) % 3
        tasks.append(asyncio.ensure_future(async_crawl_events_page_num(issue_id, cache_pos, community, access_token, writer)))
    tasks = asyncio.gather(*tasks)
    loop.run_until_complete(tasks)


def async_collect_events(community, cache_pos, ids, id2num, writer):
    loop = asyncio.get_event_loop()
    tasks = []
    j = 0
    for issue_id in ids:
        if not id2num.get(issue_id):
            page = 1
        else:
            page = id2num[issue_id]

        for p in range(1, page + 1):
            access_token = tokens[j][random.randint(0, len(tokens[j]) - 1)]
            j = (j + 1) % 3
            tasks.append(asyncio.ensure_future(async_crawl_events(issue_id, p, cache_pos, community, access_token, writer)))
    tasks = asyncio.gather(*tasks)
    loop.run_until_complete(tasks)

def async_collect_events_cache(community, cache_pos, writer):
    loop = asyncio.get_event_loop()
    tasks = []
    urls = load_list_via_txt(cache_pos)
    with open(cache_pos, 'a+', encoding='utf_8_sig') as f:
        f.truncate(0)
    j = 0
    for url in urls:
        issue_id = int(url.split('/')[-2])
        p = int(url.split('=')[-1])
        j = (j + 1) % 3
        access_token = tokens[j][random.randint(0, len(tokens[j]) - 1)]
        tasks.append(asyncio.ensure_future(async_crawl_events(issue_id, p, cache_pos, community, access_token, writer)))
    tasks = asyncio.gather(*tasks)
    loop.run_until_complete(tasks)


if __name__ == '__main__':
    community = 'tensorflow'
    option = 'cache'       # events or page or cache

    filename = ''
    start_id, end_id = -1, -1

    if option == 'events':
        start_id = 20001
        end_id = 56963

    i = 0
    maxlen = 100
    ids = [i for i in range(start_id, end_id + 1)]

    id2num = pd.read_csv(f'../data/raw/events/{community}_page_num.csv')
    id2num = dict(zip(id2num['id'], id2num['page_num']))

    if option == 'events' or option == 'cache':
        filename = f'../data/raw/events/{community}.csv'
    elif option == 'page':
        filename = f'../data/raw/events/{community}_page_num.csv'

    f = open(filename, 'a+', encoding='utf_8_sig', newline='')
    writer = csv.writer(f)

    if option == 'cache':
        async_collect_events_cache(community, cache_pos=f'./log/{community}_events.txt', writer=writer)
    else:
        while i < len(ids):
            st_time = time.time()
            next_i = min(len(ids), i + maxlen)
            sub_ids = ids[i:next_i]
            print(f'{ids[i]} ~ {ids[next_i - 1]} started.')

            # async_collect_events_page_num(community, f'./log/{community}_events_pagenum.txt', sub_ids)
            async_collect_events(community=community, cache_pos=f'./log/{community}_events.txt', ids=sub_ids, id2num=id2num, writer=writer)
            end_time = time.time()
            print(f'{ids[i]} ~ {ids[next_i - 1]} finished, it takes {end_time - st_time} seconds, now sleep 5 seconds.')

            time.sleep(5)
            i = next_i
    asyncio.get_event_loop().close()




