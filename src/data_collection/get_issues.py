import sys
sys.path.append("..")
from utils.scraper import async_crawl_issues
import asyncio
import pandas as pd


def async_collect_issues(community, cache_pos, ids):
    issue_path = f'https://github.com/{community}/{community}/issues/'
    pr_path = f'https://github.com/{community}/{community}/pull/'

    urls = []
    for i in ids:
        url = issue_path + str(i)
        urls.append(url)

    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(async_crawl_issues(url, issue_path, pr_path, community, cache_pos)) for url in urls]
    tasks = asyncio.gather(*tasks)
    loop.run_until_complete(tasks)
    loop.close()


if __name__ == '__main__':
    community = 'tensorflow'
    option = 'cache'
    root_dir = f'https://github.com/{community}/{community}/commit/'
    cache_pos = f'log/{community}_issues.txt'
    async_collect_issues(community, cache_pos, ids=[1])
