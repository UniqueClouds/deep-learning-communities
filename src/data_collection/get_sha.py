import sys
sys.path.append("..")
from utils.scraper import async_crawl_sha
import pandas as pd
import asyncio

def async_collect_sha(community, urls, cache_pos):
    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(async_crawl_sha(url, community, cache_pos)) for url in urls]
    tasks = asyncio.gather(*tasks)
    loop.run_until_complete(tasks)
    loop.close()

if __name__ == '__main__':
    community = 'pytorch'
    option = 'cache'
    root_dir = f'https://github.com/{community}/{community}/commit/'
    urls = []
    cache_pos = f'log/{community}_sha.txt'

    if option == 'crawl':
        commits = pd.read_csv(f'../data/raw/commits/{community}.csv')
        sha_list = list(commits['hash'])
        for sha in sha_list:
            url = root_dir + sha
            urls.append(url)


    elif option == 'cache':
        urls = []
        cache_pos = f'log/{community}_sha_.txt'
        with open(f'log/{community}_sha.txt', 'r', encoding='UTF8') as f:
            url = f.readline().replace('\n', '')
            while url:
                urls.append(url)
                url = f.readline().replace('\n', '')

    async_collect_sha(community, urls, cache_pos)