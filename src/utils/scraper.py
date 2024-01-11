import requests
import random
import asyncio
from pyquery import PyQuery
from aiohttp import ClientSession
from requests.adapters import HTTPAdapter
from utils.writter import write_to_cache, write_issues, write_sha2login, write_bots, write_events, write_page_num
from utils.tool import parse_issues, parse_sha
from urllib import parse
import random
import json
import sys

sem = asyncio.Semaphore(10)


def get_users_agent():
    first_num = random.randint(55, 76)
    third_num = random.randint(0, 3800)
    fourth_num = random.randint(0, 140)
    os_type = ['(Windows NT 6.1; WOW64)', '(Windows NT 10.0; WOW64)', '(X11; Linux x86_64)',
               '(Macintosh; Intel Mac OS X 10_14_5)']
    chrome_version = 'Chrome/{}.0.{}.{}'.format(first_num, third_num, fourth_num)
    ua = ' '.join(['Mozilla/5.0', random.choice(os_type), 'AppleWebKit/537.36', '(KHTML, like Gecko)', chrome_version,
                   'Safari/537.36'])
    return ua


def get_html(url):
    s = requests.Session()
    s.trust_env = False
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    s.mount('http://', HTTPAdapter(max_retries=3))
    s.mount('https://', HTTPAdapter(max_retries=3))

    try:
        headers = {'User-Agent': get_users_agent()}
        response = requests.get(url, timeout=(5, 15), headers=headers, proxies=proxies)  # timeout = (connect, read)

        if response.status_code == 200:
            response.close()
            return response.text
        response.close()

    except requests.exceptions.RequestException as e:
        print(e)

    return 'failed'


def get_login(html, source, url, sha):
    doc = PyQuery(html)
    sha, login = parse_sha(doc, sha)
    if login != '':
        write_sha2login(sha, login, f'../data/raw/sha/{source}_sha2login.csv')


def get_issues(html, pr_path, source, url, i, cache_pos):
    doc = PyQuery(html)
    issues_tab = doc('a').filter('#issues-tab').filter('.selected')
    type = 'issue' if issues_tab != [] else 'pr'
    url = url if issues_tab != [] else pr_path + i
    parsed_comments, parsed_issues = parse_issues(type, doc, i)

    save_issues = write_issues(source, 'issues', parsed_issues)
    save_comments = write_issues(source, 'comments', parsed_comments)
    if save_issues == False or save_comments == False:
        print(f'save {url} failed!')
        write_to_cache(url, cache_pos)


def get_bots(html, community):
    doc = PyQuery(html)
    authors = doc('span.Label--secondary:contains("bot")').siblings('a.Link--primary')
    author_set = set()
    for author in authors:
        if author.text != '':
            author_set.add(author.text)
    for a in author_set:
        write_bots(a, filename=f'../data/raw/bots/{community}_logins.txt')


async def async_crawl_sha(url, source, cache_pos):
    headers = {'User-Agent': get_users_agent()}  # user browsing simulation
    proxy = 'http://127.0.0.1:7890'  # selectable

    with(await sem):
        try:
            async with ClientSession() as session:
                async with session.get(url, headers=headers, proxy=proxy) as response:
                    html = await response.read()
                    sha = url.split('/')[-1]
                    print(f'handling {url}...')
                    get_login(html, source, url, sha)
                    print(f"         {url} success!!!")
        except asyncio.TimeoutError as msg:
            print('timeout: ' + str(url))
            write_to_cache(url, cache_pos)
        except Exception as e:
            print(f"{url} failed!! {e}")
            write_to_cache(url, cache_pos)


async def async_crawl_issues(url, issue_path, pr_path, source, cache_pos):
    headers = {'User-Agent': get_users_agent()}  # user browsing simulation
    proxy = 'http://127.0.0.1:7890'  # selectable

    with(await sem):
        try:
            async with ClientSession() as session:
                async with session.get(url, headers=headers, proxy=proxy) as response:
                    html = await response.read()
                    i = url.split('/')[-1]
                    print(f'dealing with {url}...')
                    get_issues(html, pr_path, source, url, i, cache_pos)
                    print(f"crawl and scrape {url} success!!!")
        except asyncio.TimeoutError as msg:
            print('timeout: ' + str(url))
            write_to_cache(url, cache_pos)
        except Exception as e:
            print(f"{url} failed!! {e}")
            write_to_cache(url, cache_pos)


async def async_crawl_bots(url, cache_pos, community):
    headers = {'User-Agent': get_users_agent()}  # user browsing simulation
    proxy = 'http://127.0.0.1:7890'  # selectable

    with(await sem):
        try:
            async with ClientSession() as session:
                async with session.get(url, headers=headers, proxy=proxy) as response:
                    html = await response.read()
                    print(f'dealing with {url}...')
                    get_bots(html, community)
                    print(f"crawl and scrape {url} success!!!")
        except asyncio.TimeoutError as msg:
            print('timeout: ' + str(url))
            write_to_cache(url, cache_pos)
        except Exception as e:
            print(f"{url} failed!! {e}")
            write_to_cache(url, cache_pos)


def get_events(issue_id, url, access_token, proxies, community):
    headers = {'User-Agent': get_users_agent(), 'Authorization': "token " + access_token}
    resp = requests.get(url, timeout=(5, 15), headers=headers, proxies=proxies)  # timeout = (connect, read)
    resp_ = resp.json()
    resp_list = []
    for resp_dic in resp_:
        if not isinstance(resp_dic, dict):
            continue
        event_type = resp_dic['event']
        actor, timestamp = '', ''
        if event_type == 'reviewed' and resp_dic.get('user') and resp_dic['user'].get('login'):
            actor = resp_dic['user']['login']
            timestamp = resp_dic['submitted_at']
        elif resp_dic.get('actor') and resp_dic['actor'].get('login'):
            actor = resp_dic['actor']['login']
            timestamp = resp_dic['created_at']

        if actor == '' or timestamp == '':
            continue

        obj = {
            'issue_id': issue_id,
            'participant': actor,
            'timestamp': timestamp,
            'type': event_type,
            'body': '',
        }
        if resp_dic.get('body'):
            obj['body'] = resp_dic['body']
        resp_list.append(obj)
    return resp_list


def get_page_num(issue_id, community, access_token, proxies):
    url = f'https://api.github.com/repos/{community}/{community}/issues/' + str(issue_id) + '/timeline'
    headers = {'User-Agent': get_users_agent(), 'Authorization': "token " + access_token}
    resp = requests.get(url, timeout=(5, 15), headers=headers, proxies=proxies,
                        params={'per_page': 100})  # timeout = (connect, read)
    if 'last' not in resp.links.keys():
        if isinstance(resp.json(), dict) and 'message' in resp.json().keys():
            return 0
        return 1
    last_url = resp.links['last']['url']
    page = parse.parse_qs(parse.urlparse(last_url).query)['page'][0]
    return page


async def async_crawl_events(issue_id, page, cache_pos, community, access_token, writer):
    s = requests.Session()
    s.trust_env = False
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    s.mount('http://', HTTPAdapter(max_retries=3))
    s.mount('https://', HTTPAdapter(max_retries=3))
    url = f'https://api.github.com/repos/{community}/{community}/issues/' + str(
        issue_id) + '/timeline?per_page=100&page=' + str(page)
    headers = {'User-Agent': get_users_agent(), 'Authorization': "token " + access_token}
    with (await sem):
        try:
            async with ClientSession() as session:
                async with session.get(url, headers=headers, proxy=proxies['https']) as resp:
                    resp_json = await resp.json()
                    resp_list = []
                    for resp_dic in resp_json:
                        if not isinstance(resp_dic, dict):
                            continue
                        event_type = resp_dic['event']
                        actor, timestamp = '', ''
                        if event_type == 'reviewed' and resp_dic.get('user') and resp_dic['user'].get('login'):
                            actor = resp_dic['user']['login']
                            timestamp = resp_dic['submitted_at']
                        elif resp_dic.get('actor') and resp_dic['actor'].get('login'):
                            actor = resp_dic['actor']['login']
                            timestamp = resp_dic['created_at']

                        if actor == '' or timestamp == '':
                            continue

                        obj = {
                            'issue_id': issue_id,
                            'participant': actor,
                            'timestamp': timestamp,
                            'type': event_type,
                            'body': '',
                        }
                        if resp_dic.get('body'):
                            obj['body'] = resp_dic['body']
                        resp_list.append(obj)

                    for obj in resp_list:
                        writer.writerow([obj['issue_id'], obj['participant'], obj['timestamp'], obj['type'], obj['body']])
                    print(f'crawl {url} success!!')

        except asyncio.TimeoutError as msg:
            print('timeout: ' + str(url) + '!!')
            write_to_cache(url, filename=cache_pos)

        except Exception as e:
            print(f'{e}, failed to crawl {url}!!')
            write_to_cache(url, filename=cache_pos)


async def async_crawl_events_page_num(issue_id, cache_pos, community, access_token, writer):
    s = requests.Session()
    s.trust_env = False
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    s.mount('http://', HTTPAdapter(max_retries=3))
    s.mount('https://', HTTPAdapter(max_retries=3))
    url = f'https://api.github.com/repos/{community}/{community}/issues/' + str(issue_id) + '/timeline?per_page=100'
    headers = {'User-Agent': get_users_agent(), 'Authorization': "token " + access_token}
    with (await sem):
        try:
            async with ClientSession() as session:
                async with session.get(url, headers=headers, proxy=proxies['https'], timeout=10) as resp:
                    resp_json = await resp.json()
                    resp_links = resp.links

                    if 'last' not in resp_links.keys():
                        if isinstance(resp_json, dict) and 'message' in resp_json.keys():
                            page_num = 0
                        else:
                            page_num = 1
                    else:
                        last_url = str(resp_links['last']['url'])
                        page_num = int(parse.parse_qs(parse.urlparse(last_url).query)['page'][0])
                        # print(issue_id, last_url, page_num)

                    if page_num > 0:
                        writer.writerow([issue_id, page_num])
                        print(f'crawl {issue_id} success!!')
        except Exception as e:
            print(f'failed to crawl {issue_id}, because of {e}!!')
            write_to_cache(issue_id, filename=cache_pos)
