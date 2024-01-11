import pandas as pd
from pydriller import Repository
import sys
sys.path.append("..")
from utils.tool import load_config, parse_timestamp
from utils.writter import write_commits

if __name__ == '__main__':
    community = 'tensorflow'
    config = load_config()
    repo = Repository(
        path_to_repo=f'https://github.com/{community}/{community}',
        since=parse_timestamp(config[community]['start_date']),
        to=parse_timestamp(config[community]['end_date']),
    )
    filepath = f"../data/raw/commits/{community}.csv"
    write_commits(filepath=filepath, repo=repo)
    df = pd.read_csv(filepath)
    df.to_csv(filepath, encoding='utf_8_sig', index=False)