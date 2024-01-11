from fuzzywuzzy import fuzz
from utils.tool import name_preprocess

def compute_similarity(x, y):
    '''
    whether developer x is similar to developer y
    '''
    name1, email1 = name_preprocess(x['name']), str(x['email'])
    name2, email2 = name_preprocess(y['name']), str(y['email'])

    score_name = 0 if len(name1) == 0 and len(name2) == 0 else fuzz.token_sort_ratio(name1, name2)
    score_email = 0
    if email1 != 'nan' and email2 != 'nan':
        email_local1, email_local2 = email1.split('@')[0], email2.split('@')[0]
        score_email = fuzz.token_sort_ratio(email_local1, email_local2)
    return max(score_name, score_email)
