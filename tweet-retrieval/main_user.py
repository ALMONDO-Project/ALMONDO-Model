from downloadData import UserDataDownload
from downloadData import LimitsExceededError
import math
from datetime import datetime
from utils import *

def read_users(path):
    with open(f'{path}/users_data.json', 'r') as file:
        users_data = json.load(file)
    return users_data

def users_update(users_data, username, path):
    users_data['to_do'].remove(username)
    users_data['done'].append(username)
    with open(f'{path}/users_data.json', 'w') as file:
        json.dump(users_data, file)
    return users_data

def main(count, username): 
    try:
        user_data = UserDataDownload(username=username, count=count)
        user_data.set_limits(max_tweets_per_session=count)
        user_data.set_client()
        user_data.set_user_data()
        user_data.make_dirs()
        user_data.set_paginator()
        user_data.download_user_tweets()
        count = user_data.get_count()
        print(f'tweets left to download:', count)
    except ValueError as e:
        print(e)
        return
    except LimitsExceededError as e:
        print(e)
        return
         
if __name__ == "__main__":
    main(293, 'DeutschePostDHL')