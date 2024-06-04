from downloadData import UserDataDownload
import math
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

def main():
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
    DATA='data'
    users_data = read_users(DATA)
    usernames = users_data['to_do']
    for username in usernames:
        maxt = compute_max_tweets(BEARER_TOKEN)
        count = min(2500, maxt)    
        try:
            while count > 0:
                user_data = UserDataDownload(username=username)
                user_data.set_limits(max_tweets_per_session=count)
                user_data.download_user_tweets()
                count = min(2500, compute_max_tweets(BEARER_TOKEN))  
        except ValueError as e:
            print(e)
            continue
        
if __name__ == "__main__":
    main()