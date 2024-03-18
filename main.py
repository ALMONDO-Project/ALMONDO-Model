import os
from downloadData import UserDataDownload
from utils import *

def main():
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
    INPUT = 'data/in'
    LOG = 'data/out'
    
    usernames = read_users(INPUT)    
    count = compute_max_tweets(BEARER_TOKEN)
    for username in usernames:
        while count > 0:
            user_data = UserDataDownload(username=username)
            user_data.set_client()
            user_data.set_user_data()
            user_data.make_dirs()
            user_data.set_until_id()
            user_data.download()
            users_to_do_update(INPUT, LOG)
            count = compute_max_tweets(BEARER_TOKEN)
        print('>>> process ended')    
    
    
if __name__ == "__main__":
    main()