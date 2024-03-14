import os
from downloadData import UserDataDownloader
from utils import *

def main():
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
    home = os.getcwd()
    usernames = read_users(f'{home}/data/in/')
    
    for username in usernames:   
        print(f'>>> process started for {username}')
        try:
            username = username.replace('@', '')     
            downloader = UserDataDownloader(username)
            downloader.make_dirs()
            downloader.set_since_id()
            downloader.set_max_tweets(BEARER_TOKEN, 5)
            downloader.download()
            downloader.save_tweets()
            print(f'>>> process ended for user {username}')
            users_done_update(f'{home}/data/log/', username)
            users_to_do_update(f'{home}/data/in/', f'{home}/data/log/')
        except:
            print(f'>>> process had some problems for user {username}')
            continue
        
    print('>>> process ended')
    
if __name__ == '__main__':
    main()
    