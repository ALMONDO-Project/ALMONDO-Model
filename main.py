from downloadData import UserDataDownload
from utils import *
import sys
import logging

# Replace sys.stdout and sys.stderr with PrintLogger
sys.stdout = PrintLogger(logging.getLogger('stdout'), logging.INFO)
sys.stderr = PrintLogger(logging.getLogger('stderr'), logging.ERROR)

def main():
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
    INPUT = 'data/in'
    LOG = 'data/log'
    
    usernames = read_users(INPUT)    
    maxt = compute_max_tweets(BEARER_TOKEN)
    count = maxt
    log_message(f'>>> the number of tweets left is {count}')
    print(f'>>> the number of tweets left is {count}')
    for username in usernames:
        while count > 0:
            user_data = UserDataDownload(username=username)
            user_data.set_client()
            user_data.set_user_data()
            user_data.make_dirs()
            user_data.download(count)
            users_to_do_update(INPUT, LOG)
            count = compute_max_tweets(BEARER_TOKEN)
        print('>>> process ended')    
    
if __name__ == "__main__":
    main()