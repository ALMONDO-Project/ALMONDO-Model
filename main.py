from downloadData import UserDataDownload
from utils import *

def main():
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
    INPUT = 'data/in'
    LOG = 'data/log'
    
    usernames = read_users(INPUT)    
    maxt = compute_max_tweets(BEARER_TOKEN)
    count = min(maxt, 3000)
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