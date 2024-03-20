import os
import tweepy
import json
import requests
import logging
from datetime import datetime, timedelta
import time

# client = tweepy.client(BEARER_TOKEN)

def users_to_do_update(input_data_path: str, log_data_path: str):
    log_message('>>> users to do updating')
    with open(log_data_path+'/users_done.txt', 'r') as lfile:
        lines = lfile.readlines()
        users_done = [line.strip() for line in lines]
    
    with open(input_data_path+'/users_to_do.txt', 'r') as ifile:
        lines = ifile.readlines() 
        users_to_do = [line.strip() for line in lines]
    
    res = filter(lambda i: i not in users_done, users_to_do)
    print(res)
    with open(input_data_path+'/users_to_do.txt', 'w') as ofile:
        users_to_do_string = '\n'.join(res)
        ofile.write(users_to_do_string)
    log_message('>>> users to do updated')
    return 

def read_users(input_data_path: str):
    try:
        with open(f'{input_data_path}/users_to_do.txt', 'r') as ifile:
            lines = ifile.readlines() 
            users_to_do = [line.strip() for line in lines]
    except FileNotFoundError:
        users_to_do = []
    return users_to_do

def compute_max_tweets(bearer_token: str):
    url = 'https://api.twitter.com/2/usage/tweets'
    headers = {
        'Authorization': f'Bearer {bearer_token}'
    }

    response = requests.get(url, headers=headers)
    
    print(response)

    if response.status_code == 200:
        responsedict = response.json()
        # Process the response data here
        max_tweets_left = int(responsedict['data']['project_cap']) - int(responsedict['data']['project_usage']) - 1000
        return max_tweets_left
    else:
        print("Error:", response.status_code)
        
# def get_tweets(client, log_data_path: str, username: str):
#     with open(log_data_path+'logfile.txt', 'a') as file:
#         print(f"retrieving tweets from {username} timeline")
#         file.write(f"START {username}: retrieving tweets from {username} timeline\n")
#         user_id = client.get_user(username=username).data.id
#         responses = tweepy.Paginator(client.get_users_tweets, user_id, start_date = datetime(year=2023, month=1, day=1, hour = 0, minute = 0), end_date = datetime(year=2023, month=12, day=31, hour=23, minute=59), max_results=1, limit=1)
#         counter = 0
#         tweet_list = []
#         for response in responses:
#             counter += 1
#             print(f"==> processing {counter * 100} to {(counter + 1) * 100} of {username}'s tweets")
#             file.write(f"==> processing {counter * 100} to {(counter + 1) * 100} of {username}'s tweets")
#             try:
#                 for tweet in response.data:  # see any individual tweet by id at: twitter.com/anyuser/status/TWEET_ID_HERE
#                     print(tweet)
#                     tweet_list.append(dict(tweet))
#             except Exception as e:
#                 print(e)
#         file.write(f"END {username}: finished retrieving tweets from {username} timeline\n")
#     return tweet_list

# def get_oldest_tweet_id(path):
#     if not os.path.exists(path):
#         return None
#     else:
#         with open(path, 'r') as file:
#             last_tweet = list(json.load(file))[-1]
#         return last_tweet['id']



# Function to log messages to a file
def log_message(message):
    now = time.time()
    LOG_FILE_PATH = "data/log/messages.log"
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(str(now) + "    " + message + "\n")
        
def update_users_done(username):
    print(f'>>> finished with user {username}')
    with open(f'data/log/users_done.txt', 'a') as file:
        file.write(f'@{username}\n')        
    
# Configure logging
logging.basicConfig(filename='data/log/output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Redirect print statements to the logger
class PrintLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass   


    









    


